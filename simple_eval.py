#!/usr/bin/env python3
"""
Classical ML Models Evaluation with 5-Fold Cross-Validation

Simple evaluation script for SVM and Random Forest models with:
- 5-fold stratified cross-validation
- Comprehensive metrics
- Feature importance (for RF)
- Statistical analysis

Author: DILANJAN DK
Email: DDIYABAL@UWO.CA
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (classification_report, confusion_matrix, 
                            balanced_accuracy_score, f1_score,
                            make_scorer, accuracy_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from scipy import stats

__author__ = "DILANJAN DK"
__email__ = "DDIYABAL@UWO.CA"

# ============================================================================
# 5-FOLD CROSS-VALIDATION FOR CLASSICAL ML
# ============================================================================

def perform_5fold_cv(model, X: np.ndarray, y: np.ndarray, 
                    model_name: str, cv: int = 5) -> Dict:
    """
    Perform 5-fold stratified cross-validation.
    
    Args:
        model: Scikit-learn model
        X: Feature matrix
        y: Labels
        model_name: Name of the model
        cv: Number of folds
    
    Returns:
        Dictionary with detailed CV results
    """
    print(f"\n{'='*70}")
    print(f" 5-FOLD CROSS-VALIDATION: {model_name}")
    print(f"{'='*70}")
    
    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
        'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
        'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
        'recall_macro': make_scorer(recall_score, average='macro', zero_division=0)
    }
    
    # Perform cross-validation
    print(f"Running {cv}-fold cross-validation...")
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                               return_train_score=True, n_jobs=-1, verbose=1)
    
    # Detailed fold-by-fold analysis
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    fold_cms = []
    fold_predictions = []
    fold_targets = []
    
    print(f"\n{'='*70}")
    print(" FOLD-BY-FOLD RESULTS")
    print(f"{'='*70}")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Scale if needed (for SVM)
        if model_name.lower() == 'svm':
            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X_train_fold)
            X_val_fold = scaler.transform(X_val_fold)
        
        # Train and predict
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        
        # Calculate metrics
        acc = accuracy_score(y_val_fold, y_pred)
        bal_acc = balanced_accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro', zero_division=0)
        
        # Store results
        cm = confusion_matrix(y_val_fold, y_pred)
        fold_cms.append(cm)
        fold_predictions.append(y_pred)
        fold_targets.append(y_val_fold)
        
        print(f"\nFold {fold_idx}:")
        print(f"  Samples: Train={len(y_train_fold)}, Val={len(y_val_fold)}")
        print(f"  Accuracy:          {acc:.4f}")
        print(f"  Balanced Accuracy: {bal_acc:.4f}")
        print(f"  Macro F1-Score:    {f1:.4f}")
    
    # Statistical Summary
    print(f"\n{'='*70}")
    print(" CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    results_summary = {}
    
    for metric in ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted', 
                  'precision_macro', 'recall_macro']:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        test_mean = np.mean(test_scores)
        test_std = np.std(test_scores)
        train_mean = np.mean(train_scores)
        
        results_summary[metric] = {
            'test_mean': float(test_mean),
            'test_std': float(test_std),
            'test_scores': test_scores.tolist(),
            'train_mean': float(train_mean),
            'train_scores': train_scores.tolist()
        }
        
        print(f"{metric:20s}: {test_mean:.4f} ± {test_std:.4f} (test) | {train_mean:.4f} (train)")
    
    # Check for overfitting
    overfit_margin = results_summary['accuracy']['train_mean'] - results_summary['accuracy']['test_mean']
    if overfit_margin > 0.1:
        print(f"\n⚠ WARNING: Potential overfitting detected!")
        print(f"  Train-Test accuracy gap: {overfit_margin:.4f}")
    else:
        print(f"\n✓ Good generalization (train-test gap: {overfit_margin:.4f})")
    
    # Average confusion matrix
    avg_cm = np.mean(fold_cms, axis=0)
    
    return {
        'cv_results': cv_results,
        'results_summary': results_summary,
        'fold_cms': fold_cms,
        'avg_cm': avg_cm.tolist(),
        'fold_predictions': fold_predictions,
        'fold_targets': fold_targets,
        'overfitting_margin': float(overfit_margin)
    }

# ============================================================================
# FEATURE IMPORTANCE (FOR RANDOM FOREST)
# ============================================================================

def analyze_feature_importance(model, feature_names: List[str] = None, 
                              top_k: int = 20, save_path: str = None):
    """
    Analyze and visualize feature importance for Random Forest.
    
    Args:
        model: Trained RandomForestClassifier
        feature_names: Names of features
        top_k: Number of top features to display
        save_path: Path to save plot
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    print(f"\n{'='*70}")
    print(" FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*70}")
    
    importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    print(f"\nTop {top_k} Most Important Features:")
    print(f"{'Rank':<6} {'Feature':<30} {'Importance':<12}")
    print("-" * 50)
    
    for i in range(min(top_k, len(indices))):
        idx = indices[i]
        print(f"{i+1:<6} {feature_names[idx]:<30} {importances[idx]:<12.6f}")
    
    # Plot
    plt.figure(figsize=(12, 8))
    top_indices = indices[:top_k]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    plt.barh(range(top_k), top_importances, align='center', alpha=0.8, color='steelblue')
    plt.yticks(range(top_k), top_names)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_k} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Feature importance plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'importances': importances.tolist(),
        'top_features': [(feature_names[i], float(importances[i])) for i in indices[:top_k]]
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_cv_metrics(results_summary: Dict, model_name: str, save_path: str = None):
    """Plot cross-validation metrics across folds."""
    metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted', 
              'precision_macro', 'recall_macro']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        scores = results_summary[metric]['test_scores']
        mean_score = results_summary[metric]['test_mean']
        std_score = results_summary[metric]['test_std']
        
        folds = range(1, len(scores) + 1)
        
        axes[idx].bar(folds, scores, alpha=0.7, color='steelblue', edgecolor='black')
        axes[idx].axhline(mean_score, color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {mean_score:.4f} ± {std_score:.4f}')
        axes[idx].fill_between(folds, mean_score - std_score, mean_score + std_score, 
                              alpha=0.2, color='red')
        axes[idx].set_xlabel('Fold', fontsize=11)
        axes[idx].set_ylabel('Score', fontsize=11)
        axes[idx].set_title(f'{metric.replace("_", " ").title()}', 
                          fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
        axes[idx].set_ylim([max(0, mean_score - 3*std_score), min(1, mean_score + 3*std_score)])
    
    plt.suptitle(f'{model_name} - Cross-Validation Results', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ CV metrics plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_avg_confusion_matrix(avg_cm: np.ndarray, class_names: List[str], 
                             model_name: str, save_path: str = None):
    """Plot average confusion matrix across folds."""
    plt.figure(figsize=(10, 8))
    
    # Normalize to percentages
    cm_percent = avg_cm / avg_cm.sum(axis=1, keepdims=True) * 100
    
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'})
    
    plt.title(f'{model_name} - Average Confusion Matrix (5-Fold CV)', 
             fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_train_test_comparison(results_summary: Dict, model_name: str, save_path: str = None):
    """Plot train vs test performance comparison."""
    metrics = ['accuracy', 'balanced_accuracy', 'f1_macro']
    train_means = [results_summary[m]['train_mean'] for m in metrics]
    test_means = [results_summary[m]['test_mean'] for m in metrics]
    test_stds = [results_summary[m]['test_std'] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, train_means, width, label='Train', 
                   alpha=0.8, color='lightgreen', edgecolor='black')
    bars2 = ax.bar(x + width/2, test_means, width, label='Test', 
                   alpha=0.8, color='lightcoral', edgecolor='black')
    
    # Add error bars for test
    ax.errorbar(x + width/2, test_means, yerr=test_stds, fmt='none', 
               color='black', capsize=5, linewidth=2)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{model_name} - Train vs Test Performance', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Train-test comparison saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Classical ML models with 5-fold CV',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model (.pkl file)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data (.npz file with X and y)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                       help='Class names for visualization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(" CLASSICAL ML MODEL EVALUATION")
    print(f"{'='*70}")
    print(f"Author: {__author__}")
    print(f"Email: {__email__}")
    
    # Set seed
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    with open(args.model_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    model = checkpoint['model']
    model_name = checkpoint['model_type']
    print(f"✓ Loaded model: {model_name}")
    
    # Load data
    print(f"\nLoading data from: {args.data_path}")
    data = np.load(args.data_path)
    X = data['X']
    y = data['y']
    
    if X.ndim == 3:  # (samples, channels, time)
        print(f"  Reshaping data from {X.shape} to (samples, features)")
        X = X.reshape(X.shape[0], -1)
    
    print(f"✓ Loaded data: X={X.shape}, y={y.shape}")
    print(f"  Number of samples: {len(y)}")
    print(f"  Number of features: {X.shape[1]}")
    print(f"  Number of classes: {len(np.unique(y))}")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples ({count/len(y)*100:.2f}%)")
    
    # Perform 5-fold CV
    cv_results = perform_5fold_cv(model, X, y, model_name, cv=args.cv_folds)
    
    # Generate class names if not provided
    if args.class_names is None:
        args.class_names = [f"Class_{i}" for i in range(len(np.unique(y)))]
    
    # Visualizations
    print(f"\n{'='*70}")
    print(" GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    # 1. CV metrics plot
    plot_cv_metrics(cv_results['results_summary'], model_name,
                   save_path=os.path.join(args.output_dir, f'{model_name}_cv_metrics.png'))
    
    # 2. Average confusion matrix
    plot_avg_confusion_matrix(np.array(cv_results['avg_cm']), args.class_names, model_name,
                             save_path=os.path.join(args.output_dir, f'{model_name}_confusion_matrix.png'))
    
    # 3. Train-test comparison
    plot_train_test_comparison(cv_results['results_summary'], model_name,
                              save_path=os.path.join(args.output_dir, f'{model_name}_train_test.png'))
    
    # 4. Feature importance (if Random Forest)
    if model_name == 'RandomForestClassifier':
        feature_importance = analyze_feature_importance(
            model, 
            save_path=os.path.join(args.output_dir, f'{model_name}_feature_importance.png')
        )
        cv_results['feature_importance'] = feature_importance
    
    # Save results to JSON
    results_path = os.path.join(args.output_dir, f'{model_name}_cv_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'model_name': model_name,
            'cv_folds': args.cv_folds,
            'data_shape': X.shape,
            'n_samples': len(y),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'class_names': args.class_names,
            'results': cv_results['results_summary'],
            'overfitting_margin': cv_results['overfitting_margin']
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    print(f"\n{'='*70}")
    print(" EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults summary:")
    print(f"  Test Accuracy: {cv_results['results_summary']['accuracy']['test_mean']:.4f} ± {cv_results['results_summary']['accuracy']['test_std']:.4f}")
    print(f"  Test Balanced Accuracy: {cv_results['results_summary']['balanced_accuracy']['test_mean']:.4f} ± {cv_results['results_summary']['balanced_accuracy']['test_std']:.4f}")
    print(f"  Test Macro F1: {cv_results['results_summary']['f1_macro']['test_mean']:.4f} ± {cv_results['results_summary']['f1_macro']['test_std']:.4f}")
    print(f"\n  All visualizations and results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
    """python eval_classical_ml.py \
  --model_path ./saved_models/none_vs_pain/20250120_143000/classical_ml/svm_best.pkl \
  --data_path ./test_data.npz \
  --output_dir ./eval_results/svm \
  --cv_folds 5"""
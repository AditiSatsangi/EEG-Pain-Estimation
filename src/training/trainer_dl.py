# src/training/trainer_dl.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

def train_one_model(model, train_loader, val_loader, class_weights, device,
                    lr=1e-3, weight_decay=1e-5, epochs=30, patience=10):

    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1 = 0.0
    best_state = None
    no_improve = 0

    for ep in range(1, epochs+1):
        model.train()
        for X, W, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X, W, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                preds.append(out.argmax(1).cpu().numpy())
                targets.append(y.cpu().numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        f1 = f1_score(targets, preds, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_f1

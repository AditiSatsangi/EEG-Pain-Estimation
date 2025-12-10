# src/models/cnn_baselines.py
import torch.nn as nn
import torch

class CNN2D(nn.Module):
    def __init__(self, n_channels, n_time, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(32, 64, (3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
        )
        self.fc = nn.Linear(64 * n_channels * (n_time//4), n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)

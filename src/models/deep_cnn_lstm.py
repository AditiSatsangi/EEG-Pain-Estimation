# src/models/deep_cnn_lstm.py
import torch.nn as nn
import torch

class DeepCNN_LSTM(nn.Module):
    def __init__(self, n_channels, n_time, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (3,3), padding=1), nn.ReLU(),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(32, 64, (3,3), padding=1), nn.ReLU(),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(64,128,(3,3), padding=1), nn.ReLU(),
        )

        reduced_time = n_time // 4
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)       # (B,128,C,T)
        x = x.mean(dim=2)      # reduce channels
        x = x.transpose(1,2)
        _, (h,_) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(h)

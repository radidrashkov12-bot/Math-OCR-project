import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=256): #!
        super().__init__()

            # Feature extractor: 4 convolutional blocks
        self.features = nn.Sequential(
        # Block 1: [B, 1, 64, 256] -> [B, 64, 32, 128]
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: [B, 64, 32, 128] -> [B, 128, 16, 64]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: [B, 128, 16, 64] -> [B, 256, 8, 64] (Downsample height only)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # Block 4: [B, 256, 8, 64] -> [B, 512, 4, 64] (Downsample height only)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
    )
        self.projection = nn.Linear(512*4, hidden_dim)

    def forward(self, x):
        features = self.features(x)
        B,C,H,W = features.shape
        features = features.view(B, C*H, W)
        features = features.permute(0,2,1)
        encoded_sequence = self.projection(features)
        return encoded_sequence
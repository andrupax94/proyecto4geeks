import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class AudioStreamCNN(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class AudioDualStreamClassifier(nn.Module):
    def __init__(self, num_classes: int, mel_ch: int, mfcc_ch: int):
        super().__init__()

        self.mel_branch = AudioStreamCNN(mel_ch)
        self.mfcc_branch = AudioStreamCNN(mfcc_ch)

        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_mel, x_mfcc):
        mel_feat = self.mel_branch(x_mel)
        mfcc_feat = self.mfcc_branch(x_mfcc)

        x = torch.cat([mel_feat, mfcc_feat], dim=1)
        return self.classifier(x)
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AudioCNNClassifier(nn.Module):
    """CNN pura para audio sobre mel spectrogram.

    Diseñada para evitar rutas CRNN/RNN y minimizar el uso de operaciones
    problemáticas en MIOpen/ROCm en Windows.
    """

    def __init__(self, num_classes: int, input_shape: Tuple[int, int, int]):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("input_shape debe ser (channels, mel_bins, frames)")

        channels, mel_bins, frames = input_shape
        self.features = nn.Sequential(
            ConvBlock(channels, 32, dropout=0.10),
            ConvBlock(32, 64, dropout=0.15),
            ConvBlock(64, 128, dropout=0.20),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, mel_bins, frames)
            feat = self.features(dummy)
            flat_dim = feat.flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.features(mel)
        return self.classifier(x)

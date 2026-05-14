import torch
import torch.nn as nn


class Conv2DBranch(nn.Module):
    def __init__(self, in_channels: int = 1, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        # x: [B, F, T] -> [B, 1, F, T]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.net(x)


class Conv1DBranch(nn.Module):
    def __init__(self, in_channels: int = 1, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        # x: [B, T] -> [B, 1, T]
        if x.dim() == 3:
            # Si viene como [B, C, T], colapsar canales si hace falta
            if x.size(1) > 1:
                x = x.mean(dim=1)
            else:
                x = x.squeeze(1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.net(x)


class HybridAudioClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mode: str = "mel_only",
        branch_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.mode = mode

        if mode not in {"mel_only", "mel_mfcc", "all_three"}:
            raise ValueError("mode debe ser: mel_only, mel_mfcc o all_three")

        self.use_mel = mode in {"mel_only", "mel_mfcc", "all_three"}
        self.use_mfcc = mode in {"mel_mfcc", "all_three"}
        self.use_waveform = mode == "all_three"

        if self.use_mel:
            self.mel_branch = Conv2DBranch(out_dim=branch_dim)
        if self.use_mfcc:
            self.mfcc_branch = Conv2DBranch(out_dim=branch_dim)
        if self.use_waveform:
            self.waveform_branch = Conv1DBranch(out_dim=branch_dim)

        n_branches = int(self.use_mel) + int(self.use_mfcc) + int(self.use_waveform)

        self.classifier = nn.Sequential(
            nn.Linear(n_branches * branch_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, mel, mfcc=None, waveform=None):
        feats = []

        if self.use_mel:
            feats.append(self.mel_branch(mel))

        if self.use_mfcc:
            feats.append(self.mfcc_branch(mfcc))

        if self.use_waveform:
            feats.append(self.waveform_branch(waveform))

        x = torch.cat(feats, dim=1)
        return self.classifier(x)
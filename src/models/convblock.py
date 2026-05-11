from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """ConvBlock sin BatchNorm para evitar error MIOpen"""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # ❌ REMOVIDO: nn.BatchNorm2d(out_channels),  <- Causa error MIOpen
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AudioBranchSimple(nn.Module):
    """
    AudioBranch sin GRU (que falla en ROCm 7.12)
    Usa promediado temporal + FC layers en su lugar
    """

    def __init__(
        self,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            ConvBlock(1, 32, dropout=dropout),
            ConvBlock(32, 64, dropout=dropout),
            ConvBlock(64, 128, dropout=dropout),
        )

        # En lugar de GRU, usamos promediado temporal + FC
        self.temporal_fc = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size * 2),  # Simular bidireccional (2H)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, F, T]
        x = self.cnn(x)              # [B, C, F', T']
        x = x.mean(dim=2)            # [B, C, T'] - promedio sobre frecuencias
        x = x.mean(dim=2)            # [B, C] - promedio sobre tiempo
        x = self.temporal_fc(x)       # [B, 2H]
        return x


class SimpleAudioCRNN(nn.Module):
    """
    CRNN seguro para ROCm 7.12:
    - Sin BatchNorm2d
    - Sin GRU/LSTM
    - Usa operaciones básicas de Conv + FC
    """

    def __init__(
        self,
        num_classes: int,
        use_mfcc: bool = True,
        scalar_dim: int = 0,
        branch_hidden_size: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.use_mfcc = use_mfcc
        self.scalar_dim = scalar_dim

        self.mel_branch = AudioBranchSimple(
            hidden_size=branch_hidden_size,
            dropout=dropout,
        )

        if self.use_mfcc:
            self.mfcc_branch = AudioBranchSimple(
                hidden_size=branch_hidden_size,
                dropout=dropout,
            )
        else:
            self.mfcc_branch = None

        fusion_dim = branch_hidden_size * 2
        if self.use_mfcc:
            fusion_dim += branch_hidden_size * 2
        fusion_dim += scalar_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        mel: torch.Tensor,
        mfcc: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        mel:     [B, 1, F, T]
        mfcc:    [B, 1, F, T] (opcional)
        scalars: [B, N]       (opcional)
        """
        mel_feat = self.mel_branch(mel)

        feats = [mel_feat]

        if self.use_mfcc and self.mfcc_branch is not None:
            if mfcc is None:
                raise ValueError("El modelo fue creado con use_mfcc=True, pero mfcc=None en forward().")
            mfcc_feat = self.mfcc_branch(mfcc)
            feats.append(mfcc_feat)

        if self.scalar_dim > 0:
            if scalars is None:
                raise ValueError("El modelo espera scalars, pero scalars=None en forward().")
            feats.append(scalars.float())

        x = torch.cat(feats, dim=1)
        logits = self.classifier(x)
        return logits


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    import os
    
    # Variables de entorno para MIOpen (por si acaso)
    os.environ["MIOPEN_FIND_ENFORCE"] = "SEARCH_DB_ONLY"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Crear modelo
    model = SimpleAudioCRNN(
        num_classes=10,
        use_mfcc=True,
        scalar_dim=5,
        branch_hidden_size=128,
        dropout=0.25,
    ).to(device)
    
    print("\n✅ Modelo creado correctamente")
    print(model)
    
    # Test forward pass
    print("\n🔥 TEST: Forward pass")
    mel = torch.randn(4, 1, 64, 256, device=device)  # [B, 1, F, T]
    mfcc = torch.randn(4, 1, 64, 256, device=device)
    scalars = torch.randn(4, 5, device=device)
    
    output = model(mel, mfcc=mfcc, scalars=scalars)
    print(f"✅ Output shape: {output.shape}")
    
    # Test backward pass
    print("\n🔥 TEST: Backward pass")
    loss = output.mean()
    loss.backward()
    print(f"✅ Backward OK, loss: {loss.item():.6f}")
    
    print("\n🏁 TODO FUNCIONA EN ROCM 7.12")
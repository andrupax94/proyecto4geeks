from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BUILDING BLOCK: Conv + BN + ReLU
# ============================================================
class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm2d → ReLU. bias=False porque BN ya centra."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================
# BUILDING BLOCK: Residual Block
# ============================================================
class ResBlock(nn.Module):
    """
    Bloque residual ligero:
        x → Conv-BN-ReLU → Conv-BN → (+x) → ReLU
    Si los canales cambian, se proyecta con una conv 1×1.
    """

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.drop  = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


# ============================================================
# SINGLE FEATURE BRANCH  (mel  ó  mfcc)
# ============================================================
class FeatureBranch(nn.Module):
    """
    Encoder CNN para un único espectrograma [B, 1, F, T].

    Pooling asimétrico:
      - (2, 2) en bloques iniciales para reducir F y T por igual.
      - (1, 2) en bloques finales para conservar resolución de frecuencia.

    Salida: [B, out_channels] tras Global Average Pooling.
    """

    def __init__(self, out_channels: int = 256, dropout: float = 0.2):
        super().__init__()

        # ── Stem ──────────────────────────────────────────
        self.stem = nn.Sequential(
            ConvBNReLU(1, 32, kernel_size=3, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),   # F/2, T/2
            nn.Dropout2d(dropout * 0.5),
        )

        # ── Block 1 ───────────────────────────────────────
        self.block1 = nn.Sequential(
            ConvBNReLU(32, 64),
            ResBlock(64, dropout=dropout * 0.5),
            nn.MaxPool2d(kernel_size=(2, 2)),   # F/4, T/4
            nn.Dropout2d(dropout),
        )

        # ── Block 2 ───────────────────────────────────────
        self.block2 = nn.Sequential(
            ConvBNReLU(64, 128),
            ResBlock(128, dropout=dropout),
            nn.MaxPool2d(kernel_size=(2, 2)),   # F/8, T/8
            nn.Dropout2d(dropout),
        )

        # ── Block 3 ───────────────────────────────────────
        self.block3 = nn.Sequential(
            ConvBNReLU(128, out_channels),
            ResBlock(out_channels, dropout=dropout),
            nn.MaxPool2d(kernel_size=(1, 2)),   # conserva F, T/16
            nn.Dropout2d(dropout),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, F, T] → [B, out_channels]"""
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return x.view(x.size(0), -1)


# ============================================================
# MULTI-BRANCH CNN  (mel + mfcc)
# ============================================================
class DualBranchCNN(nn.Module):
    """
    Arquitectura de dos ramas para clasificación de audio urbano.

    Modos de operación
    ------------------
    - "mel_only"  : solo rama mel  (compatible con código anterior)
    - "mel_mfcc"  : mel + mfcc, fusión por concatenación

    La fusión ocurre antes del clasificador final, lo que permite
    que cada rama aprenda representaciones especializadas.

    Esquema (modo mel_mfcc):
        Mel  → FeatureBranch → [B, branch_channels]  ──┐
                                                         cat → [B, 2*branch_channels]
        MFCC → FeatureBranch → [B, branch_channels]  ──┘
                                   ↓
                             Classifier → [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        mode: str = "mel_mfcc",      # "mel_only" | "mel_mfcc"
        branch_channels: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        if mode not in ("mel_only", "mel_mfcc"):
            raise ValueError(f"mode debe ser 'mel_only' o 'mel_mfcc', recibido: {mode!r}")

        self.mode = mode

        # ── Ramas ─────────────────────────────────────────
        self.mel_branch = FeatureBranch(out_channels=branch_channels, dropout=dropout)

        if mode == "mel_mfcc":
            self.mfcc_branch = FeatureBranch(out_channels=branch_channels, dropout=dropout)
            fusion_dim = branch_channels * 2
        else:
            self.mfcc_branch = None
            fusion_dim = branch_channels

        # ── Clasificador ──────────────────────────────────
        # hidden_dim ligeramente reducido respecto a la fusión
        hidden_dim = max(fusion_dim, 512)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),

            nn.Linear(256, num_classes),
        )

    # ----------------------------------------------------------
    def forward(self, mel: torch.Tensor, mfcc: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parámetros
        ----------
        mel  : [B, 1, F_mel, T]
        mfcc : [B, 1, F_mfcc, T]  — requerido si mode == 'mel_mfcc'

        Retorna
        -------
        logits : [B, num_classes]
        """
        mel_feat = self.mel_branch(mel)       # [B, branch_channels]

        if self.mode == "mel_mfcc":
            if mfcc is None:
                raise ValueError("Se esperaba 'mfcc' pero recibió None en modo 'mel_mfcc'.")
            mfcc_feat = self.mfcc_branch(mfcc) # [B, branch_channels]
            features = torch.cat([mel_feat, mfcc_feat], dim=1)  # [B, 2*branch_channels]
        else:
            features = mel_feat                # [B, branch_channels]

        return self.classifier(features)       # [B, num_classes]


# ============================================================
# ALIAS de compatibilidad  (drop-in replacement del modelo v1)
# ============================================================
class ImprovedMFCCCNN(DualBranchCNN):
    """
    Alias para mantener compatibilidad con trainV1.py.

    En trainV1.py se instancia como:
        model = ImprovedMFCCCNN(num_classes=num_classes, dropout=0.25)

    Ahora DualBranchCNN acepta los mismos kwargs, por lo que
    no hay que tocar el training loop.

    Para activar la segunda rama, cambia en CFG:
        mode = "mel_mfcc"     (en lugar de "mel_only")
        use_mfcc = True
    Y pasa mode al constructor si quieres sobrescribir el default.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3, mode: str = "mel_mfcc"):
        super().__init__(num_classes=num_classes, mode=mode, dropout=dropout)


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    import os

    os.environ["MIOPEN_FIND_ENFORCE"] = "SEARCH_DB_ONLY"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Modo mel_only (backward compatible) ───────────────
    model_single = ImprovedMFCCCNN(num_classes=206, dropout=0.3, mode="mel_only").to(device)
    mel  = torch.randn(4, 1, 128, 256, device=device)  # [B, 1, mel_bins, T]
    out  = model_single(mel)
    out.mean().backward()
    params = sum(p.numel() for p in model_single.parameters())
    print(f"✅ mel_only  | output: {out.shape} | params: {params:,}")

    # ── Modo mel_mfcc (dual branch) ────────────────────────
    model_dual = ImprovedMFCCCNN(num_classes=206, dropout=0.3, mode="mel_mfcc").to(device)
    mel  = torch.randn(4, 1, 128, 256, device=device)  # mel_bins = 128
    mfcc = torch.randn(4, 1,  40, 256, device=device)  # mfcc_bins =  40
    out  = model_dual(mel, mfcc)
    out.mean().backward()
    params = sum(p.numel() for p in model_dual.parameters())
    print(f"✅ mel_mfcc  | output: {out.shape} | params: {params:,}")

    print("\n✅ Forward + Backward: OK")

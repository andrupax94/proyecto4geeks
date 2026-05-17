from __future__ import annotations

import torch
import torch.nn as nn

# Modos soportados
# - "mel_only"  : solo espectrograma mel  (comportamiento original)
# - "mfcc_only" : solo MFCC
# - "mel_mfcc"  : ambos — los embeddings de 256-d se suman antes del clasificador


def _make_cnn_backbone(dropout: float) -> nn.Sequential:
    """
    Backbone CNN compartido (sin BatchNorm, compatible con AMD gfx1102).
    Entrada:  [B, 1, F, T]
    Salida:   [B, 256, F', T']  (antes del global pool)
    """
    return nn.Sequential(
        # Block 1 — 64 canales
        nn.Conv2d(1, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Dropout2d(dropout * 0.8),

        # Block 2 — 128 canales
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Dropout2d(dropout * 0.8),

        # Block 3 — 256 canales
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Dropout2d(dropout * 0.8),
    )


class ImprovedMFCCCNN(nn.Module):
    """
    CNN optimizado para AMD RX 7600 XT (gfx1102).

    Modos de entrada (argumento `mode`):
        "mel_only"  — solo espectrograma mel       [B, 1, F_mel, T]
        "mfcc_only" — solo coeficientes MFCC       [B, 1, F_mfcc, T]
        "mel_mfcc"  — mel + MFCC (fusión por suma) [B, 1, F_mel, T]  +  [B, 1, F_mfcc, T]

    En modo "mel_mfcc" el forward espera dos tensores:
        logits = model(mel, mfcc)

    En los otros modos espera uno:
        logits = model(x)

    Notas de diseño:
        - SIN BatchNorm (causa miopenStatusUnknownError en gfx1102)
        - LayerNorm en capas dense para estabilidad
        - Fusión por suma: misma dimensión de embedding (256-d), sin parámetros extra,
          gradientes fluyen por ambas ramas de forma independiente.
        - Los dos backbones comparten arquitectura pero NO comparten pesos
          (mel y MFCC tienen distribuciones muy distintas).
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.25,
        mode: str = "mel_only",
    ):
        if mode not in ("mel_only", "mfcc_only", "mel_mfcc"):
            raise ValueError(f"mode debe ser 'mel_only', 'mfcc_only' o 'mel_mfcc'. Recibido: {mode!r}")

        super().__init__()
        self.mode = mode

        # ── Backbones ──────────────────────────────────────────────────────────
        if mode in ("mel_only", "mel_mfcc"):
            self.cnn_mel = _make_cnn_backbone(dropout)

        if mode in ("mfcc_only", "mel_mfcc"):
            self.cnn_mfcc = _make_cnn_backbone(dropout)

        # ── Global Average Pooling ─────────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Clasificador (LayerNorm, sin BatchNorm) ───────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def _embed(self, cnn: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        """Pasa x por un backbone y devuelve el embedding [B, 256]."""
        feat = cnn(x)                        # [B, 256, F', T']
        feat = self.global_pool(feat)        # [B, 256, 1, 1]
        return feat.view(feat.size(0), -1)   # [B, 256]

    def forward(self, mel: torch.Tensor, mfcc: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parámetros
        ----------
        mel  : [B, 1, F_mel, T]   — requerido siempre excepto en modo "mfcc_only"
        mfcc : [B, 1, F_mfcc, T]  — requerido en modos "mfcc_only" y "mel_mfcc"

        Retorna
        -------
        logits : [B, num_classes]
        """
        if self.mode == "mel_only":
            embedding = self._embed(self.cnn_mel, mel)

        elif self.mode == "mfcc_only":
            if mfcc is None:
                raise ValueError("mode='mfcc_only' requiere el tensor 'mfcc'.")
            embedding = self._embed(self.cnn_mfcc, mfcc)

        else:  # "mel_mfcc"
            if mfcc is None:
                raise ValueError("mode='mel_mfcc' requiere el tensor 'mfcc'.")
            # Fusión por suma: misma dimensión, gradientes independientes por rama
            embedding = self._embed(self.cnn_mel, mel) + self._embed(self.cnn_mfcc, mfcc)

        return self.classifier(embedding)


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    import os
    import time

    os.environ["MIOPEN_FIND_ENFORCE"] = "SEARCH_DB_ONLY"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 23
    BATCH = 16

    for mode in ("mel_only", "mfcc_only", "mel_mfcc"):
        print(f"\n{'=' * 60}")
        print(f"🔧 Modo: {mode}")

        model = ImprovedMFCCCNN(num_classes=NUM_CLASSES, dropout=0.25, mode=mode).to(device)

        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total params    : {total_params:,}")
        print(f"   Trainable params: {trainable_params:,}")

        mel  = torch.randn(BATCH, 1, 128, 360, device=device)
        mfcc = torch.randn(BATCH, 1,  40, 360, device=device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()

        times = []
        for i in range(5):
            t0 = time.time()

            if mode == "mel_only":
                out = model(mel)
            elif mode == "mfcc_only":
                out = model(mel, mfcc)   # mel ignorado internamente
            else:
                out = model(mel, mfcc)

            labels = torch.randint(0, NUM_CLASSES, (BATCH,), device=device)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dt = time.time() - t0
            times.append(dt)
            print(f"   iter {i+1}: {dt*1000:.1f}ms  ({BATCH/dt:.0f} img/s)")

        avg = sum(times) / len(times)
        print(f"   ⚡ Promedio: {avg*1000:.1f}ms/batch  |  {BATCH/avg:.0f} img/s")

    print(f"\n{'=' * 60}")
    print("✅ Todos los modos OK")

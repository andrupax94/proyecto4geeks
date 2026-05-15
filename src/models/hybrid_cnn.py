from __future__ import annotations

import torch
import torch.nn as nn


class ImprovedMFCCCNN(nn.Module):
    """
    CNN mejorado para clasificación de 206 clases
    - Arquitectura más profunda
    - Más canales
    - Mejor regularización
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Convoluciones - Arquitectura más potente
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(dropout),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(dropout),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(dropout),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Clasificador más robusto
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, F, T] - MFCC espectrograma
        output: [B, num_classes]
        """
        x = self.cnn(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    import os
    
    os.environ["MIOPEN_FIND_ENFORCE"] = "SEARCH_DB_ONLY"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test con 206 clases
    model = ImprovedMFCCCNN(num_classes=206, dropout=0.3).to(device)
    
    print("✅ Modelo para 206 clases creado")
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Test forward
    mfcc = torch.randn(4, 1, 40, 256, device=device)
    output = model(mfcc)
    loss = output.mean()
    loss.backward()
    
    print(f"✅ Forward: {output.shape}")
    print(f"✅ Backward: OK")
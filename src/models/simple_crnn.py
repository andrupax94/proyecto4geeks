from __future__ import annotations

import torch
import torch.nn as nn


class SimpleMFCCCNN(nn.Module):
    """
    CNN simple para clasificación de audio a partir de MFCC
    - Solo lee MFCC [B, 1, F, T]
    - Sin BatchNorm (evita error MIOpen en ROCm 7.12)
    - Sin RNN
    - Solo Conv2d + MaxPool2d + FC
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.25,
    ):
        super().__init__()

        # Convoluciones - MaxPool SOLO temporal (dim 3), no en frecuencias (dim 2)
        self.cnn = nn.Sequential(
            # Block 1: Conv + ReLU + Pool temporal
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # ✅ Pool solo en tiempo
            nn.Dropout2d(dropout),

            # Block 2: Conv + ReLU + Pool temporal
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # ✅ Pool solo en tiempo
            nn.Dropout2d(dropout),

            # Block 3: Conv + ReLU + Pool temporal
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # ✅ Pool solo en tiempo
            nn.Dropout2d(dropout),

            # Block 4: Conv + ReLU (sin pool, usamos adaptive pooling después)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # Global Average Pooling en lugar de Flatten
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, F, T] - MFCC espectrograma
           B: batch size
           1: canal (monocanal)
           F: número de coeficientes MFCC (típicamente 40)
           T: timesteps (duración temporal)
        
        output: [B, num_classes]
        
        Nota: MaxPool se aplica solo en dimensión temporal (T),
        no en frecuencias (F), para evitar reducir demasiado las dimensiones.
        """
        x = self.cnn(x)                 # [B, 256, F, T'] (F se mantiene igual)
        x = self.global_pool(x)         # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)       # [B, 256]
        logits = self.classifier(x)     # [B, num_classes]
        return logits


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    import os
    
    # Variables de entorno para MIOpen
    os.environ["MIOPEN_FIND_ENFORCE"] = "SEARCH_DB_ONLY"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Crear modelo
    model = SimpleMFCCCNN(
        num_classes=10,
        dropout=0.25,
    ).to(device)
    
    print("✅ Modelo creado")
    print(model)
    print()
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parámetros: {total_params:,}")
    print(f"Parámetros entrenable: {trainable_params:,}\n")
    
    # Test forward pass
    print("🔥 TEST 1: Forward pass")
    batch_size = 4
    mfcc_dim = 40  # Coeficientes MFCC típicos
    time_steps = 256  # Duración temporal
    
    mfcc = torch.randn(batch_size, 1, mfcc_dim, time_steps, device=device)
    print(f"Input shape: {mfcc.shape}")
    
    output = model(mfcc)
    print(f"Output shape: {output.shape}")
    print(f"✅ Forward OK\n")
    
    # Test backward pass
    print("🔥 TEST 2: Backward pass")
    loss = output.mean()
    loss.backward()
    print(f"Loss: {loss.item():.6f}")
    print(f"✅ Backward OK\n")
    
    # Test con batch diferente
    print("🔥 TEST 3: Batch diferente")
    batch_size2 = 8
    mfcc2 = torch.randn(batch_size2, 1, mfcc_dim, time_steps, device=device, requires_grad=True)
    output2 = model(mfcc2)
    loss2 = output2.mean()
    loss2.backward()
    print(f"Input shape: {mfcc2.shape}")
    print(f"Output shape: {output2.shape}")
    print(f"Loss: {loss2.item():.6f}")
    print(f"✅ Backward OK\n")
    
    print("🏁 MODELO LISTO PARA ROCM 7.12")
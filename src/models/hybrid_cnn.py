from __future__ import annotations

import torch
import torch.nn as nn


class ImprovedMFCCCNN(nn.Module):
    """
    CNN optimizado para AMD RX 7600 XT (gfx1102)
    - SIN BatchNorm (causa miopenStatusUnknownError en gfx1102)
    - Arquitectura más ligera (1.2M parámetros)
    - LayerNorm en capas dense para estabilidad
    - Diseñado para entrenamiento rápido sin bugs de MIOpen
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.25,
    ):
        super().__init__()

        # 🔥 OPTIMIZACIÓN: CNN sin BatchNorm
        self.cnn = nn.Sequential(
            # Block 1 - 64 canales
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout * 0.8),

            # Block 2 - 128 canales
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout * 0.8),

            # Block 3 - 256 canales
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout * 0.8),
        )

        # 🔥 Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 🔥 OPTIMIZACIÓN: Clasificador compacto con LayerNorm
        # LayerNorm funciona bien en AMD GPU a diferencia de BatchNorm
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),  # ✅ Usa LayerNorm en lugar de BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(128, num_classes),
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
    
    # Test con 23 clases
    model = ImprovedMFCCCNN(num_classes=23, dropout=0.25).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("✅ Modelo creado (sin BatchNorm, compatible con AMD)")
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}\n")
    
    # Benchmark
    import time
    
    print("🔥 Benchmark (10 iteraciones):")
    print("=" * 60)
    
    batch_size = 16
    mfcc = torch.randn(batch_size, 1, 128, 360, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    
    times = []
    for i in range(10):
        t0 = time.time()
        
        # Forward
        output = model(mfcc)
        
        # Loss
        labels = torch.randint(0, 23, (batch_size,), device=device)
        loss = criterion(output, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        dt = time.time() - t0
        times.append(dt)
        
        img_per_sec = batch_size / dt
        print(f"   Iteración {i+1:2d}: {dt*1000:6.1f}ms ({img_per_sec:6.0f} img/s)")
    
    avg_time = sum(times) / len(times)
    avg_speed = batch_size / avg_time
    
    print("=" * 60)
    print(f"   ⚡ Promedio: {avg_time*1000:.1f}ms por batch")
    print(f"   ⚡ Velocidad: {avg_speed:.0f} imágenes/segundo")
    
    epoch_time = 1598 * avg_time / 60  # 1598 batches por época
    print(f"   ⚡ Una época: ~{epoch_time:.1f} minutos")
    print(f"   ⚡ 20 épocas: ~{epoch_time*20/60:.1f} horas")
    print("=" * 60)
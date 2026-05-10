import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# --- FIX PARA AMD EN WINDOWS ---
# Desactiva el dropout de MIOpen que rompe la GRU
os.environ["MIOPEN_DEBUG_DISABLE_DROPOUT"] = "1"
# Fuerza a MIOpen a usar búsqueda rápida de kernels en lugar de compilar nuevos
os.environ["MIOPEN_FIND_MODE"] = "1"
# Evita conflictos de memoria en el compilador JIT
os.environ["MIOPEN_ENABLE_LOGGING_ELAPSED_TIME"] = "0"

class AudioCRNN(nn.Module):
    """
    Versión V4: Diseñada para evitar kernels de reducción JIT en AMD/Windows.
    """
    def __init__(self, num_classes, scalar_dim=0, use_mfcc=True):
        super().__init__()
        
        self.use_mfcc = use_mfcc
        self.use_scalars = scalar_dim > 0
        
        # --- STREAM MEL (CNN 2D) ---
        # Entrada esperada: (B, 1, 128, T)
        self.mel_conv = nn.Sequential(
            # Bloque 1: 128 -> 64
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), 
            
            # Bloque 2: 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            # Bloque 3: 32 -> 8 (Reducción de frecuencia manual para evitar AdaptivePool)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1)) # Reduce frecuencia 32/4 = 8, mantiene tiempo
        )
        
        # --- STREAM MFCC (CNN 1D) ---
        if self.use_mfcc:
            self.mfcc_conv = nn.Sequential(
                nn.Conv1d(13, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )

        # --- RNN (GRU) ---
        # 128 canales * 8 frecuencias resultantes = 1024
        rnn_input_size = 1024 
        
        self.gru = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 # MANTENER EN 0.0 PARA AMD WINDOWS
        )
        
        # --- ESCALARES ---
        if self.use_scalars:
            self.scalar_fc = nn.Sequential(
                nn.Linear(scalar_dim, 32),
                nn.ReLU()
            )
        
        # --- CLASIFICADOR ---
        # 256 (GRU bidirectional) + 32 (scalars)
        head_input = 256 + (32 if self.use_scalars else 0)
        self.classifier = nn.Sequential(
            nn.Linear(head_input, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, mel, mfcc=None, scalars=None):
        # 1. MEL Stream
        if mel.dim() == 3: mel = mel.unsqueeze(1)
        x = self.mel_conv(mel) # Shape: (B, 128, 8, T_reduced)
        
        # Reshape para GRU: (B, T, Features)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, t, -1) # (B, T, 128*8=1024)
        
        # 2. GRU
        x_rnn, _ = self.gru(x)
        x_rnn = x_rnn[:, -1, :] # Tomamos el último estado temporal (256 dims)
        
        # 3. Scalars
        if self.use_scalars and scalars is not None:
            x_s = self.scalar_fc(scalars)
            x_rnn = torch.cat([x_rnn, x_s], dim=-1)
            
        # 4. Final
        return self.classifier(x_rnn)

if __name__ == "__main__":
    model = AudioCRNN(num_classes=527, scalar_dim=9) # Ajustado a tus datos
    test_mel = torch.randn(2, 128, 200) 
    test_scalars = torch.randn(2, 9)
    output = model(test_mel, scalars=test_scalars)
    print("Salida exitosa:", output.shape)
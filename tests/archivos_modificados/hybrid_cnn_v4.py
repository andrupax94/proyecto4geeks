from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# HYBRID_CNN_V5 — Optimizado para RX 7600 XT (gfx1102 / ROCm)
# ─────────────────────────────────────────────────────────────────────────────
#
# NOVEDADES vs v4
# ─────────────────────────────────────────────────────────────────────────────
#   ✅ Modo BINARIO  (task="binary")   → salida [B, 1], loss BCEWithLogitsLoss
#   ✅ Modo MULTICLASE (task="multiclass") → salida [B, C], loss CrossEntropyLoss
#   ✅ Fix ROCm gfx1102: evita nll_loss_forward_reduce_cuda_kernel_2d (crash)
#        → Se usa BCEWithLogitsLoss en binario (no llama a nll_loss)
#        → En multiclase: loss calculada FUERA del modelo (en el trainer),
#          donde se puede usar label_smoothing con F.cross_entropy que es más
#          estable en ROCm que nn.CrossEntropyLoss con weight= en fp16
#   ✅ Contiguous() forzado antes de ops que crashean en ROCm con fp16
#   ✅ Classifier con LayerNorm estable (reemplaza BN, compatible gfx1102)
#   ✅ Embedding clamp para evitar NaN/Inf en fp16 con AMP
#
# Modos de fusión soportados (argumento ``mode``)
# ─────────────────────────────────────────────────────────────────────────────
#   "mel_only"      : solo espectrograma mel
#   "mfcc_only"     : solo MFCC
#   "mel_mfcc"      : mel + MFCC  (fusión por suma)
#   "mel_waveform"  : mel + waveform crudo (fusión por concatenación → 512-d)
#
# Dimensiones de referencia (sr=22050, dur=3s, n_fft=1024, hop=256):
#   mel      : [B, 1, 128, 259]
#   mfcc     : [B, 1,  13, 259]
#   waveform : [B, 1, 66150]
# ─────────────────────────────────────────────────────────────────────────────


# ─── CONSTANTE: clamp para embeddings en fp16 ─────────────────────────────
_EMB_CLAMP = 1e4   # evita overflow en fp16 antes del classifier


def _make_cnn_backbone(dropout: float) -> nn.Sequential:
    """
    Backbone CNN 2-D compartido para mel y MFCC.
    Sin BatchNorm (inestable en ROCm gfx1102 con AMP + grupos pequeños).

    Entrada : [B, 1, F, T]
    Salida  : [B, 256, F', T']
    """
    return nn.Sequential(
        # Block 1 — 64 canales
        nn.Conv2d(1, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Dropout2d(dropout * 0.5),

        # Block 2 — 128 canales
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Dropout2d(dropout * 0.5),

        # Block 3 — 256 canales
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Dropout2d(dropout * 0.5),
    )


def _make_waveform_backbone(dropout: float) -> nn.Sequential:
    """
    Backbone CNN 1-D para waveform crudo.
    Sin BatchNorm (misma razón que 2D).

    Entrada : [B, 1, 66150]
    Salida  : [B, 256, L']
    """
    return nn.Sequential(
        # Block 1 — kernel=15, stride=8
        nn.Conv1d(1,   64,  kernel_size=15, stride=8,  padding=7),
        nn.ReLU(inplace=True),
        nn.Conv1d(64,  64,  kernel_size=7,  stride=1,  padding=3),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout * 0.4),

        # Block 2 — stride=4
        nn.Conv1d(64,  128, kernel_size=7,  stride=4,  padding=3),
        nn.ReLU(inplace=True),
        nn.Conv1d(128, 128, kernel_size=5,  stride=1,  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout * 0.4),

        # Block 3 — stride=2
        nn.Conv1d(128, 256, kernel_size=5,  stride=2,  padding=2),
        nn.ReLU(inplace=True),
        nn.Conv1d(256, 256, kernel_size=3,  stride=2,  padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout * 0.4),
    )


def _make_classifier_narrow(embed_dim: int, num_classes_or_1: int, dropout: float) -> nn.Sequential:
    """
    Classifier para embed_dim=256.
    num_classes_or_1 : num_classes en multiclase, 1 en binario.
    """
    return nn.Sequential(
        nn.LayerNorm(embed_dim),          # estable en ROCm con fp16 (reemplaza BN)
        nn.Linear(embed_dim, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(128, num_classes_or_1),
    )


def _make_classifier_wide(embed_dim: int, num_classes_or_1: int, dropout: float) -> nn.Sequential:
    """
    Classifier para embed_dim=512 (modo mel_waveform).
    """
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes_or_1),
    )


class HybridCNNV5(nn.Module):
    """
    CNN multi-modal v5 para RX 7600 XT (ROCm gfx1102).

    Parámetros
    ----------
    num_classes : int
        Número de clases. En task="binary" se ignora (salida siempre 1-d).
    dropout     : float  (default 0.25)
    mode        : str    Modo de fusión de entrada.
        "mel_only" | "mfcc_only" | "mel_mfcc" | "mel_waveform"
    task        : str
        "binary"     → salida [B, 1] logit, usar BCEWithLogitsLoss
        "multiclass" → salida [B, num_classes] logits, usar CrossEntropyLoss

    Notas ROCm
    ----------
    - El crash HSA_STATUS_ERROR_EXCEPTION / nll_loss_forward_reduce_cuda_kernel_2d
      ocurre en ROCm cuando CrossEntropyLoss recibe índices fuera de rango o
      tensores no contiguos en fp16.  En task="binary" esto se evita por
      completo usando BCEWithLogitsLoss.  En task="multiclass" el modelo
      devuelve logits crudos: la loss se calcula en el trainer con
      F.cross_entropy(...) que es más estable que nn.CrossEntropyLoss con
      weight= en fp16.
    - Los embeddings se clampean a ±1e4 antes del classifier para evitar
      Inf/NaN en autocast fp16.
    - Se llama .contiguous() en los tensores antes de ops que internamente
      usan kernels reducción (fuente habitual del crash).
    """

    _VALID_MODES = ("mel_only", "mfcc_only", "mel_mfcc", "mel_waveform")
    _VALID_TASKS = ("binary", "multiclass")

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.25,
        mode: str = "mel_only",
        task: str = "binary",
    ):
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode debe ser uno de {self._VALID_MODES}. Recibido: {mode!r}")
        if task not in self._VALID_TASKS:
            raise ValueError(f"task debe ser uno de {self._VALID_TASKS}. Recibido: {task!r}")

        super().__init__()
        self.mode = mode
        self.task = task
        self.num_classes = num_classes

        # ── Salida: 1 logit en binario, num_classes en multiclase ────────
        out_dim = 1 if task == "binary" else num_classes

        # ── Backbones 2D ──────────────────────────────────────────────────
        if mode in ("mel_only", "mel_mfcc", "mel_waveform"):
            self.cnn_mel = _make_cnn_backbone(dropout)

        if mode in ("mfcc_only", "mel_mfcc"):
            self.cnn_mfcc = _make_cnn_backbone(dropout)

        # ── Backbone 1D ───────────────────────────────────────────────────
        if mode == "mel_waveform":
            self.cnn_wave = _make_waveform_backbone(dropout)
            self.global_pool_1d = nn.AdaptiveAvgPool1d(1)

        # ── Global Average Pooling 2D ─────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Clasificadores ─────────────────────────────────────────────────
        embed_dim = 512 if mode == "mel_waveform" else 256

        if mode == "mel_waveform":
            self.classifier = _make_classifier_wide(embed_dim, out_dim, dropout)
        else:
            self.classifier = _make_classifier_narrow(embed_dim, out_dim, dropout)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _embed_2d(self, cnn: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        """Backbone 2D → embedding [B, 256]. Contiguous forzado (fix ROCm)."""
        x = x.contiguous()                        # ← fix ROCm gfx1102
        feat = cnn(x)                              # [B, 256, F', T']
        feat = self.global_pool(feat)              # [B, 256, 1, 1]
        return feat.view(feat.size(0), -1)         # [B, 256]

    def _embed_1d(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone 1D → embedding [B, 256]. Contiguous forzado (fix ROCm)."""
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = x.contiguous()                         # ← fix ROCm gfx1102
        feat = self.cnn_wave(x)                    # [B, 256, L']
        feat = self.global_pool_1d(feat)           # [B, 256, 1]
        return feat.squeeze(-1)                    # [B, 256]

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        mel: torch.Tensor,
        mfcc: torch.Tensor | None = None,
        waveform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parámetros
        ----------
        mel      : [B, 1, F_mel, T]
        mfcc     : [B, 1, F_mfcc, T]   (requerido en mfcc_only / mel_mfcc)
        waveform : [B, 1, T_samples]   (requerido en mel_waveform)

        Retorna
        -------
        task="binary"     → [B, 1]          logit crudo (BCEWithLogitsLoss)
        task="multiclass" → [B, num_classes] logits crudos (CrossEntropyLoss)
        """
        if self.mode == "mel_only":
            emb = self._embed_2d(self.cnn_mel, mel)

        elif self.mode == "mfcc_only":
            if mfcc is None:
                raise ValueError("mode='mfcc_only' requiere 'mfcc'.")
            emb = self._embed_2d(self.cnn_mfcc, mfcc)

        elif self.mode == "mel_mfcc":
            if mfcc is None:
                raise ValueError("mode='mel_mfcc' requiere 'mfcc'.")
            emb = self._embed_2d(self.cnn_mel, mel) + self._embed_2d(self.cnn_mfcc, mfcc)

        else:  # mel_waveform
            if waveform is None:
                raise ValueError("mode='mel_waveform' requiere 'waveform'.")
            emb_mel  = self._embed_2d(self.cnn_mel, mel)
            emb_wave = self._embed_1d(waveform)
            emb = torch.cat([emb_mel, emb_wave], dim=1)   # [B, 512]

        # Clamp anti-overflow fp16 (fix ROCm con AMP) ─────────────────────
        emb = emb.clamp(-_EMB_CLAMP, _EMB_CLAMP)

        return self.classifier(emb)    # [B, 1] o [B, C]

    # ── Utilidades de inferencia ──────────────────────────────────────────

    @torch.no_grad()
    def predict_proba(
        self,
        mel: torch.Tensor,
        mfcc: torch.Tensor | None = None,
        waveform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Probabilidades (no logits).

        task="binary"     → [B, 1]  valores en [0, 1]  (sigmoid)
        task="multiclass" → [B, C]  valores en [0, 1]  (softmax)
        """
        logits = self.forward(mel=mel, mfcc=mfcc, waveform=waveform)
        if self.task == "binary":
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(
        self,
        mel: torch.Tensor,
        mfcc: torch.Tensor | None = None,
        waveform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Clases predichas.

        task="binary"     → [B]  LongTensor {0, 1}
        task="multiclass" → [B]  LongTensor {0, ..., C-1}
        """
        proba = self.predict_proba(mel=mel, mfcc=mfcc, waveform=waveform)
        if self.task == "binary":
            return (proba.squeeze(1) >= 0.5).long()
        return proba.argmax(dim=-1)


# ============================================================
# LOSS FUNCTIONS RECOMENDADAS (ROCm-safe)
# ============================================================

def build_criterion(
    task: str,
    num_classes: int,
    pos_weight: torch.Tensor | None = None,   # solo binario
    class_weights: torch.Tensor | None = None, # solo multiclase
    label_smoothing: float = 0.05,             # solo multiclase
) -> nn.Module:
    """
    Devuelve la función de loss correcta según el task.

    ROCm gfx1102: 
        - BCEWithLogitsLoss: estable en fp16, no usa nll_loss kernel
        - CrossEntropyLoss con reduction='mean': más seguro que con weight= en fp16
          Si necesitas class_weights en fp16, convierte el tensor a fp32 antes.
    """
    if task == "binary":
        # BCEWithLogitsLoss es inmune al crash nll_loss_2d
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # multiclass — CrossEntropyLoss
    # IMPORTANTE para ROCm: si pasas weight=, asegúrate de que sea fp32
    # aunque el modelo esté en fp16 (AMP lo gestiona internamente)
    if class_weights is not None:
        class_weights = class_weights.float()   # fp32 forzado (fix ROCm)

    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


# ============================================================
# TRAINER STEP (ROCm-safe) — snippet listo para pegar
# ============================================================
"""
Uso en tu loop de entrenamiento:

    from torch.amp import autocast, GradScaler

    model = HybridCNNV5(num_classes=num_classes, mode=CFG.mode, task=CFG.task).to(device)
    criterion = build_criterion(task=CFG.task, num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler    = GradScaler(device="cuda")

    model.train()
    for batch, labels, _ in train_loader:
        mel      = batch["mel"].to(device, non_blocking=True)
        waveform = batch.get("waveform")
        if waveform is not None:
            waveform = waveform.to(device, non_blocking=True)

        # ── IMPORTANTE para ROCm binario ──────────────────────────────────
        if CFG.task == "binary":
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)  # [B,1]
        else:
            labels = labels.long().to(device, non_blocking=True)                # [B]

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            logits = model(mel=mel, waveform=waveform)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # ── Predicciones para métricas ────────────────────────────────────────
    preds = model.predict(mel=mel, waveform=waveform)   # [B] LongTensor siempre
"""


# ============================================================
# CONFIGURACIÓN DE REFERENCIA
# ============================================================

class CFG:
    # ── Task ─────────────────────────────────────────────────────────────
    task = "binary"          # "binary" | "multiclass"
    mode = "mel_waveform"    # "mel_only" | "mfcc_only" | "mel_mfcc" | "mel_waveform"

    # ── Entrenamiento ────────────────────────────────────────────────────
    batch_size   = 48
    lr           = 3e-4
    weight_decay = 1e-4
    dropout      = 0.25
    epochs       = 10
    num_workers  = 6

    # ── Dataset ──────────────────────────────────────────────────────────
    use_mfcc      = True
    use_scalars   = False
    seed          = 42
    print_every   = 50
    target_type   = "alertable"
    label_version = 2
    version       = 5

    # ── GPU ──────────────────────────────────────────────────────────────
    use_amp            = True
    pin_memory         = True
    persistent_workers = True
    prefetch_factor    = 2

    # ── Checkpoints ──────────────────────────────────────────────────────
    save_every = 1
    resume_lr  = None


# ============================================================
# TEST — verifica modos y tasks con AMP
# ============================================================
if __name__ == "__main__":
    import os
    import time

    os.environ["MIOPEN_FIND_ENFORCE"] = "SEARCH_DB_ONLY"
    # Habilita aserciones en device para diagnosticar futuros crashes ROCm:
    # os.environ["TORCH_USE_HIP_DSA"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    BATCH       = 32
    NUM_CLASSES = 4   # para multiclase

    mel_t  = torch.randn(BATCH, 1, 128, 259, device=device)
    mfcc_t = torch.randn(BATCH, 1,  13, 259, device=device)
    wave_t = torch.randn(BATCH, 1, 66150,    device=device)

    modes_inputs = {
        "mel_only":     {"mel": mel_t},
        "mfcc_only":    {"mel": mel_t, "mfcc": mfcc_t},
        "mel_mfcc":     {"mel": mel_t, "mfcc": mfcc_t},
        "mel_waveform": {"mel": mel_t, "waveform": wave_t},
    }

    from torch.amp import autocast, GradScaler

    print("\n" + "=" * 70)
    print("🚀 HYBRID CNN V5 — BINARY + MULTICLASS + ROCm FIXES")
    print("=" * 70)

    for task in ("binary", "multiclass"):
        for mode, inputs in modes_inputs.items():
            print(f"\n{'─' * 70}")
            print(f"🔧  task={task}  |  mode={mode}")

            nc = 1 if task == "binary" else NUM_CLASSES
            model = HybridCNNV5(
                num_classes=NUM_CLASSES,
                dropout=0.25,
                mode=mode,
                task=task,
            ).to(device)

            criterion = build_criterion(task=task, num_classes=NUM_CLASSES)

            if task == "binary":
                labels = torch.randint(0, 2, (BATCH,)).float().unsqueeze(1).to(device)
            else:
                labels = torch.randint(0, NUM_CLASSES, (BATCH,)).long().to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            scaler    = GradScaler(device="cuda")

            total_params = sum(p.numel() for p in model.parameters())
            print(f"   Params: {total_params:,}")

            # Warmup pass
            model.train()
            times = []
            for i in range(3):
                t0 = time.time()
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type="cuda"):
                    out  = model(**inputs)
                    loss = criterion(out, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                dt = time.time() - t0
                times.append(dt)

            avg = sum(times) / len(times)
            print(f"   ⚡ Avg: {avg*1000:.1f}ms/batch  ({BATCH/avg:.0f} img/s)")

            # Verificar shapes
            model.eval()
            with torch.no_grad():
                logits = model(**inputs)
                proba  = model.predict_proba(**inputs)
                preds  = model.predict(**inputs)

            expected_logit_shape = (BATCH, 1) if task == "binary" else (BATCH, NUM_CLASSES)
            assert logits.shape == torch.Size(expected_logit_shape), \
                f"Shape logits incorrecto: {logits.shape}"
            assert preds.shape == torch.Size([BATCH]), \
                f"Shape preds incorrecto: {preds.shape}"
            assert preds.dtype == torch.long, \
                f"dtype preds incorrecto: {preds.dtype}"

            print(f"   ✅ logits {logits.shape}  preds {preds.shape}  OK")

    print(f"\n{'=' * 70}")
    print("✅ Todos los modos y tasks OK")
    print("=" * 70)
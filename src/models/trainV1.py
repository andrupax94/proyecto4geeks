from __future__ import annotations
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from src.models.hybrid_cnn import HybridAudioClassifier


from src.utils.config import PROCESSED_METADATA, LABEL_MAPPING
from src.models.audio_dataset import ProcessedAudioDataset, crnn_collate_fn



# =========================================================
# CONFIG
# =========================================================
class CFG:
    batch_size = 16
    lr = 3e-4
    epochs = 20
    num_workers = 0
    use_mfcc = True
    use_scalars = False
    seed = 42
    print_every = 50
    target_type = "human_label"
    mode = "mel_only"   # "mel_only", "mel_mfcc", "all_three"

def prepare_model_inputs(batch, device, mode: str):
    """
    Devuelve una tupla de tensores según el modo:
    - mel_only: (mel,)
    - mel_mfcc: (mel, mfcc)
    - all_three: (mel, mfcc, waveform)
    """
    mel = batch.get("mel", None)
    mfcc = batch.get("mfcc", None)
    waveform = batch.get("waveform", None)

    if mel is not None:
        mel = mel.to(device)
    if mfcc is not None:
        mfcc = mfcc.to(device)
    if waveform is not None:
        waveform = waveform.to(device)

    if mode == "mel_only":
        if mel is None:
            raise KeyError("El batch no contiene 'mel'")
        return (mel,)

    if mode == "mel_mfcc":
        if mel is None or mfcc is None:
            raise KeyError("El batch no contiene 'mel' o 'mfcc'")
        return (mel, mfcc)

    if mode == "all_three":
        if mel is None or mfcc is None or waveform is None:
            raise KeyError("El batch no contiene 'mel', 'mfcc' o 'waveform'")
        return (mel, mfcc, waveform)

    raise ValueError("mode debe ser 'mel_only', 'mel_mfcc' o 'all_three'")
# =========================================================
# REPRODUCIBILIDAD
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# DEVICE (ROCm usa "cuda")
# =========================================================
def get_device():
    # return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# DATALOADERS
# =========================================================
# =========================================================
# DATALOADERS
# =========================================================
def build_loaders(cfg: CFG):

    # =====================================================
    # Elegir mapping automáticamente
    # =====================================================
    if cfg.target_type == "human_label":
        label_mapping_path = LABEL_MAPPING["human_label"]

    elif cfg.target_type == "alertable":
        label_mapping_path = LABEL_MAPPING["alertable"]

    else:
        raise ValueError(
            f"target_type inválido: {cfg.target_type}"
        )

    print(f"\n🎯 Target seleccionado: {cfg.target_type}")
    print(f"🗂️ Label mapping: {label_mapping_path}")

    # =====================================================
    # TRAIN
    # =====================================================
    train_ds = ProcessedAudioDataset(
        metadata_csv=PROCESSED_METADATA,
        label_mapping_path=label_mapping_path,
        split="train",

        # 🔥 columna target
        target_column=cfg.target_type,

        use_mfcc=cfg.use_mfcc,
        use_scalars=cfg.use_scalars,
    )

    # =====================================================
    # TEST
    # =====================================================
    test_ds = ProcessedAudioDataset(
        metadata_csv=PROCESSED_METADATA,
        label_mapping_path=label_mapping_path,
        split="test",

        # 🔥 misma columna target
        target_column=cfg.target_type,

        use_mfcc=cfg.use_mfcc,
        use_scalars=cfg.use_scalars,

        target_frames=train_ds.target_frames,
    )

    # =====================================================
    # LOADERS
    # =====================================================
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=crnn_collate_fn,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=crnn_collate_fn,
    )

    return train_ds, test_ds, train_loader, test_loader


# =========================================================
# TRAIN
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device, cfg, epoch):
    model.train()

    total_loss = 0

    # ✅ FIX
    y_true, y_pred = [], []

    num_batches = len(loader)

    for batch_idx, (batch, labels, _) in enumerate(loader, 1):
        labels = labels.to(device)

        inputs = prepare_model_inputs(batch, device, cfg.mode)

        optimizer.zero_grad()

        outputs = model(*inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        if batch_idx % cfg.print_every == 0:
            avg_loss = total_loss / batch_idx
            avg_acc = accuracy_score(y_true, y_pred)

            print(
                f"  Epoch {epoch} | Batch {batch_idx}/{num_batches} | "
                f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}"
            )

    epoch_loss = total_loss / num_batches
    epoch_acc = accuracy_score(y_true, y_pred)

    return epoch_loss, epoch_acc


# =========================================================
# EVAL
# =========================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg, epoch):
    model.eval()

    total_loss = 0
    y_true, y_pred = [],[]

    num_batches = len(loader)

    for batch_idx, (batch, labels, _) in enumerate(loader, 1):
        labels = labels.to(device)

        inputs = prepare_model_inputs(batch, device, cfg.mode)

        outputs = model(*inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        if batch_idx % cfg.print_every == 0:
            avg_loss = total_loss / batch_idx
            avg_acc = accuracy_score(y_true, y_pred)
            print(
                f"  Epoch {epoch} | Val Batch {batch_idx}/{num_batches} | "
                f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}"
            )

    epoch_loss = total_loss / num_batches
    epoch_acc = accuracy_score(y_true, y_pred)
    return epoch_loss, epoch_acc


# =========================================================
# MAIN
# =========================================================
def main():
    cfg = CFG()
    set_seed(cfg.seed)

    device = get_device()

    print("=" * 70)
    print(f"🚀 Device: {device}")
    print(f"📦 Metadata: {PROCESSED_METADATA}")
    print(f"🎵 Modelo: SimpleMFCCCNN (solo MFCC)")
    print("=" * 70)

    train_ds, test_ds, train_loader, test_loader = build_loaders(cfg)

    num_classes = len(train_ds.label_mapping)

    print(f"📊 Dataset Info:")
    print(f"   - Train samples: {len(train_ds)}")
    print(f"   - Test samples: {len(test_ds)}")
    print(f"   - Clases: {num_classes}")
    print(f"   - Batch size: {cfg.batch_size}")
    print(f"   - Epochs: {cfg.epochs}")
    print(f"   - LR: {cfg.lr}")
    print("=" * 70)

  

    model = HybridAudioClassifier(
        num_classes=num_classes,
        mode=cfg.mode,
        branch_dim=128,
        hidden_dim=256,
        dropout=0.25,
    ).to(device)

    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Modelo Info:")
    print(f"   - Total params: {total_params:,}")
    print(f"   - Trainable params: {trainable_params:,}")
    print("=" * 70)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = 0.0
    best_epoch = 0

    # =====================================================
    # TRAIN LOOP
    # =====================================================
    print("\n🔥 INICIANDO ENTRENAMIENTO\n")
    
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        print(f"\n📌 Epoch {epoch}/{cfg.epochs}")
        print("-" * 70)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg, epoch
        )

        print(f"\n  📊 Train Summary:")
        print(f"     Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        print(f"\n  🧪 Validación:")
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, cfg, epoch
        )

        print(f"\n  📊 Val Summary:")
        print(f"     Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")

        dt = time.time() - t0

        print(f"\n⏱️  Tiempo epoch: {dt:.1f}s")

        # guardar mejor modelo
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "best_mfcc_cnn.pt")
            print(f"💾 Nuevo mejor modelo guardado (acc: {best_acc:.4f})")

        print("-" * 70)

    print("\n" + "=" * 70)
    print(f"✅ Entrenamiento terminado!")
    print(f"🏆 Best Epoch: {best_epoch} | Best Acc: {best_acc:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
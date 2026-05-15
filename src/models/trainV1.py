from __future__ import annotations
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.hybrid_cnn import ImprovedMFCCCNN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import glob
from src.utils.config import PROCESSED_METADATA,PROCESSED_METADATA_SPLIT_FIX, LABEL_MAPPING,CHECKPOINT_DIR, FINAL_MODEL_DIR
from src.models.audio_dataset import ProcessedAudioDataset, crnn_collate_fn



class CFG:
    batch_size = 32
    lr = 3e-4
    weight_decay = 1e-2
    epochs = 12
    num_workers = 11
    use_mfcc = True
    use_scalars = False
    seed = 42
    print_every = 50
    target_type = "human_label"
    mode = "mel_only"

    # 🔥 NUEVO
    checkpoint_dir = CHECKPOINT_DIR
    save_every = 2   # cada X épocas
def save_checkpoint(cfg, model, optimizer, epoch, history, best_acc, best_epoch):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
    }, path)

    print(f"💾 Checkpoint guardado: {path}")
def load_latest_checkpoint(cfg, model, optimizer):
    if not cfg.checkpoint_dir or not os.path.exists(cfg.checkpoint_dir):
        return None, None, None, None

    checkpoints = glob.glob(os.path.join(cfg.checkpoint_dir, "checkpoint_epoch_*.pt"))

    if not checkpoints:
        return None, None, None, None

    latest = max(checkpoints, key=os.path.getctime)
    print(f"♻️ Reanudando desde: {latest}")

    ckpt = torch.load(latest, map_location="cpu")

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    return (
        ckpt["epoch"],
        ckpt["history"],
        ckpt["best_acc"],
        ckpt["best_epoch"]
    )
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
        metadata_csv=PROCESSED_METADATA_SPLIT_FIX,
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
        metadata_csv=PROCESSED_METADATA_SPLIT_FIX,
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


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg, epoch):
    model.eval()

    total_loss = 0
    y_true, y_pred = [], []

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

    # =====================================================
    # 🔥 SKLEARN METRICS REALES
    # =====================================================
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",   # importante en clasificación multiclase
        zero_division=0
    )

    return epoch_loss, epoch_acc, precision, recall, f1


# =========================================================
# MAIN
# =========================================================
# =========================================================
# MAIN
# =========================================================
def main():

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "lr": [],
        "epoch_time": [],
        "images_per_sec": []
    }

    cfg = CFG()

    set_seed(cfg.seed)

    device = get_device()

    print("=" * 70)
    print(f"🚀 Device: {device}")
    print(f"📦 Metadata: {PROCESSED_METADATA_SPLIT_FIX}")
    print(f"🎵 Modelo: ImprovedMFCCCNN")
    print("=" * 70)

    # =====================================================
    # DATASETS
    # =====================================================
    train_ds, test_ds, train_loader, test_loader = build_loaders(cfg)

    # =====================================================
    # 🔥 SPLIT VALIDACIÓN DESDE TRAIN
    # =====================================================
    from torch.utils.data import random_split

    val_size = int(len(train_ds) * 0.1)
    train_size = len(train_ds) - val_size

    train_subset, val_subset = random_split(
        train_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=crnn_collate_fn,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=crnn_collate_fn,
        pin_memory=True,
    )

    num_classes = len(train_ds.label_mapping)

    print(f"📊 Dataset Info:")
    print(f"   - Train samples: {len(train_subset)}")
    print(f"   - Val samples: {len(val_subset)}")
    print(f"   - Test samples: {len(test_ds)}")
    print(f"   - Clases: {num_classes}")
    print(f"   - Batch size: {cfg.batch_size}")
    print(f"   - Epochs: {cfg.epochs}")
    print(f"   - LR inicial: {cfg.lr}")
    print("=" * 70)

    # =====================================================
    # MODEL
    # =====================================================
    model = ImprovedMFCCCNN(
        num_classes=num_classes,
        dropout=0.25,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"\n🧠 Modelo Info:")
    print(f"   - Total params: {total_params:,}")
    print(f"   - Trainable params: {trainable_params:,}")
    print("=" * 70)

    # =====================================================
    # LOSS
    # =====================================================
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05
    )

    # =====================================================
    # OPTIMIZER
    # =====================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # =====================================================
    # SCHEDULER
    # =====================================================
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    # =====================================================
    # CHECKPOINT LOAD
    # =====================================================
    start_epoch = 0
    best_acc = 0.0
    best_epoch = 0

    ckpt_epoch, ckpt_history, ckpt_best_acc, ckpt_best_epoch = (
        load_latest_checkpoint(
            cfg,
            model,
            optimizer
        )
    )

    if ckpt_epoch is not None:
        start_epoch = ckpt_epoch
        history = ckpt_history
        best_acc = ckpt_best_acc
        best_epoch = ckpt_best_epoch
    # =====================================================
    # FIX HISTORY COMPATIBILITY
    # =====================================================
    required_keys = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "precision",
        "recall",
        "f1",
        "lr",
        "epoch_time",
        "images_per_sec"
    ]

    # convertir antiguos test_* -> val_*
    if "test_loss" in history and "val_loss" not in history:
        history["val_loss"] = history["test_loss"]

    if "test_acc" in history and "val_acc" not in history:
        history["val_acc"] = history["test_acc"]

    # crear claves faltantes
    for key in required_keys:
        if key not in history:
            history[key] = []
    print(f"🔄 Reanudado desde epoch {start_epoch}")

    # =====================================================
    # TRAIN LOOP
    # =====================================================
    print("\n🔥 INICIANDO ENTRENAMIENTO\n")

    for epoch in range(start_epoch + 1, cfg.epochs + 1):

        t0 = time.time()

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n📌 Epoch {epoch}/{cfg.epochs}")
        print(f"📉 LR actual: {current_lr:.8f}")
        print("-" * 70)

        # =================================================
        # TRAIN
        # =================================================
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            cfg,
            epoch
        )

        # =================================================
        # VALIDATION
        # =================================================
        val_loss, val_acc, precision, recall, f1 = evaluate(
            model,
            val_loader,
            criterion,
            device,
            cfg,
            epoch
        )

        # =================================================
        # SCHEDULER STEP
        # =================================================
        scheduler.step(val_acc)

        dt = time.time() - t0

        images_per_sec = len(train_loader.dataset) / dt

        # =================================================
        # HISTORY
        # =================================================
        history["epoch"].append(epoch)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))

        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        history["precision"].append(float(precision))
        history["recall"].append(float(recall))
        history["f1"].append(float(f1))

        history["lr"].append(float(current_lr))

        history["epoch_time"].append(float(dt))
        history["images_per_sec"].append(float(images_per_sec))

        # =================================================
        # LOGS
        # =================================================
        print("\n📈 RESULTADOS")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")

        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")

        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print(f"F1 Score   : {f1:.4f}")

        print(f"⏱️ Epoch time: {dt:.1f}s")

        # =================================================
        # BEST MODEL
        # =================================================
        if val_acc > best_acc:

            best_acc = val_acc
            best_epoch = epoch

            os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

            torch.save(
                model.state_dict(),
                FINAL_MODEL_DIR / "best_mfcc_cnn.pt"
            )

            print("💾 Best model updated")

        # =================================================
        # CHECKPOINT SAVE
        # =================================================
        if epoch % cfg.save_every == 0:

            save_checkpoint(
                cfg,
                model,
                optimizer,
                epoch,
                history,
                best_acc,
                best_epoch
            )

    # =====================================================
    # FINAL TEST EVALUATION
    # =====================================================
    print("\n🧪 EVALUACIÓN FINAL EN TEST")

    best_model = ImprovedMFCCCNN(
        num_classes=num_classes,
        dropout=0.25,
    ).to(device)

    best_model.load_state_dict(
        torch.load(
            FINAL_MODEL_DIR / "best_mfcc_cnn.pt",
            map_location=device
        )
    )

    test_loss, test_acc, precision, recall, f1 = evaluate(
        best_model,
        test_loader,
        criterion,
        device,
        cfg,
        epoch="TEST"
    )

    print("\n🏁 RESULTADOS FINALES TEST")
    print(f"Test Loss : {test_loss:.4f}")
    print(f"Test Acc  : {test_acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print(f"\n🥇 Mejor epoch: {best_epoch}")
    print(f"🥇 Mejor val_acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()
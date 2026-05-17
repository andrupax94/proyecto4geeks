from __future__ import annotations

import os
import time
import random
import glob
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from src.models.hybrid_cnn import ImprovedMFCCCNN
from src.utils.config import PROCESSED_METADATA, LABEL_MAPPING, CHECKPOINT_DIR, FINAL_MODEL_DIR
from src.models.audio_dataset import ProcessedAudioDataset, crnn_collate_fn


class CFG:
    batch_size = 32
    lr = 3e-4
    finetune_lr = 1e-5
    weight_decay = 1e-2
    epochs = 12
    num_workers = 4
    use_mfcc = True
    use_scalars = False
    seed = 42
    print_every = 50
    target_type = "human_label"
    mode = "mel_only"

    checkpoint_dir = CHECKPOINT_DIR
    save_every = 2
    use_amp = False
    prefetch_factor = 1
    persistent_workers = False

    # Focus fine-tuning
    focus_labels = ["fire", "explosion"]  # None o [] para desactivar
    focus_boost = 8.0
    use_focused_loss_weights = True
    oversample_replacement = True
    focused_eval_report = True

    # Optional
    freeze_backbone = False
    freeze_classifier = False


def save_checkpoint(cfg, model, optimizer, epoch, history, best_acc, best_epoch, scheduler=None):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()

    torch.save(payload, path)
    print(f"💾 Checkpoint guardado: {path}")


def load_latest_checkpoint(cfg, model, optimizer, scheduler=None):
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

    if scheduler is not None and "scheduler_state" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception as e:
            print(f"⚠️ No se pudo restaurar scheduler: {e}")

    return ckpt["epoch"], ckpt["history"], ckpt["best_acc"], ckpt["best_epoch"]


def prepare_model_inputs(batch, device, mode: str):
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
# DEVICE
# =========================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# HELPERS FOCUS MODE
# =========================================================
def _resolve_label_to_idx(label_mapping):
    if isinstance(label_mapping, dict):
        if all(isinstance(k, str) for k in label_mapping.keys()):
            return dict(label_mapping)
        if all(isinstance(v, str) for v in label_mapping.values()):
            return {v: int(k) for k, v in label_mapping.items()}

    if isinstance(label_mapping, (list, tuple)):
        return {str(name): i for i, name in enumerate(label_mapping)}

    raise TypeError("No pude inferir el formato de label_mapping.")


def _extract_label_id(sample_label):
    if torch.is_tensor(sample_label):
        return int(sample_label.item())
    return int(sample_label)


def _build_sample_weights_from_subset(train_subset, focus_label_names, label_to_idx, focus_boost):
    focus_ids = {label_to_idx[name] for name in focus_label_names if name in label_to_idx}
    if not focus_ids:
        print("⚠️ focus_labels no coincide con ninguna clase del mapping. Se desactiva el foco.")
        return None, set()

    weights = []
    label_counter = Counter()

    for idx in train_subset.indices:
        _, label, _ = train_subset.dataset[idx]
        label_id = _extract_label_id(label)
        label_counter[label_id] += 1
        weights.append(float(focus_boost) if label_id in focus_ids else 1.0)

    print("📊 Distribución del train subset:")
    for label_id, count in sorted(label_counter.items(), key=lambda x: x[0]):
        mark = "⭐" if label_id in focus_ids else " "
        print(f"   {mark} label_id={label_id}: {count}")

    print(f"🎯 focus_ids: {sorted(focus_ids)}")
    print(f"🎚️ focus_boost: {focus_boost}")
    return weights, focus_ids


def _build_class_weights(num_classes: int, train_subset, focus_ids=None, focus_multiplier=1.0):
    counts = Counter()
    for idx in train_subset.indices:
        _, label, _ = train_subset.dataset[idx]
        label_id = _extract_label_id(label)
        counts[label_id] += 1

    class_weights = np.zeros(num_classes, dtype=np.float32)
    total = sum(counts.values())
    if total == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        n = counts.get(c, 0)
        if n > 0:
            class_weights[c] = total / (num_classes * n)
        else:
            class_weights[c] = 0.0

    if focus_ids:
        for c in focus_ids:
            class_weights[c] *= float(focus_multiplier)

    class_weights = np.clip(class_weights, 0.1, 10.0)
    return torch.tensor(class_weights, dtype=torch.float32)


# =========================================================
# DATALOADERS
# =========================================================
def build_loaders(cfg: CFG):
    if cfg.target_type == "human_label":
        label_mapping_path = LABEL_MAPPING["human_label"]
    elif cfg.target_type == "alertable":
        label_mapping_path = LABEL_MAPPING["alertable"]
    else:
        raise ValueError(f"target_type inválido: {cfg.target_type}")

    print(f"
🎯 Target seleccionado: {cfg.target_type}")
    print(f"🗂️ Label mapping: {label_mapping_path}")

    print("📦 Cargando dataset de entrenamiento...")
    train_ds = ProcessedAudioDataset(
        metadata_csv=PROCESSED_METADATA,
        label_mapping_path=label_mapping_path,
        split="train",
        target_column=cfg.target_type,
        use_mfcc=cfg.use_mfcc,
        use_scalars=cfg.use_scalars,
    )
    print(f"✅ Dataset de entrenamiento cargado: {len(train_ds)} muestras")

    print("📦 Cargando dataset de test...")
    test_ds = ProcessedAudioDataset(
        metadata_csv=PROCESSED_METADATA,
        label_mapping_path=label_mapping_path,
        split="test",
        target_column=cfg.target_type,
        use_mfcc=cfg.use_mfcc,
        use_scalars=cfg.use_scalars,
        target_frames=train_ds.target_frames,
    )
    print(f"✅ Dataset de test cargado: {len(test_ds)} muestras")

    print("🔧 Creando dataloaders base...")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=crnn_collate_fn,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        timeout=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=crnn_collate_fn,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        timeout=0,
    )

    print(f"✅ Dataloaders creados correctamente")
    return train_ds, test_ds, train_loader, test_loader


# =========================================================
# TRAIN
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device, cfg, epoch, scaler=None):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []
    num_batches = len(loader)

    for batch_idx, (batch, labels, _) in enumerate(loader, 1):
        labels = labels.to(device)
        inputs = prepare_model_inputs(batch, device, cfg.mode)

        optimizer.zero_grad(set_to_none=True)

        if cfg.use_amp and scaler is not None:
            with autocast(device_type=device.type, enabled=True):
                outputs = model(*inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())

        if batch_idx % cfg.print_every == 0:
            avg_loss = total_loss / batch_idx
            avg_acc = accuracy_score(y_true, y_pred)
            print(f"  Epoch {epoch} | Batch {batch_idx}/{num_batches} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

    epoch_loss = total_loss / max(num_batches, 1)
    epoch_acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg, epoch, class_names=None, return_preds=False):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    num_batches = len(loader)

    with torch.inference_mode():
        for batch_idx, (batch, labels, _) in enumerate(loader, 1):
            labels = labels.to(device)
            inputs = prepare_model_inputs(batch, device, cfg.mode)
            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            total_loss += float(loss.item())

            preds = outputs.argmax(dim=1)
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())

            if batch_idx % cfg.print_every == 0:
                avg_loss = total_loss / batch_idx
                avg_acc = accuracy_score(y_true, y_pred)
                print(f"  Epoch {epoch} | Val Batch {batch_idx}/{num_batches} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

    epoch_loss = total_loss / max(num_batches, 1)
    epoch_acc = accuracy_score(y_true, y_pred) if y_true else 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = None
    cm = None
    if class_names is not None and len(class_names) > 0:
        try:
            report = classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                zero_division=0,
                digits=4,
            )
            cm = confusion_matrix(y_true, y_pred)
        except Exception as e:
            print(f"⚠️ No se pudo generar report/confusion matrix: {e}")

    if return_preds:
        return epoch_loss, epoch_acc, precision, recall, f1, y_true, y_pred, report, cm

    return epoch_loss, epoch_acc, precision, recall, f1, report, cm


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
        "images_per_sec": [],
    }

    cfg = CFG()
    set_seed(cfg.seed)
    device = get_device()

    print("=" * 70)
    print(f"🚀 Device: {device}")
    print(f"📦 Metadata: {PROCESSED_METADATA}")
    print(f"🎵 Modelo: ImprovedMFCCCNN")
    print(f"🔥 num_workers: {cfg.num_workers}")
    print(f"🔥 batch_size: {cfg.batch_size}")
    print(f"🔥 AMP (Mixed Precision): {cfg.use_amp}")
    print(f"🔥 prefetch_factor: {cfg.prefetch_factor}")
    print(f"🔥 persistent_workers: {cfg.persistent_workers}")
    print(f"🎯 focus_labels: {cfg.focus_labels}")
    print(f"🎚️ focus_boost: {cfg.focus_boost}")
    print("=" * 70)

    # DATASETS
    train_ds, test_ds, train_loader, test_loader = build_loaders(cfg)

    # VALIDACIÓN DESDE TRAIN
    print("
🔀 Creando split de validación...")
    val_size = int(len(train_ds) * 0.1)
    train_size = len(train_ds) - val_size

    train_subset, val_subset = random_split(
        train_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    num_classes = len(train_ds.label_mapping)
    label_to_idx = _resolve_label_to_idx(train_ds.label_mapping)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    class_names = [idx_to_label[i] if i in idx_to_label else str(i) for i in range(num_classes)]

    print(f"📊 Dataset Info:")
    print(f"   - Train samples: {len(train_subset)}")
    print(f"   - Val samples: {len(val_subset)}")
    print(f"   - Test samples: {len(test_ds)}")
    print(f"   - Clases: {num_classes}")
    print(f"   - Batch size: {cfg.batch_size}")
    print(f"   - Epochs: {cfg.epochs}")
    print(f"   - LR inicial: {cfg.lr}")
    print("=" * 70)

    # DATALOADER DE ENTRENAMIENTO (con focus opcional)
    train_sampler = None
    class_weights = None
    focus_ids = set()

    if cfg.focus_labels:
        focus_info = _build_sample_weights_from_subset(
            train_subset=train_subset,
            focus_label_names=cfg.focus_labels,
            label_to_idx=label_to_idx,
            focus_boost=cfg.focus_boost,
        )
        if focus_info is not None:
            sample_weights, focus_ids = focus_info
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=cfg.oversample_replacement,
            )
            print("✅ WeightedRandomSampler activado para focus_labels")

            if cfg.use_focused_loss_weights:
                class_weights = _build_class_weights(
                    num_classes=num_classes,
                    train_subset=train_subset,
                    focus_ids=focus_ids,
                    focus_multiplier=cfg.focus_boost,
                )
                print(f"✅ class_weights activados: {class_weights.tolist()}")

    if train_sampler is None:
        train_loader = DataLoader(
            train_subset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=crnn_collate_fn,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=cfg.prefetch_factor,
            persistent_workers=cfg.persistent_workers,
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=crnn_collate_fn,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=cfg.prefetch_factor,
            persistent_workers=cfg.persistent_workers,
        )

    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=crnn_collate_fn,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
    )

    # MODEL
    print("
🧠 Inicializando modelo...")
    model = ImprovedMFCCCNN(
        num_classes=num_classes,
        dropout=0.25,
    ).to(device)

    if cfg.freeze_backbone and hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("🧊 Backbone congelado")

    if cfg.freeze_classifier:
        for name, param in model.named_parameters():
            if any(k in name.lower() for k in ["classifier", "head", "fc"]):
                param.requires_grad = False
        print("🧊 Classifier congelado")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"🧠 Modelo Info:")
    print(f"   - Total params: {total_params:,}")
    print(f"   - Trainable params: {trainable_params:,}")
    print("=" * 70)

    # LOSS
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=0.03,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # OPTIMIZER
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # SCHEDULER
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = GradScaler() if cfg.use_amp else None

    # CHECKPOINT LOAD
    start_epoch = 0
    best_acc = 0.0
    best_epoch = 0

    ckpt_epoch, ckpt_history, ckpt_best_acc, ckpt_best_epoch = load_latest_checkpoint(
        cfg,
        model,
        optimizer,
        scheduler=scheduler,
    )

    if ckpt_epoch is not None:
        start_epoch = ckpt_epoch
        history = ckpt_history
        best_acc = ckpt_best_acc
        best_epoch = ckpt_best_epoch

        # Reanudar con LR de fine-tuning si estás en focus mode.
        resumed_lr = cfg.finetune_lr if cfg.focus_labels else cfg.lr
        for pg in optimizer.param_groups:
            pg["lr"] = resumed_lr
        print(f"🎯 LR ajustado al reanudar: {resumed_lr}")

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
        "images_per_sec",
    ]

    if "test_loss" in history and "val_loss" not in history:
        history["val_loss"] = history["test_loss"]
    if "test_acc" in history and "val_acc" not in history:
        history["val_acc"] = history["test_acc"]

    for key in required_keys:
        if key not in history:
            history[key] = []

    print(f"🔄 Reanudado desde epoch {start_epoch}")

    # DEBUG del dataloader
    print("
🧪 Probando dataloader...")
    try:
        t0 = time.time()
        for batch, labels, filenames in train_loader:
            dt = time.time() - t0
            print(f"  ✅ Primer batch obtenido en {dt:.2f}s")
            if isinstance(batch, dict) and "mel" in batch:
                print(f"     - MEL shape: {batch['mel'].shape}")
            print(f"     - Labels shape: {labels.shape}")
            break
    except Exception as e:
        print(f"  ❌ Error al cargar batch: {e}")
        print("  🔧 Intenta reducir num_workers o batch_size")
        return

    # TRAIN LOOP
    print("
🔥 INICIANDO ENTRENAMIENTO
")

    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"
📌 Epoch {epoch}/{cfg.epochs}")
        print(f"📉 LR actual: {current_lr:.8f}")
        print("-" * 70)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            cfg,
            epoch,
            scaler=scaler,
        )

        val_loss, val_acc, precision, recall, f1, report, cm = evaluate(
            model,
            val_loader,
            criterion,
            device,
            cfg,
            epoch,
            class_names=class_names,
            return_preds=False,
        )

        scheduler.step(val_acc)
        dt = time.time() - t0
        images_per_sec = len(train_loader.dataset) / max(dt, 1e-9)

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

        print("
📈 RESULTADOS")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")
        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print(f"F1 Score   : {f1:.4f}")
        print(f"⏱️ Epoch time: {dt:.1f}s ({images_per_sec:.0f} img/s)")

        if cfg.focused_eval_report and report is not None:
            print("
🧾 Classification report (val)")
            print(report)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), Path(FINAL_MODEL_DIR) / "best_mfcc_cnn.pt")
            print("💾 Best model updated")

        if epoch % cfg.save_every == 0:
            save_checkpoint(
                cfg,
                model,
                optimizer,
                epoch,
                history,
                best_acc,
                best_epoch,
                scheduler=scheduler,
            )

    # FINAL TEST
    print("
🧪 EVALUACIÓN FINAL EN TEST")

    best_model = ImprovedMFCCCNN(
        num_classes=num_classes,
        dropout=0.25,
    ).to(device)

    best_model.load_state_dict(
        torch.load(
            Path(FINAL_MODEL_DIR) / "best_mfcc_cnn.pt",
            map_location=device,
        )
    )

    test_loss, test_acc, precision, recall, f1, report, cm = evaluate(
        best_model,
        test_loader,
        criterion,
        device,
        cfg,
        epoch="TEST",
        class_names=class_names,
        return_preds=False,
    )

    print("
🏁 RESULTADOS FINALES TEST")
    print(f"Test Loss : {test_loss:.4f}")
    print(f"Test Acc  : {test_acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    if report is not None:
        print("
🧾 Classification report (test)")
        print(report)

    print(f"
🥇 Mejor epoch: {best_epoch}")
    print(f"🥇 Mejor val_acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()

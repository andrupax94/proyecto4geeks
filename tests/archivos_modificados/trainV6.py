from __future__ import annotations

import os
import re
import time
import random
import glob
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

from src.models.hybrid_cnn_v4 import HybridCNNV5, build_criterion
from src.utils.config import PROCESSED_METADATA, LABEL_MAPPING, CHECKPOINT_DIR, FINAL_MODEL_DIR
from src.models.audio_dataset import ProcessedAudioDataset, crnn_collate_fn


class CFG:
    # ─────────────────────────────────────────────────────────
    # TRAIN
    # ─────────────────────────────────────────────────────────
    
    batch_size =  48 # Alertable :64   
    lr = 3e-4   # Alertable :3e-4      
    weight_decay = 1e-4 # Alertable :1e-4
    dropout = 0.40 # Alertable :0.25
    task="binary"
    target_type = "alertable"
    
    # batch_size = 48
    # lr = 2e-4
    # weight_decay = 2e-4
    # dropout = 0.30
    # task="multiclass"
    # target_type = "human_label"
    
    epochs = 28
    num_workers = 6
    # ─────────────────────────────────────────────────────────
    # DATASET
    # ─────────────────────────────────────────────────────────
    use_mfcc = True
    use_scalars = False
    seed = 42
    print_every = 50
    # target_type = "alertable"
   
    mode = "mel_waveform"
    label_version = 2
    version = 4
    checkpoint_dir = CHECKPOINT_DIR / f"{target_type}_V{version}"

    # ─────────────────────────────────────────────────────────
    # GPU
    # ─────────────────────────────────────────────────────────
    use_amp = True
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 2

    # ─────────────────────────────────────────────────────────
    # CHECKPOINTS
    # ─────────────────────────────────────────────────────────
    save_every = 1  # mejor cada epoch si quieres reanudar fino

    # Si no tienes scheduler guardado, puedes meter aquí el LR exacto
    # con el que quieres continuar.
    resume_lr = None #2.61e-06  # usa el último LR confiable que quieras retomar

    # ─────────────────────────────────────────────────────────
    # FOCUS CLASSES
    # ─────────────────────────────────────────────────────────
    # focus_classes: list = ["fire","traffic","siren_alarm"]
    focus_classes: list = [True]
    focus_loss_weight: float = 3.5
    focus_oversample_factor: float = 2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    """Fuerza el LR actual del optimizador y lo deja consistente para futuras cargas."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        param_group["initial_lr"] = lr


def get_optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _extract_epoch(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"checkpoint_epoch_(\d+)\.pt$", name)
    return int(m.group(1)) if m else -1


def save_checkpoint(cfg, model, optimizer, scaler, scheduler, epoch, history, best_acc, best_epoch):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

    checkpoint_dict = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "dropout": cfg.dropout,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    if scaler is not None:
        checkpoint_dict["scaler_state"] = scaler.state_dict()

    torch.save(checkpoint_dict, path)
    print(f"💾 Checkpoint guardado: {path}")


def load_latest_checkpoint(cfg, model, optimizer, scaler=None, scheduler=None):
    if not cfg.checkpoint_dir or not os.path.exists(cfg.checkpoint_dir):
        return None, None, None, None

    checkpoints = glob.glob(os.path.join(cfg.checkpoint_dir, "checkpoint_epoch_*.pt"))
    if not checkpoints:
        return None, None, None, None

    latest = max(checkpoints, key=_extract_epoch)
    print(f"♻️ Reanudando desde: {latest}")

    # IMPORTANTE: checkpoint antiguo con objetos no permitidos por weights_only=True
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    device = next(model.parameters()).device
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    rng = ckpt.get("rng_state")
    if rng is not None:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])

    history = ckpt["history"]
    # Asegurar que las nuevas métricas existan si se carga un checkpoint antiguo
    if "auc_roc" not in history:
        history["auc_roc"] = [0.0] * len(history["epoch"])
    
    return (
        ckpt["epoch"],
        history,
        ckpt["best_acc"],
        ckpt["best_epoch"]
    )


def prepare_model_inputs(batch, device, mode: str):
    mel = batch.get("mel", None)
    mfcc = batch.get("mfcc", None)
    waveform = batch.get("waveform", None)

    if mel is not None:
        mel = mel.to(device, non_blocking=True)
    if mfcc is not None:
        mfcc = mfcc.to(device, non_blocking=True)
    if waveform is not None:
        waveform = waveform.to(device, non_blocking=True)

    if mode == "mel_only":
        if mel is None:
            raise KeyError("El batch no contiene 'mel'")
        return mel, None, None

    if mode == "mel_mfcc":
        if mel is None or mfcc is None:
            raise KeyError("El batch no contiene 'mel' o 'mfcc'")
        return mel, mfcc, None

    if mode == "mel_waveform":
        if mel is None or waveform is None:
            raise KeyError("El batch no contiene 'mel' o 'waveform'")
        return mel, None, waveform

    raise ValueError("mode debe ser 'mel_only', 'mel_mfcc' o 'mel_waveform'")


def resolve_focus_class_indices(focus_classes: list, label_mapping: dict) -> list[int]:
    if not focus_classes:
        return []

    if label_mapping and isinstance(next(iter(label_mapping)), int):
        name_to_idx = {v: k for k, v in label_mapping.items()}
    else:
        name_to_idx = label_mapping

    indices = []
    for cls in focus_classes:
        if cls in name_to_idx:
            indices.append(name_to_idx[cls])
        else:
            print(f"⚠️  Focus class '{cls}' no encontrada. Se ignora.")
    return indices


def build_focus_loss_weights(
    num_classes: int,
    focus_indices: list[int],
    focus_weight: float,
    device: torch.device,
) -> torch.Tensor | None:
    if not focus_indices:
        return None

    weights = torch.ones(num_classes, dtype=torch.float32)
    for idx in focus_indices:
        weights[idx] = focus_weight
    return weights.to(device)


def build_focus_sampler(dataset, focus_indices: list[int], focus_factor: float) -> WeightedRandomSampler | None:
    if not focus_indices:
        return None

    if hasattr(dataset, "labels"):
        labels = list(dataset.labels)

    elif hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_ds = dataset.dataset
        indices = dataset.indices

        if hasattr(base_ds, "labels"):
            labels = [base_ds.labels[i] for i in indices]
        elif hasattr(base_ds, "df") and hasattr(base_ds, "label_mapping"):
            target_col = base_ds.target_column
            label_mapping = base_ds.label_mapping
            raw_labels = base_ds.df.iloc[indices][target_col].tolist()

            labels = []
            for raw in raw_labels:
                key = str(raw).strip() if raw is not None else raw
                if key in label_mapping:
                    labels.append(int(label_mapping[key]))
                elif raw in label_mapping:
                    labels.append(int(label_mapping[raw]))
                else:
                    try:
                        labels.append(int(raw))
                    except (ValueError, TypeError):
                        labels.append(0)
        else:
            print("⚠️  No se pudo extraer etiquetas. Oversample desactivado.")
            return None
    else:
        print("⚠️  No se pudo extraer .labels. Oversample desactivado.")
        return None

    focus_set = set(focus_indices)
    sample_weights = [
        focus_factor if int(lbl) in focus_set else 1.0
        for lbl in labels
    ]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def print_focus_class_report(
    y_true: list,
    y_pred: list,
    focus_indices: list[int],
    label_mapping: dict,
    header: str = "🎯 FOCUS CLASS REPORT",
):
    if not focus_indices:
        return

    if label_mapping and isinstance(next(iter(label_mapping)), str):
        idx_to_name = {v: k for k, v in label_mapping.items()}
    else:
        idx_to_name = label_mapping

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true_np, y_pred_np, labels=focus_indices, zero_division=0
    )

    print(f"\n{'=' * 70}")
    print(header)
    print(f"{'=' * 70}")
    print(f"{'Clase':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Soporte':>10}")
    print("-" * 70)

    for i, idx in enumerate(focus_indices):
        name = idx_to_name.get(idx, f"class_{idx}")
        print(
            f"{name:<25} {precision_per[i]:>10.4f} {recall_per[i]:>10.4f} "
            f"{f1_per[i]:>10.4f} {int(support_per[i]):>10}"
        )

    cm = confusion_matrix(y_true_np, y_pred_np, labels=focus_indices)
    class_names = [idx_to_name.get(idx, f"class_{idx}") for idx in focus_indices]

    print(f"\n📊 Sub-matriz de confusión (focus classes):")
    header_row = f"{'':>20}" + "".join(f"{n:>12}" for n in class_names)
    print(header_row)
    for i, name in enumerate(class_names):
        row_str = f"{name:>20}" + "".join(f"{cm[i, j]:>12}" for j in range(len(class_names)))
        print(row_str)

    print("=" * 70)


def build_loaders(cfg: CFG):
    column_target=cfg.target_type
    if cfg.target_type == "human_label":
        label_mapping_path = LABEL_MAPPING[f"human_label{cfg.label_version}"]
    elif cfg.target_type == "alertable":
        label_mapping_path = LABEL_MAPPING[f"alertable{cfg.label_version}"]
    elif cfg.target_type == "no_alertable":
        column_target="human_label"
        label_mapping_path = LABEL_MAPPING[f"human_label_no_alertable{cfg.label_version}"]
    elif cfg.target_type == "emergency":
        label_mapping_path = LABEL_MAPPING[f"emergency{cfg.label_version}"]
    elif cfg.target_type == "total":
        label_mapping_path = LABEL_MAPPING[f"total{cfg.label_version}"]
    else:
        raise ValueError(f"target_type inválido: {cfg.target_type}")
    
    print(f"\n🎯 Target seleccionado: {cfg.target_type}")
    print(f"🗂️ Label mapping: {label_mapping_path}")

    print("📦 Cargando dataset de entrenamiento...")
    train_ds = ProcessedAudioDataset(
        metadata_csv=PROCESSED_METADATA[str(cfg.label_version)],
        label_mapping_path=label_mapping_path,
        split="train",
        target_column=column_target,
        use_mfcc=cfg.use_mfcc,
        use_scalars=cfg.use_scalars,
    )
    print(f"✅ Dataset de entrenamiento cargado: {len(train_ds)} muestras")

    print("📦 Cargando dataset de test...")
    test_ds = ProcessedAudioDataset(
        metadata_csv=PROCESSED_METADATA[str(cfg.label_version)],
        label_mapping_path=label_mapping_path,
        split="test",
        target_column=column_target,
        use_mfcc=cfg.use_mfcc,
        use_scalars=cfg.use_scalars,
        target_frames=train_ds.target_frames,
    )
    print(f"✅ Dataset de test cargado: {len(test_ds)} muestras")

    print("🔧 Creando dataloaders...")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=crnn_collate_fn,
        drop_last=True,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        timeout=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=crnn_collate_fn,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        timeout=0,
    )

    train_labels = collections.Counter(train_ds.df[column_target].tolist())
    test_labels = collections.Counter(test_ds.df[column_target].tolist())

    print(f"✅ Dataloaders creados correctamente")
    print(f"Proporciones Train: {train_labels}")
    print(f"Proporciones Test: {test_labels}")

    return train_ds, test_ds, train_loader, test_loader


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    cfg,
    epoch,
    scaler=None,
    scheduler=None,
):
    model.train()

    total_loss = 0
    y_true, y_pred = [], []
    num_batches = len(loader)
    batch_times = []

    for batch_idx, (batch, labels, _) in enumerate(loader, 1):
        batch_start = time.time()

        if cfg.task == "binary":
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
        else:
            labels = labels.long().to(device, non_blocking=True)
        mel, mfcc, waveform = prepare_model_inputs(batch, device, cfg.mode)

        optimizer.zero_grad(set_to_none=True)

        if cfg.use_amp:
            with autocast(device_type="cuda"):
                outputs = model(mel=mel, mfcc=mfcc, waveform=waveform)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(mel=mel, mfcc=mfcc, waveform=waveform)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if cfg.task == "binary":
            y_true.extend(labels.squeeze(1).long().cpu().numpy())
        else:
            y_true.extend(labels.cpu().numpy())
        preds = model.predict(mel, mfcc, waveform)
        y_pred.extend(preds.cpu().numpy())

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        if batch_idx % cfg.print_every == 0:
            avg_loss = total_loss / batch_idx
            avg_acc = accuracy_score(y_true, y_pred)
            current_lr = get_optimizer_lr(optimizer)
            avg_batch_time = np.mean(batch_times[-cfg.print_every:])
            img_per_sec = cfg.batch_size / avg_batch_time

            print(
                f"  Epoch {epoch} | Batch {batch_idx}/{num_batches} | "
                f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | "
                f"LR: {current_lr:.2e} | {img_per_sec:.0f} img/s"
            )

    epoch_loss = total_loss / num_batches
    epoch_acc = accuracy_score(y_true, y_pred)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    cfg,
    epoch,
    focus_indices: list = None,
    label_mapping: dict = None,
    show_focus_report: bool = True,
):
    model.eval()

    total_loss = 0
    y_true, y_pred, y_probs = [], [], []
    num_batches = len(loader)

    with torch.inference_mode():
        for batch_idx, (batch, labels, _) in enumerate(loader, 1):
            if cfg.task == "binary":
                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
            else:
                labels = labels.long().to(device, non_blocking=True)
            mel, mfcc, waveform = prepare_model_inputs(batch, device, cfg.mode)

            outputs = model(mel=mel, mfcc=mfcc, waveform=waveform)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            if cfg.task == "binary":
                y_true.extend(labels.squeeze(1).long().cpu().numpy())
                probs = torch.sigmoid(outputs)
                y_probs.extend(probs.cpu().numpy())
                
                # Threshold de 0.75 solicitado para binario
                # Si prob < 0.75 y prob > 0.25 (zona incierta), podríamos marcarlo, 
                # pero aquí el usuario pide que si no sabe lo clasifique como unknown.
                # Para la evaluación estándar, usaremos un threshold de 0.75 para la clase positiva.
                # Sin embargo, para mantener compatibilidad con métricas, calculamos y_pred normal
                # pero informamos del impacto del threshold si se desea.
            else:
                y_true.extend(labels.cpu().numpy())
                probs = torch.softmax(outputs, dim=1)
                y_probs.extend(probs.cpu().numpy())
            
            preds = model.predict(mel, mfcc, waveform)
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

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    try:
        if cfg.task == "binary":
            auc_roc = roc_auc_score(y_true, y_probs)
        else:
            auc_roc = roc_auc_score(y_true, y_probs, multi_class="ovr")
    except:
        auc_roc = 0.0

    if show_focus_report and focus_indices:
        print_focus_class_report(y_true, y_pred, focus_indices, label_mapping)

    return epoch_loss, epoch_acc, precision, recall, f1, auc_roc


def main():
    print("=" * 70)
    print("🚀 HYBRID CNN V4 — TRAINING OPTIMIZADO")
    print("=" * 70)

    set_seed(CFG.seed)
    device = get_device()
    print(f"\n📱 Device: {device}")

    train_ds, test_ds, train_loader, test_loader = build_loaders(CFG)

    num_classes = len(train_ds.label_mapping)
    print(f"\n📊 Clases: {num_classes}")

    focus_indices = resolve_focus_class_indices(CFG.focus_classes, train_ds.label_mapping)
    focus_class_weights = build_focus_loss_weights(
        num_classes,
        focus_indices,
        CFG.focus_loss_weight,
        device,
    )

    if focus_indices:
        print(f"🎯 Focus indices: {focus_indices}")
        print(f"🎯 Focus loss weight: {CFG.focus_loss_weight}")

    print(f"\n🏗️ Creando modelo {CFG.mode} v{CFG.version}...")
    model = HybridCNNV5(
        num_classes=num_classes,
        dropout=CFG.dropout,
        mode=CFG.mode,
        task=CFG.task,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    label_smoothing = 0.0 if (focus_indices or CFG.task == "binary") else 0.05
    criterion = build_criterion(
        task=CFG.task,
        num_classes=num_classes,
        class_weights=focus_class_weights,   # None si no hay focus classes
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    print(f"\n📈 Scheduler: CosineAnnealingWarmRestarts")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-6,
    )

    scaler = GradScaler(device="cuda") if CFG.use_amp else None
    print("✅ AMP habilitado" if CFG.use_amp else "⚠️ AMP deshabilitado")

    start_epoch = 0
    best_acc = 0.0
    best_epoch = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc_roc": [],
        "lr": [],
        "epoch_time": [],
        "images_per_sec": [],
    }

    ckpt_epoch, ckpt_history, ckpt_best_acc, ckpt_best_epoch = load_latest_checkpoint(
        CFG,
        model,
        optimizer,
        scaler=scaler,
        scheduler=scheduler,
    )

    if ckpt_epoch is not None:
        start_epoch = ckpt_epoch
        history = ckpt_history
        best_acc = ckpt_best_acc
        best_epoch = ckpt_best_epoch
        print(f"🔄 Reanudado desde epoch {start_epoch}")
    else:
        # Si no hay checkpoint previo con scheduler, fuerza el LR manualmente
        if CFG.resume_lr is not None:
            set_optimizer_lr(optimizer, CFG.resume_lr)
            print(f"🔧 LR forzado manualmente a {CFG.resume_lr:.2e}")

    print("\n🧪 Probando dataloader...")
    try:
        t0 = time.time()
        for batch, labels, filenames in train_loader:
            dt = time.time() - t0
            print(f"  ✅ Primer batch en {dt:.2f}s")
            print(f"     - MEL shape: {batch['mel'].shape}")
            print(f"     - Labels shape: {labels.shape}")
            break
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print("  🔧 Reduce num_workers o batch_size")
        return

    print("\n🔥 INICIANDO ENTRENAMIENTO\n")

    for epoch in range(start_epoch + 1, CFG.epochs + 1):
        t0 = time.time()
        current_lr = get_optimizer_lr(optimizer)

        print(f"\n📌 Epoch {epoch}/{CFG.epochs}")
        print(f"📉 LR: {current_lr:.2e}")
        print("-" * 70)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            CFG,
            epoch,
            scaler=scaler,
            scheduler=scheduler,
        )

        val_loss, val_acc, precision, recall, f1, auc_roc = evaluate(
            model,
            test_loader,
            criterion,
            device,
            CFG,
            epoch,
            focus_indices=focus_indices,
            label_mapping=train_ds.label_mapping,
            show_focus_report=bool(focus_indices),
        )

        dt = time.time() - t0
        images_per_sec = len(train_loader.dataset) / dt

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["precision"].append(float(precision))
        history["recall"].append(float(recall))
        history["f1"].append(float(f1))
        history["auc_roc"].append(float(auc_roc) if 'auc_roc' in locals() else 0.0)
        history["lr"].append(float(current_lr))
        history["epoch_time"].append(float(dt))
        history["images_per_sec"].append(float(images_per_sec))

        print("\n📈 RESULTADOS")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")
        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print(f"F1 Score   : {f1:.4f}")
        print(f"⏱️ Epoch time: {dt:.1f}s ({images_per_sec:.0f} img/s)")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch

            os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
            # Guardamos un diccionario con el state_dict y el dropout
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "dropout": CFG.dropout,
                    "target_type": CFG.target_type,
                    "version": CFG.version,
                    "task": CFG.task
                },
                FINAL_MODEL_DIR / f"best_{CFG.target_type}_v{CFG.version}.pt"
            )
            print("💾 Best model updated")

        if epoch % CFG.save_every == 0:
            save_checkpoint(
                CFG,
                model,
                optimizer,
                scaler,
                scheduler,
                epoch,
                history,
                best_acc,
                best_epoch
            )

    print("\n🧪 EVALUACIÓN FINAL EN TEST")

    best_model = HybridCNNV5(
        num_classes=num_classes,
        dropout=CFG.dropout,
        mode=CFG.mode,
        task=CFG.task,
    ).to(device)

    ckpt = torch.load(
        FINAL_MODEL_DIR / f"best_{CFG.target_type}_v{CFG.version}.pt",
        map_location=device
    )
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        best_model.load_state_dict(ckpt["model_state"])
    else:
        best_model.load_state_dict(ckpt)

    test_loss, test_acc, precision, recall, f1, test_auc_roc = evaluate(
        best_model,
        test_loader,
        criterion,
        device,
        CFG,
        epoch="TEST",
        focus_indices=focus_indices,
        label_mapping=train_ds.label_mapping,
        show_focus_report=bool(focus_indices),
    )

    print("\n🏁 RESULTADOS FINALES TEST")
    print(f"Test Loss : {test_loss:.4f}")
    print(f"Test Acc  : {test_acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print(f"\n🥇 Best epoch: {best_epoch}")
    print(f"🥇 Best val_acc: {best_acc:.4f}")

    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
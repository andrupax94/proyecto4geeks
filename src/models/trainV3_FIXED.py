from __future__ import annotations
import os
import time
import random
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from src.models.hybrid_cnn import ImprovedMFCCCNN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import glob
from src.utils.config import PROCESSED_METADATA, LABEL_MAPPING, CHECKPOINT_DIR, FINAL_MODEL_DIR
from src.models.audio_dataset import ProcessedAudioDataset, crnn_collate_fn



class CFG:
    batch_size = 32  # ⬇️ Reducido de 64 para evitar memory issues
    lr = 3e-4
    weight_decay = 1e-2
    epochs = 12
    num_workers = 4  # ⬇️ Reducido de 8 para evitar deadlocks
    use_mfcc = True
    use_scalars = False
    seed = 42
    print_every = 50
    target_type = "human_label"
    mode = "mel_only"

    # 🔥 AJUSTES CRÍTICOS PARA AMD GPU
    checkpoint_dir = CHECKPOINT_DIR
    save_every = 2
    use_amp = False  # ⬇️ DESHABILITADO temporalmente para debugging
    prefetch_factor = 1  # ⬇️ Reducido de 2 para evitar overflow de memoria
    persistent_workers = False  # ⬇️ Deshabilitado para evitar memory leaks

    # 🎯 FOCUS CLASSES — clases problemáticas detectadas en la matriz de confusión.
    # Ejemplo: focus_classes = ["dog_bark", "car_horn", "siren"]
    # Dejar vacío para desactivar: focus_classes = []
    focus_classes: list = ["explosion","fire"]

    # Multiplicador de peso en la loss para las focus classes (>1 = más penalización)
    focus_loss_weight: float = 3.0

    # Multiplicador de oversample en el dataloader para las focus classes (>1 = más muestras)
    focus_oversample_factor: float = 2.0


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
# FOCUS CLASSES — HELPERS
# =========================================================

def resolve_focus_class_indices(focus_classes: list, label_mapping: dict) -> list[int]:
    """
    Convierte nombres de clases a índices numéricos usando el label_mapping del dataset.
    label_mapping puede ser {nombre: idx} o {idx: nombre}.
    """
    if not focus_classes:
        return []

    # Normalizar a {nombre: idx}
    if label_mapping and isinstance(next(iter(label_mapping)), int):
        name_to_idx = {v: k for k, v in label_mapping.items()}
    else:
        name_to_idx = label_mapping

    indices = []
    for cls in focus_classes:
        if cls in name_to_idx:
            indices.append(name_to_idx[cls])
        else:
            print(f"⚠️  Focus class '{cls}' no encontrada en label_mapping. Se ignora.")

    return indices


def build_focus_loss_weights(
    num_classes: int,
    focus_indices: list[int],
    focus_weight: float,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Devuelve un tensor de pesos por clase para CrossEntropyLoss.
    Las focus classes reciben `focus_weight`; el resto recibe 1.0.
    """
    if not focus_indices:
        return None

    weights = torch.ones(num_classes, dtype=torch.float32)
    for idx in focus_indices:
        weights[idx] = focus_weight

    return weights.to(device)


def build_focus_sampler(
    dataset,
    focus_indices: list[int],
    focus_factor: float,
) -> WeightedRandomSampler | None:
    """
    Crea un WeightedRandomSampler que oversamples las focus classes.
    ✅ MEJORADO: Extrae labels de múltiples formas:
    1. Si dataset tiene atributo .labels
    2. Si es un Subset, extrae de dataset.dataset.labels
    3. Si no, ITERA el dataset para extraer labels de cada muestra
    
    dataset[i] devuelve (batch, label, filename)
    """
    if not focus_indices:
        return None

    labels = None
    
    # Estrategia 1: Dataset tiene atributo .labels
    if hasattr(dataset, "labels"):
        labels = list(dataset.labels)
        print("   ✅ Labels extraídas de dataset.labels")
    
    # Estrategia 2: Es un Subset de random_split con .dataset.labels
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "labels"):
        labels = [dataset.dataset.labels[i] for i in dataset.indices]
        print("   ✅ Labels extraídas de dataset.dataset.labels (Subset)")
    
    # Estrategia 3: ITERA el dataset para extraer labels
    # Esto funciona porque dataset[i] devuelve (batch, label, filename)
    elif len(dataset) > 0:
        print("   ⏳ Extrayendo labels iterando dataset (esto puede tardar)...")
        labels = []
        for i in range(len(dataset)):
            try:
                _, label, _ = dataset[i]  # (batch, label, filename)
                # label puede ser tensor o int
                if hasattr(label, 'item'):
                    labels.append(label.item())
                else:
                    labels.append(int(label))
            except Exception as e:
                print(f"   ⚠️  Error extrayendo label en índice {i}: {e}")
                return None
        print(f"   ✅ {len(labels)} labels extraídas correctamente")
    else:
        print("⚠️  No se pudo extraer .labels del dataset. Oversample desactivado.")
        return None

    if labels is None or len(labels) == 0:
        print("⚠️  No hay labels para procesar. Oversample desactivado.")
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
    """
    Imprime métricas detalladas (precision, recall, f1, soporte) para cada focus class
    y muestra la sub-matriz de confusión entre ellas.
    """
    if not focus_indices:
        return

    # Normalizar a {idx: nombre}
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

    # Sub-matriz de confusión entre focus classes
    cm = confusion_matrix(y_true_np, y_pred_np, labels=focus_indices)
    class_names = [idx_to_name.get(idx, f"class_{idx}") for idx in focus_indices]

    print(f"\n📊 Sub-matriz de confusión (focus classes):")
    header_row = f"{'':>20}" + "".join(f"{n:>12}" for n in class_names)
    print(header_row)
    for i, name in enumerate(class_names):
        row_str = f"{name:>20}" + "".join(f"{cm[i, j]:>12}" for j in range(len(class_names)))
        print(row_str)

    print("=" * 70)


# =========================================================
# DEVICE
# =========================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    # =====================================================
    # TEST
    # =====================================================
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

    # =====================================================
    # LOADERS
    # =====================================================
    print("🔧 Creando dataloaders...")
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
        timeout=0,  # 🔥 Sin timeout para debugging
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

    total_loss = 0
    y_true, y_pred = [], []
    num_batches = len(loader)

    for batch_idx, (batch, labels, _) in enumerate(loader, 1):
        labels = labels.to(device)
        inputs = prepare_model_inputs(batch, device, cfg.mode)

        optimizer.zero_grad()

        # 🔥 Sin AMP para debugging
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
    y_true, y_pred = [], []
    num_batches = len(loader)

    with torch.inference_mode():
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

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # 🎯 Reporte de focus classes (si están configuradas)
    if show_focus_report and focus_indices and label_mapping is not None:
        print_focus_class_report(
            y_true=y_true,
            y_pred=y_pred,
            focus_indices=focus_indices,
            label_mapping=label_mapping,
            header=f"🎯 FOCUS CLASS REPORT — Epoch {epoch}",
        )

    return epoch_loss, epoch_acc, precision, recall, f1


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
    print(f"📦 Metadata: {PROCESSED_METADATA}")
    print(f"🎵 Modelo: ImprovedMFCCCNN")
    print(f"🔥 num_workers: {cfg.num_workers}")
    print(f"🔥 batch_size: {cfg.batch_size}")
    print(f"🔥 AMP (Mixed Precision): {cfg.use_amp}")
    print(f"🔥 prefetch_factor: {cfg.prefetch_factor}")
    print(f"🔥 persistent_workers: {cfg.persistent_workers}")
    if cfg.focus_classes:
        print(f"🎯 Focus classes: {cfg.focus_classes}")
        print(f"🎯 Focus loss weight: {cfg.focus_loss_weight}x | Oversample: {cfg.focus_oversample_factor}x")
    print("=" * 70)

    # =====================================================
    # DATASETS
    # =====================================================
    train_ds, test_ds, train_loader, test_loader = build_loaders(cfg)

    # =====================================================
    # SPLIT VALIDACIÓN DESDE TRAIN
    # =====================================================
    print("\n🔀 Creando split de validación...")
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

    num_classes = len(train_ds.label_mapping)

    # =====================================================
    # 🎯 FOCUS CLASSES SETUP
    # =====================================================
    focus_indices = resolve_focus_class_indices(cfg.focus_classes, train_ds.label_mapping)

    if focus_indices:
        focus_names = [cfg.focus_classes[i] for i in range(len(focus_indices))]
        print(f"\n🎯 Focus classes activas: {focus_names}")
        print(f"   - Índices: {focus_indices}")
        print(f"   - Loss weight: {cfg.focus_loss_weight}x")
        print(f"   - Oversample factor: {cfg.focus_oversample_factor}x")

        # Reemplazar train_loader con oversample si es posible
        focus_sampler = build_focus_sampler(
            train_subset,
            focus_indices,
            cfg.focus_oversample_factor,
        )
        if focus_sampler is not None:
            print("   ✅ WeightedRandomSampler activado para focus classes")
            train_loader = DataLoader(
                train_subset,
                batch_size=cfg.batch_size,
                sampler=focus_sampler,          # shuffle=False cuando se usa sampler
                num_workers=cfg.num_workers,
                collate_fn=crnn_collate_fn,
                drop_last=True,
                pin_memory=True,
                prefetch_factor=cfg.prefetch_factor,
                persistent_workers=cfg.persistent_workers,
            )
    else:
        focus_sampler = None
        print("\n🎯 Focus classes: ninguna configurada")

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
    print("\n🧠 Inicializando modelo...")
    model = ImprovedMFCCCNN(
        num_classes=num_classes,
        dropout=0.25,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"🧠 Modelo Info:")
    print(f"   - Total params: {total_params:,}")
    print(f"   - Trainable params: {trainable_params:,}")
    print("=" * 70)

    # =====================================================
    # LOSS — pesos extra para focus classes
    # =====================================================
    focus_class_weights = build_focus_loss_weights(
        num_classes=num_classes,
        focus_indices=focus_indices,
        focus_weight=cfg.focus_loss_weight,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=focus_class_weights,   # None si no hay focus classes (comportamiento original)
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
    # GRADIENT SCALER PARA AMP
    # =====================================================
    scaler = GradScaler() if cfg.use_amp else None

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

    if "test_loss" in history and "val_loss" not in history:
        history["val_loss"] = history["test_loss"]

    if "test_acc" in history and "val_acc" not in history:
        history["val_acc"] = history["test_acc"]

    for key in required_keys:
        if key not in history:
            history[key] = []

    print(f"🔄 Reanudado desde epoch {start_epoch}")

    # =====================================================
    # 🔥 TEST DE DATALOADER ANTES DE ENTRENAR
    # =====================================================
    print("\n🧪 Probando dataloader...")
    try:
        print("  - Obteniendo primer batch...")
        t0 = time.time()
        for batch, labels, filenames in train_loader:
            dt = time.time() - t0
            print(f"  ✅ Primer batch obtenido en {dt:.2f}s")
            print(f"     - MEL shape: {batch['mel'].shape}")
            print(f"     - Labels shape: {labels.shape}")
            break
    except Exception as e:
        print(f"  ❌ Error al cargar batch: {e}")
        print("  🔧 Intenta reducir num_workers o batch_size")
        return

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
            epoch,
            scaler=scaler
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
            epoch,
            focus_indices=focus_indices,
            label_mapping=train_ds.label_mapping,
            show_focus_report=bool(focus_indices),
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
        print(f"⏱️ Epoch time: {dt:.1f}s ({images_per_sec:.0f} img/s)")

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

    print(f"\n🥇 Mejor epoch: {best_epoch}")
    print(f"🥇 Mejor val_acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
# train_crnn.py

from __future__ import annotations

import os

# =========================================================
# ROCm / MIOpen SAFETY
# Debe ir antes de importar torch
# =========================================================
os.environ["MIOPEN_COMPILE_PARALLEL_LEVEL"] = "1"
os.environ["MIOPEN_DISABLE_CACHE"] = "1"  # ← Cambia a 1 (desactiva caché)
os.environ["MIOPEN_FIND_MODE"] = "FAST"
os.environ["MIOPEN_ENABLE_LOGGING"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["MIOPEN_FORCE_KERNEL_LDS"] = "0"  # ← Añade esto
os.environ["MIOPEN_DEBUG_LEVEL"] = "0"  # ← Añade esto
os.environ["MIOPEN_DEBUG_DISABLE_DROPOUT"] = "1"
import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.utils.config import (
    PROCESSED_METADATA,
    LABEL_MAPPING,
    CHECKPOINT_MODEL,
    ROOT_DIR,
)

from src.models.audio_dataset import (
    ProcessedAudioDataset,
    crnn_collate_fn,
)

from src.models.optimized_crnn_model_v2 import AudioCRNN

@dataclass
class TrainConfig:
    epochs: int = 30  # Aumenta a 30 (modelo más simple = más épocas)
    batch_size: int = 32  # Reduce a 32 para CPU
    lr: float = 1e-3  # Aumenta learning rate ligeramente
    weight_decay: float = 1e-5  # Reduce regularización
    num_workers: int = 2  # ⚠️ 0 para CPU
    seed: int = 42
    use_amp: bool = False  # ⚠️ Desactiva AMP
    use_mfcc: bool = True
    use_scalars: bool = True
    patience: int = 15  # Aumenta paciencia
    save_every: int = 2
    log_every_batches: int = 20


# =========================================================
# UTILS
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    # return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_history(history: Dict[str, List], history_json: Path, history_csv: Path) -> None:
    with open(history_json, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    pd.DataFrame(history).to_csv(history_csv, index=False)


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    history: Dict[str, List],
    best_f1: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "history": history,
            "best_f1": best_f1,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = int(checkpoint.get("epoch", 0))
    history = checkpoint.get("history", None)
    best_f1 = float(checkpoint.get("best_f1", 0.0))

    return epoch, history, best_f1


# =========================================================
# DATALOADERS
# =========================================================
def build_loaders(cfg: TrainConfig):
    train_ds = ProcessedAudioDataset(
        metadata_csv=PROCESSED_METADATA,
        label_mapping_path=LABEL_MAPPING,
        split="train",
        use_mfcc=cfg.use_mfcc,
        use_scalars=cfg.use_scalars,
    )

    test_ds = ProcessedAudioDataset(
        metadata_csv=PROCESSED_METADATA,
        label_mapping_path=LABEL_MAPPING,
        split="test",
        use_mfcc=cfg.use_mfcc,
        use_scalars=cfg.use_scalars,
        target_frames=train_ds.target_frames,
    )

    pin_memory = False
    persistent_workers = cfg.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=crnn_collate_fn,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=crnn_collate_fn,
        drop_last=False,
    )

    return train_ds, test_ds, train_loader, test_loader


# =========================================================
# TRAIN / EVAL
# =========================================================
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    log_every_batches: int = 50,
):
    model.train()

    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    start_epoch_time = time.time()

    amp_enabled = scaler is not None and scaler.is_enabled()

    total_batches = len(loader)

    for batch_idx, (batch, labels, _filenames) in enumerate(loader, start=1):
        mel = batch["mel"].to(device, non_blocking=True)
        mfcc = batch.get("mfcc")
        scalars = batch.get("scalars")

        if mfcc is not None:
            mfcc = mfcc.to(device, non_blocking=True)

        if scalars is not None:
            scalars = scalars.to(device, non_blocking=True)

        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(mel=mel, mfcc=mfcc, scalars=scalars)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        running_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()
        running_correct += correct
        running_total += labels.size(0)

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

        if batch_idx % log_every_batches == 0 or batch_idx == total_batches:
            elapsed = max(time.time() - start_epoch_time, 1e-8)
            seen = batch_idx * labels.size(0)
            speed = seen / elapsed

            window_batches = log_every_batches if batch_idx % log_every_batches == 0 else (batch_idx % log_every_batches)
            avg_loss = running_loss / max(1, window_batches)
            avg_acc = running_correct / max(1, running_total)

            print(
                f"  Batch {batch_idx}/{total_batches} | "
                f"loss={avg_loss:.4f} acc={avg_acc:.4f} | "
                f"{speed:.2f} img/s"
            )

            running_loss = 0.0
            running_correct = 0
            running_total = 0

    avg_loss = total_loss / max(1, total_batches)
    acc = accuracy_score(y_true, y_pred)

    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch, labels, _filenames in loader:
        mel = batch["mel"].to(device, non_blocking=True)
        mfcc = batch.get("mfcc")
        scalars = batch.get("scalars")

        if mfcc is not None:
            mfcc = mfcc.to(device, non_blocking=True)

        if scalars is not None:
            scalars = scalars.to(device, non_blocking=True)

        labels = labels.to(device, non_blocking=True)

        outputs = model(mel=mel, mfcc=mfcc, scalars=scalars)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return avg_loss, acc, precision, recall, f1


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=50,
        help="Imprime progreso cada N batches dentro de la época",
    )
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        use_amp=not args.no_amp,
        use_mfcc=True,
        use_scalars=True,
        patience=args.patience,
        save_every=args.save_every,
        log_every_batches=args.log_every_batches,
    )

    set_seed(cfg.seed)
    device = get_device()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("medium")

    print(f"🚀 Device: {device}")
    print(f"Metadata: {PROCESSED_METADATA}")
    print(f"Label mapping: {LABEL_MAPPING}")
    print(f"Checkpoint dir: {CHECKPOINT_MODEL}")

    train_ds, test_ds, train_loader, test_loader = build_loaders(cfg)

    scalar_dim = len(train_ds.available_scalar_columns)
    num_classes = len(train_ds.label_mapping)

    model = AudioCRNN(
        num_classes=num_classes,
        use_mfcc=cfg.use_mfcc,
        scalar_dim=scalar_dim,
    ).to(device)

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    
    scaler = torch.amp.GradScaler(
        enabled=(cfg.use_amp and device.type == "cuda")
    )

    # =====================================================
    # PATHS
    # =====================================================
    checkpoint_dir = CHECKPOINT_MODEL
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    final_dir = ROOT_DIR / "models" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    history_dir = ROOT_DIR / "models" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    latest_checkpoint = checkpoint_dir / "latest_checkpoint.pt"
    best_checkpoint = checkpoint_dir / "best_checkpoint.pt"

    best_model_path = final_dir / "crnn_best.pt"
    final_model_path = final_dir / "crnn_final.pt"

    history_json = history_dir / "history.json"
    history_csv = history_dir / "history.csv"

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "epoch_time": [],
        "images_per_sec": [],
    }

    # =====================================================
    # RESUME
    # =====================================================
    start_epoch = 1
    best_f1 = 0.0

    if args.resume and latest_checkpoint.exists():
        print(f"📦 Resumiendo desde: {latest_checkpoint}")

        last_epoch, loaded_history, best_f1 = load_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )

        start_epoch = last_epoch + 1

        if loaded_history is not None:
            history = loaded_history

    total_epochs = start_epoch + cfg.epochs - 1

    print(f"\n🚀 Entrenando {len(train_ds)} train / {len(test_ds)} test")
    print(
        f"Batch size: {cfg.batch_size} | Workers: {cfg.num_workers} | AMP: {scaler.is_enabled()}"
    )
    print(f"Print de progreso cada {cfg.log_every_batches} batches")
    print(f"Patience: {cfg.patience} | Save every: {cfg.save_every}")

    patience_left = cfg.patience

    # =====================================================
    # TRAIN LOOP
    # =====================================================
    for current_epoch in range(start_epoch, total_epochs + 1):
        if device.type == "cuda":
            torch.cuda.empty_cache()

        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            log_every_batches=cfg.log_every_batches,
        )

        test_loss, test_acc, precision, recall, f1 = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        scheduler.step(test_loss)

        epoch_time = time.time() - t0
        images_per_sec = len(train_loader.dataset) / max(epoch_time, 1e-8)

        history["epoch"].append(current_epoch)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["test_loss"].append(float(test_loss))
        history["test_acc"].append(float(test_acc))
        history["precision"].append(float(precision))
        history["recall"].append(float(recall))
        history["f1"].append(float(f1))
        history["epoch_time"].append(float(epoch_time))
        history["images_per_sec"].append(float(images_per_sec))

        print(
            f"Epoch {current_epoch} ({current_epoch}/{total_epochs}) | "
            f"time={epoch_time:.2f}s | img/s={images_per_sec:.2f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}"
        )

        save_history(history, history_json, history_csv)

        # =================================================
        # BEST MODEL
        # =================================================
        if f1 > best_f1:
            best_f1 = f1
            patience_left = cfg.patience

            torch.save(model.state_dict(), best_model_path)
            save_checkpoint(
                best_checkpoint,
                current_epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                history,
                best_f1,
            )

            print(f"💾 Nuevo mejor modelo guardado en: {best_model_path}")

        else:
            patience_left -= 1
            if patience_left <= 0:
                print(
                    f"⏹️ Early stopping en epoch {current_epoch} | best_f1={best_f1:.4f}"
                )
                save_checkpoint(
                    latest_checkpoint,
                    current_epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    history,
                    best_f1,
                )
                break

        # =================================================
        # CHECKPOINT PERIÓDICO
        # =================================================
        if current_epoch % cfg.save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{current_epoch}.pt"

            save_checkpoint(
                checkpoint_path,
                current_epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                history,
                best_f1,
            )

            save_checkpoint(
                latest_checkpoint,
                current_epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                history,
                best_f1,
            )

            print(f"📦 Checkpoint guardado: {checkpoint_path}")

    # =====================================================
    # GUARDADO FINAL
    # =====================================================
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Modelo final guardado en: {final_model_path}")
    print(f"📄 Historial JSON: {history_json}")
    print(f"📄 Historial CSV: {history_csv}")


if __name__ == "__main__":
    main()
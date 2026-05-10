# train_audio_cnn.py

from __future__ import annotations

import os
import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.utils.config import (
    ROOT_DIR,
    PROCESSED_METADATA,
    LABEL_MAPPING,
    CHECKPOINT_MODEL,
)
from src.models.audio_dataset import ProcessedAudioDataset, crnn_collate_fn
from src.models.audio_cnn_model import AudioCNNClassifier


# =========================================================
# ROCm SAFETY
# =========================================================
os.environ["MIOPEN_COMPILE_PARALLEL_LEVEL"] = "1"
os.environ["MIOPEN_DISABLE_CACHE"] = "0"
os.environ["MIOPEN_FIND_MODE"] = "FAST"
os.environ["MIOPEN_ENABLE_LOGGING"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# =========================================================
# CONFIG
# =========================================================
@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    use_amp: bool = False
    patience: int = 10
    save_every: int = 2
    log_every_batches: int = 50
    feature_mode: str = "both"  # mel | mfcc | both


# =========================================================
# UTILS
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_features(batch):
    return batch["mel"], batch["mfcc"]


# =========================================================
# DATA
# =========================================================
def build_loaders(cfg: TrainConfig):
    train_ds = ProcessedAudioDataset(
        metadata_csv=PROCESSED_METADATA,
        label_mapping_path=LABEL_MAPPING,
        split="train",
        use_mfcc=True,
        use_scalars=False,
    )

    test_ds = ProcessedAudioDataset(
        metadata_csv=PROCESSED_METADATA,
        label_mapping_path=LABEL_MAPPING,
        split="test",
        use_mfcc=True,
        use_scalars=False,
        target_frames=train_ds.target_frames,
    )

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=crnn_collate_fn,
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    return train_ds, test_ds, train_loader, test_loader


# =========================================================
# TRAIN
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device, cfg):
    model.train()

    y_true, y_pred = [], []
    total_loss = 0

    start = time.time()

    for i, (batch, labels, _) in enumerate(loader, 1):
        x = extract_features(batch, cfg.feature_mode).to(device)
        y = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        if i % cfg.log_every_batches == 0:
            acc = accuracy_score(y_true, y_pred)
            print(f"  Batch {i}/{len(loader)} | loss={loss.item():.4f} acc={acc:.4f}")

    return total_loss / len(loader), accuracy_score(y_true, y_pred)


# =========================================================
# EVAL
# =========================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg):
    model.eval()

    y_true, y_pred = [], []
    total_loss = 0

    for batch, labels, _ in loader:
        x = extract_features(batch, cfg.feature_mode).to(device)
        y = labels.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return total_loss / len(loader), acc, p, r, f1


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
    parser.add_argument("--feature-mode", choices=["mel", "mfcc", "both"], default="both")
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        use_amp=not args.no_amp,
        feature_mode=args.feature_mode,
    )

    set_seed(cfg.seed)
    device = get_device()

    print("Device:", device)
    print("Feature mode:", cfg.feature_mode)

    train_ds, test_ds, train_loader, test_loader = build_loaders(cfg)

    sample = extract_features(train_ds[0], cfg.feature_mode)
    input_shape = tuple(sample.shape)

    model = AudioCNNClassifier(
        num_classes=len(train_ds.label_mapping),
        input_shape=input_shape,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_f1 = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg
        )

        test_loss, test_acc, p, r, f1 = evaluate(
            model, test_loader, criterion, device, cfg
        )

        print(
            f"\nEpoch {epoch} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"f1={f1:.4f} precision={p:.4f} recall={r:.4f} | "
            f"time={time.time() - t0:.1f}s"
        )

        if f1 > best_f1:
            best_f1 = f1
            Path(CHECKPOINT_MODEL).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(CHECKPOINT_MODEL) / "best_model.pt")
            print("💾 Nuevo mejor modelo guardado")


if __name__ == "__main__":
    main()
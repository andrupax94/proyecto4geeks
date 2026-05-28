# app/routers/training.py

import glob
import os
import re

import torch
from fastapi import APIRouter, HTTPException

from src.utils.config import CHECKPOINT_DIR

router = APIRouter()


def _extract_epoch(path: str) -> int:
    m = re.search(r"checkpoint_epoch_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


@router.get("/training/history")
async def get_training_history(target_type: str = "alertable", version: int = 4):
    """
    Devuelve el history del checkpoint más reciente para graficar en el frontend.
    Parámetros opcionales:
      - target_type: carpeta dentro de CHECKPOINT_DIR  (default: "alertable")
      - version:     versión del modelo                (default: 4)
    """
    checkpoint_dir = CHECKPOINT_DIR / f"{target_type}_V{version}"

    if not checkpoint_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No existe el directorio de checkpoints: {checkpoint_dir}"
        )

    checkpoints = glob.glob(str(checkpoint_dir / "checkpoint_epoch_*.pt"))
    if not checkpoints:
        raise HTTPException(
            status_code=404,
            detail="No hay checkpoints disponibles en ese directorio."
        )

    latest = max(checkpoints, key=_extract_epoch)
    latest_epoch = _extract_epoch(latest)

    # Cargamos solo los metadatos, sin pesos del modelo (más rápido)
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)

    history: dict = ckpt.get("history", {})
    best_acc: float = ckpt.get("best_acc", None)
    best_epoch: int = ckpt.get("best_epoch", None)

    return {
        "checkpoint_epoch": latest_epoch,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "history": history,
    }
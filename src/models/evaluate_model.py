from __future__ import annotations
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.models.hybrid_cnn import ImprovedMFCCCNN
from src.models.audio_dataset import ProcessedAudioDataset, crnn_collate_fn
from src.utils.config import (
    PROCESSED_METADATA,
    LABEL_MAPPING,
    FINAL_MODEL_DIR
)

# ======================================================
# CONFIG
# ======================================================
class CFG:
    batch_size = 32
    num_workers = 4
    target_type = "human_label"
    mode = "mel_only"
    model_path = os.path.join(FINAL_MODEL_DIR, "best_model.pt")


# ======================================================
# DEVICE
# ======================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# INPUTS
# ======================================================
def prepare_model_inputs(batch, device, mode):

    mel = batch.get("mel")
    mfcc = batch.get("mfcc")
    waveform = batch.get("waveform")

    if mel is not None:
        mel = mel.to(device)
    if mfcc is not None:
        mfcc = mfcc.to(device)
    if waveform is not None:
        waveform = waveform.to(device)

    if mode == "mel_only":
        return (mel,)

    if mode == "mel_mfcc":
        return (mel, mfcc)

    if mode == "all_three":
        return (mel, mfcc, waveform)

    raise ValueError("Modo inválido")


# ======================================================
# DATASET TEST
# ======================================================
def build_test_loader(cfg):

    label_mapping_path = LABEL_MAPPING[cfg.target_type]

    # ===============================
    # 1️⃣ CREAR TRAIN SOLO PARA SHAPE
    # ===============================
    train_ds = ProcessedAudioDataset(
        metadata_csv=PROCESSED_METADATA,
        label_mapping_path=label_mapping_path,
        split="train",
        target_column=cfg.target_type,
        use_mfcc=True,
        use_scalars=False,
    )

    # ===============================
    # 2️⃣ TEST DATASET
    # ===============================
    test_ds = ProcessedAudioDataset(
        metadata_csv=PROCESSED_METADATA,
        label_mapping_path=label_mapping_path,
        split="test",
        target_column=cfg.target_type,
        use_mfcc=True,
        use_scalars=False,

        # 🔥 CLAVE ABSOLUTA
        target_frames=train_ds.target_frames,
    )

    loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=crnn_collate_fn,
    )

    return test_ds, loader


# ======================================================
# MATRIZ CONFUSIÓN
# ======================================================
@torch.no_grad()
def confusion_matrix_test(model, loader, device, class_names):

    model.eval()

    y_true = []
    y_pred = []

    for batch, labels, _ in loader:

        labels = labels.to(device)
        inputs = prepare_model_inputs(batch, device, CFG.mode)

        outputs = model(*inputs)

        preds = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(cmap="Blues", xticks_rotation=45)

    plt.title("Confusion Matrix - TEST SET")
    plt.tight_layout()

    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

    save_path = os.path.join(
        FINAL_MODEL_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\n✅ Matriz guardada en: {save_path}")


# ======================================================
# MAIN
# ======================================================
def main():

    cfg = CFG()

    device = get_device()

    print(f"🚀 Device: {device}")

    # Dataset
    test_ds, test_loader = build_test_loader(cfg)

    num_classes = len(test_ds.label_mapping)

    # Modelo
    model = ImprovedMFCCCNN(
        num_classes=num_classes,
        dropout=0.25,
    ).to(device)

    # Cargar pesos entrenados
    print("📦 Cargando modelo entrenado...")
    checkpoint = torch.load(cfg.model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])

    class_names = list(test_ds.label_mapping.keys())

    # Generar matriz
    confusion_matrix_test(
        model,
        test_loader,
        device,
        class_names
    )


if __name__ == "__main__":
    main()
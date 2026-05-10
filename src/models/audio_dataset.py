"""audio_dataset.py

Dataset para audios ya preprocesados.

Lee el CSV de metadatos generado por el pipeline de preprocesado y carga:
- mel spectrogram (.npy)
- mfcc (.npy) [opcional]
- features escalares del CSV [opcional]

Diseñado para trabajar bien con:
- 60k muestras
- RAM limitada
- DataLoader con 4-6 workers
- entrenamiento en GPU/ROCm con batches grandes
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PathLike = Union[str, Path]

DEFAULT_SCALAR_COLUMNS = [
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "zcr_mean",
    "zcr_std",
    "rms_mean",
    "rms_std",
    "chroma_mean",
    "chroma_std",
]


class ProcessedAudioDataset(Dataset):
    """
    Dataset de features ya procesadas.

    Devuelve un diccionario con:
      - mel: Tensor [1, mel_bins, time]
      - mfcc: Tensor [1, mfcc_bins, time] (si use_mfcc=True)
      - scalars: Tensor [N] (si use_scalars=True y columnas disponibles)
      - label: int
      - filename: str

    La secuencia temporal se fija a `target_frames` para poder usar batches
    estables sin que el DataLoader tenga que pelearse con tamaños variables.
    """

    def __init__(
        self,
        metadata_csv: PathLike,
        label_mapping_path: Optional[PathLike] = None,
        split: Optional[str] = None,
        use_mfcc: bool = True,
        use_scalars: bool = True,
        target_frames: Optional[int] = None,
        scalar_columns: Optional[Sequence[str]] = None,
        mel_column_candidates: Sequence[str] = ("mel_spec_path", "mel_path"),
        mfcc_column_candidates: Sequence[str] = ("mfcc_path",),
        label_column: str = "label_encoded",
        label_name_column: str = "label",
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.df = pd.read_csv(self.metadata_csv)

        if split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if self.df.empty:
            raise ValueError(f"No hay muestras disponibles en {self.metadata_csv} para split={split!r}.")

        self.use_mfcc = use_mfcc
        self.use_scalars = use_scalars
        self.scalar_columns = list(scalar_columns or DEFAULT_SCALAR_COLUMNS)

        self.mel_column = self._resolve_existing_column(self.df.columns, mel_column_candidates)
        self.mfcc_column = self._resolve_existing_column(self.df.columns, mfcc_column_candidates, required=False)

        self.label_column = label_column if label_column in self.df.columns else None
        self.label_name_column = label_name_column if label_name_column in self.df.columns else None

        self.label_mapping: Optional[Dict[str, int]] = None
        self.num_classes: Optional[int] = None

        if label_mapping_path is not None and Path(label_mapping_path).exists():
            with open(label_mapping_path, "rb") as f:
                self.label_mapping = pickle.load(f)
            if isinstance(self.label_mapping, dict) and self.label_mapping:
                self.num_classes = int(max(int(v) for v in self.label_mapping.values()) + 1)

        self.available_scalar_columns = [c for c in self.scalar_columns if c in self.df.columns]

        self.target_frames = target_frames or self._infer_target_frames()
        self.mel_bins, self.mfcc_bins = self._infer_feature_bins()

    @staticmethod
    def _resolve_existing_column(
        columns: Sequence[str],
        candidates: Sequence[str],
        required: bool = True,
    ) -> Optional[str]:
        for c in candidates:
            if c in columns:
                return c
        if required:
            raise KeyError(f"No encontré ninguna de estas columnas en el CSV: {list(candidates)}")
        return None

    def _infer_target_frames(self) -> int:
        for _, row in self.df.iterrows():
            mel_path = row.get(self.mel_column)
            if isinstance(mel_path, str) and Path(mel_path).exists():
                arr = np.load(mel_path, mmap_mode="r")
                if arr.ndim == 2:
                    return int(arr.shape[-1])
                if arr.ndim == 3:
                    return int(arr.shape[-1])
        raise FileNotFoundError("No pude inferir target_frames porque no encontré ningún mel válido.")

    def _infer_feature_bins(self) -> Tuple[int, Optional[int]]:
        mel_bins = None
        mfcc_bins = None

        for _, row in self.df.iterrows():
            mel_path = row.get(self.mel_column)
            if mel_bins is None and isinstance(mel_path, str) and Path(mel_path).exists():
                arr = np.load(mel_path, mmap_mode="r")
                if arr.ndim == 2:
                    mel_bins = int(arr.shape[0])
                elif arr.ndim == 3:
                    mel_bins = int(arr.shape[-2])

            if self.use_mfcc and self.mfcc_column is not None:
                mfcc_path = row.get(self.mfcc_column)
                if mfcc_bins is None and isinstance(mfcc_path, str) and Path(mfcc_path).exists():
                    arr = np.load(mfcc_path, mmap_mode="r")
                    if arr.ndim == 2:
                        mfcc_bins = int(arr.shape[0])
                    elif arr.ndim == 3:
                        mfcc_bins = int(arr.shape[-2])

            if mel_bins is not None and (not self.use_mfcc or mfcc_bins is not None):
                break

        if mel_bins is None:
            raise FileNotFoundError("No pude inferir mel_bins porque no encontré ningún mel válido.")

        return mel_bins, mfcc_bins

    @staticmethod
    def _load_npy(path: str | Path) -> np.ndarray:
        arr = np.load(path, mmap_mode="r")
        if isinstance(arr, np.memmap):
            arr = np.asarray(arr)
        return np.asarray(arr, dtype=np.float32)

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]
        if arr.ndim == 3 and arr.shape[1] == 1:
            return arr[:, 0, :]
        raise ValueError(f"Forma no soportada: {arr.shape}")

    def _fix_time_axis(self, arr: np.ndarray, target_frames: int) -> np.ndarray:
        arr = self._ensure_2d(arr)
        time_dim = arr.shape[-1]

        if time_dim > target_frames:
            arr = arr[..., :target_frames]
        elif time_dim < target_frames:
            pad_width = target_frames - time_dim
            arr = np.pad(arr, ((0, 0), (0, pad_width)), mode="constant")

        return arr.astype(np.float32, copy=False)

    def _load_label(self, row: pd.Series) -> int:
        if self.label_column is not None and pd.notna(row.get(self.label_column)):
            return int(row[self.label_column])

        if self.label_name_column is not None and self.label_mapping is not None:
            label_name = str(row[self.label_name_column])
            return int(self.label_mapping[label_name])

        raise KeyError(
            "No pude resolver la etiqueta. Necesito 'label_encoded' o 'label' + label_mapping.pkl."
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int | str]:
        row = self.df.iloc[idx]
        sample: Dict[str, torch.Tensor | int | str] = {}

        mel_path = row[self.mel_column]
        mel = self._load_npy(mel_path)
        mel = self._fix_time_axis(mel, self.target_frames)
        sample["mel"] = torch.from_numpy(mel).unsqueeze(0)  # [1, mel_bins, time]

        if self.use_mfcc and self.mfcc_column is not None and pd.notna(row.get(self.mfcc_column)):
            mfcc = self._load_npy(row[self.mfcc_column])
            mfcc = self._fix_time_axis(mfcc, self.target_frames)
            mfcc = np.array(mfcc, copy=True).astype(np.float32, copy=False)
            sample["mfcc"] = torch.from_numpy(mfcc).unsqueeze(0)

        if self.use_scalars and self.available_scalar_columns:
            scalars = [float(row[c]) for c in self.available_scalar_columns]
            sample["scalars"] = torch.tensor(scalars, dtype=torch.float32)

        sample["label"] = self._load_label(row)
        sample["filename"] = str(row.get("filename", idx))
        return sample


def crnn_collate_fn(batch: List[Dict[str, torch.Tensor | int | str]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[str]]:
    """
    Collate simple para batches homogéneos.

    Como el dataset ya fija `target_frames`, aquí solo apilamos tensores.
    """
    mel = torch.stack([item["mel"] for item in batch], dim=0)

    output: Dict[str, torch.Tensor] = {"mel": mel}
    if "mfcc" in batch[0]:
        output["mfcc"] = torch.stack([item["mfcc"] for item in batch], dim=0)
    if "scalars" in batch[0]:
        output["scalars"] = torch.stack([item["scalars"] for item in batch], dim=0)

    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    filenames = [str(item["filename"]) for item in batch]
    return output, labels, filenames

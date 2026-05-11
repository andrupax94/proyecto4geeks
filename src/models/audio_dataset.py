from __future__ import annotations

import pickle
import re
from pathlib import Path, PureWindowsPath
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

        # 🔥 NORMALIZACIÓN AUTOMÁTICA DE RUTAS (FIX PRINCIPAL)
        self._normalize_dataframe_paths()

        self.target_frames = target_frames or self._infer_target_frames()
        self.mel_bins, self.mfcc_bins = self._infer_feature_bins()

    # =========================================================
    # 🔧 PATH NORMALIZATION (FIX PRINCIPAL)
    # =========================================================

    @staticmethod
    def _normalize_path_value(path_value: object, project_root: Path) -> str:
        if path_value is None or (isinstance(path_value, float) and pd.isna(path_value)):
            return path_value

        path_str = str(path_value).strip()
        if not path_str:
            return path_str

        # Linux path válido
        if path_str.startswith("/"):
            return path_str

        # Windows absolute path (E:\...)
        if re.match(r"^[A-Za-z]:\\", path_str):
            win_path = PureWindowsPath(path_str)
            parts = win_path.parts

            # intenta reconstruir desde proyecto
            project_name = project_root.name

            for i, part in enumerate(parts):
                if part.lower() == project_name.lower():
                    rel_path = Path(*parts[i:])
                    return str((project_root / rel_path.relative_to(project_name)).resolve()).replace("\\", "/")

            # fallback: elimina drive
            return str(Path(*parts[1:])).replace("\\", "/")

        return path_str.replace("\\", "/")

    def _normalize_dataframe_paths(self) -> None:
        project_root = Path.cwd().resolve()

        for col in [self.mel_column, self.mfcc_column, "audio_normalized_path"]:
            if col is not None and col in self.df.columns:
                self.df[col] = self.df[col].apply(
                    lambda x: self._normalize_path_value(x, project_root)
                )

    # =========================================================
    # EXISTING LOGIC (UNCHANGED)
    # =========================================================

    @staticmethod
    def _resolve_existing_column(columns: Sequence[str], candidates: Sequence[str], required: bool = True) -> Optional[str]:
        for c in candidates:
            if c in columns:
                return c
        if required:
            raise KeyError(f"No encontré ninguna de estas columnas: {list(candidates)}")
        return None

    def _infer_target_frames(self) -> int:
        for _, row in self.df.iterrows():
            mel_path = row.get(self.mel_column)
            if isinstance(mel_path, str) and Path(mel_path).exists():
                arr = np.load(mel_path, mmap_mode="r+")
                return int(arr.shape[-1]) if arr.ndim in (2, 3) else None

        raise FileNotFoundError("No pude inferir target_frames.")

    def _infer_feature_bins(self) -> Tuple[int, Optional[int]]:
        mel_bins = None
        mfcc_bins = None

        for _, row in self.df.iterrows():
            mel_path = row.get(self.mel_column)
            if mel_bins is None and isinstance(mel_path, str) and Path(mel_path).exists():
                arr = np.load(mel_path, mmap_mode="r+")
                mel_bins = int(arr.shape[-2]) if arr.ndim == 3 else int(arr.shape[0])

            if self.use_mfcc and self.mfcc_column is not None:
                mfcc_path = row.get(self.mfcc_column)
                if mfcc_bins is None and isinstance(mfcc_path, str) and Path(mfcc_path).exists():
                    arr = np.load(mfcc_path, mmap_mode="r+")
                    mfcc_bins = int(arr.shape[-2]) if arr.ndim == 3 else int(arr.shape[0])

            if mel_bins is not None and (not self.use_mfcc or mfcc_bins is not None):
                break

        if mel_bins is None:
            raise FileNotFoundError("No pude inferir mel_bins.")

        return mel_bins, mfcc_bins

    @staticmethod
    def _load_npy(path: str | Path) -> np.ndarray:
        arr = np.load(path, mmap_mode="r+")
        return np.array(arr, dtype=np.float32, copy=True)

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
            pad = target_frames - time_dim
            arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant")

        return arr.astype(np.float32, copy=False)

    def _load_label(self, row: pd.Series) -> int:
        if self.label_column is not None and pd.notna(row.get(self.label_column)):
            return int(row[self.label_column])

        if self.label_name_column is not None and self.label_mapping is not None:
            return int(self.label_mapping[str(row[self.label_name_column])])

        raise KeyError("No se pudo resolver label.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        mel = np.array(
            self._fix_time_axis(
                self._load_npy(row[self.mel_column]),
                self.target_frames,
            ),
            dtype=np.float32,
            copy=True,
        )
        

        sample = {
            "mel": torch.from_numpy(mel).unsqueeze(0),
            "label": self._load_label(row),
            "filename": str(row.get("filename", idx)),
        }

        if self.use_mfcc and self.mfcc_column is not None and pd.notna(row.get(self.mfcc_column)):
            mfcc = self._fix_time_axis(
                self._load_npy(row[self.mfcc_column]),
                self.target_frames,
            )
            mfcc = np.array(mfcc, dtype=np.float32, copy=True)
            sample["mfcc"] = torch.from_numpy(mfcc).unsqueeze(0)

        if self.use_scalars and self.available_scalar_columns:
            sample["scalars"] = torch.tensor(
                [float(row[c]) for c in self.available_scalar_columns],
                dtype=torch.float32,
            )

        return sample


def crnn_collate_fn(batch):
    mel = torch.stack([b["mel"] for b in batch])

    out = {"mel": mel}

    if "mfcc" in batch[0]:
        out["mfcc"] = torch.stack([b["mfcc"] for b in batch])

    if "scalars" in batch[0]:
        out["scalars"] = torch.stack([b["scalars"] for b in batch])

    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    filenames = [b["filename"] for b in batch]

    return out, labels, filenames
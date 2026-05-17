from __future__ import annotations

import pickle
import re
from pathlib import Path, PureWindowsPath
from typing import Dict, Optional, Sequence, Tuple, Union, Any

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
        target_column: str = "human_label",
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

        self.target_column = target_column
        if self.target_column not in self.df.columns:
            raise KeyError(f"No existe la columna target '{self.target_column}' en el CSV.")

        self.label_mapping: Optional[Dict[Any, int]] = None
        self.num_classes: Optional[int] = None

        if label_mapping_path is not None and Path(label_mapping_path).exists():
            with open(label_mapping_path, "rb") as f:
                payload = pickle.load(f)

            self.label_mapping = self._extract_label_mapping(payload)
            self.num_classes = len(self.label_mapping)

        self.available_scalar_columns = [c for c in self.scalar_columns if c in self.df.columns]

        # 🔥 OPTIMIZACIÓN: Normalizar paths UNA SOLA VEZ
        self._normalize_dataframe_paths()

        # 🔥 OPTIMIZACIÓN: Inferir dimensiones sin loops iterrows
        self.target_frames = target_frames or self._infer_target_frames_fast()
        self.mel_bins, self.mfcc_bins = self._infer_feature_bins_fast()

    @staticmethod
    def _extract_label_mapping(payload: Any) -> Dict[Any, int]:
        """
        Soporta:
        - dict plano: {"dog": 0, "gunshot": 1}
        - dict envuelto: {"label2idx": {...}, "idx2label": {...}, ...}
        """
        if isinstance(payload, dict) and "label2idx" in payload:
            mapping = payload["label2idx"]
        else:
            mapping = payload

        if not isinstance(mapping, dict) or not mapping:
            raise TypeError("El label mapping cargado no tiene el formato esperado.")

        return mapping

    @staticmethod
    def _normalize_alertable_value(value: Any) -> Optional[bool]:
        if pd.isna(value):
            return None

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, np.integer)):
            return bool(value)

        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y", "t"}:
            return True
        if text in {"false", "0", "no", "n", "f"}:
            return False

        return None

    @staticmethod
    def _resolve_existing_column(columns: Sequence[str], candidates: Sequence[str], required: bool = True) -> Optional[str]:
        for c in candidates:
            if c in columns:
                return c
        if required:
            raise KeyError(f"No encontré ninguna de estas columnas: {list(candidates)}")
        return None

    @staticmethod
    def _normalize_path_value(path_value: object, project_root: Path) -> str:
        if path_value is None or (isinstance(path_value, float) and pd.isna(path_value)):
            return path_value

        path_str = str(path_value).strip()
        if not path_str:
            return path_str

        if path_str.startswith("/"):
            return path_str

        if re.match(r"^[A-Za-z]:\\", path_str):
            win_path = PureWindowsPath(path_str)
            parts = win_path.parts
            project_name = project_root.name

            for i, part in enumerate(parts):
                if part.lower() == project_name.lower():
                    rel_path = Path(*parts[i:])
                    return str((project_root / rel_path.relative_to(project_name)).resolve()).replace("\\", "/")

            return str(Path(*parts[1:])).replace("\\", "/")

        return path_str.replace("\\", "/")

    def _normalize_dataframe_paths(self) -> None:
        project_root = Path.cwd().resolve()

        for col in [self.mel_column, self.mfcc_column, "audio_normalized_path"]:
            if col is not None and col in self.df.columns:
                # 🔥 OPTIMIZACIÓN: Usar apply vectorizado, más rápido que iterrows
                self.df[col] = self.df[col].apply(
                    lambda x: self._normalize_path_value(x, project_root)
                )

    @staticmethod
    def _load_npy_header(path: str | Path) -> Tuple[Tuple[int, ...], np.dtype]:
        """
        🔥 OPTIMIZACIÓN: Lee SOLO el header del archivo .npy sin cargar datos.
        Retorna (shape, dtype)
        """
        try:
            with open(path, 'rb') as f:
                magic = f.read(6)
                if magic[:2] != b'\x93N':
                    raise ValueError("No es un archivo NPY válido")
                
                version = tuple(f.read(2))
                header_len = int.from_bytes(f.read(4 if version[1] >= 3 else 2), 'little')
                header = f.read(header_len).decode('latin1')
                
                # Extraer shape y dtype del header
                import ast
                header_dict = ast.literal_eval(header.strip())
                return tuple(header_dict['shape']), np.dtype(header_dict['descr'])
        except Exception:
            # Fallback si algo falla
            arr = np.load(path, mmap_mode='r')
            return arr.shape, arr.dtype

    def _infer_target_frames_fast(self) -> int:
        """
        🔥 OPTIMIZACIÓN: Lee SOLO el header sin cargar el array completo.
        """
        for mel_path in self.df[self.mel_column]:
            if isinstance(mel_path, str) and Path(mel_path).exists():
                try:
                    shape, _ = self._load_npy_header(mel_path)
                    return int(shape[-1]) if len(shape) in (2, 3) else None
                except Exception:
                    continue
        raise FileNotFoundError("No pude inferir target_frames.")

    def _infer_feature_bins_fast(self) -> Tuple[int, Optional[int]]:
        """
        🔥 OPTIMIZACIÓN: Lee headers sin cargar datos completos.
        """
        mel_bins = None
        mfcc_bins = None

        for idx, row in self.df.iterrows():
            # Mel bins
            if mel_bins is None:
                mel_path = row.get(self.mel_column)
                if isinstance(mel_path, str) and Path(mel_path).exists():
                    try:
                        shape, _ = self._load_npy_header(mel_path)
                        mel_bins = int(shape[-2]) if len(shape) == 3 else int(shape[0])
                    except Exception:
                        continue

            # MFCC bins
            if self.use_mfcc and self.mfcc_column is not None and mfcc_bins is None:
                mfcc_path = row.get(self.mfcc_column)
                if isinstance(mfcc_path, str) and Path(mfcc_path).exists():
                    try:
                        shape, _ = self._load_npy_header(mfcc_path)
                        mfcc_bins = int(shape[-2]) if len(shape) == 3 else int(shape[0])
                    except Exception:
                        continue

            # Early exit si encontramos lo que necesitamos
            if mel_bins is not None and (not self.use_mfcc or mfcc_bins is not None):
                break

        if mel_bins is None:
            raise FileNotFoundError("No pude inferir mel_bins.")

        return mel_bins, mfcc_bins

    @staticmethod
    def _load_npy(path: str | Path) -> np.ndarray:
        """
        🔥 OPTIMIZACIÓN: Carga eficiente sin copies innecesarias.
        """
        arr = np.load(path, allow_pickle=False)
        return arr.astype(np.float32)  # Convierte en un paso, no copy=True

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

        return arr.astype(np.float32)

    def _load_label(self, row: pd.Series) -> int:
        raw = row.get(self.target_column)

        if pd.isna(raw):
            raise KeyError(f"No hay valor en la columna target '{self.target_column}'.")

        if self.label_mapping is not None:
            key = raw

            if self.target_column == "alertable":
                norm = self._normalize_alertable_value(raw)
                if norm is not None:
                    key = norm

            if key in self.label_mapping:
                return int(self.label_mapping[key])

            key_str = str(key).strip()
            if key_str in self.label_mapping:
                return int(self.label_mapping[key_str])

            if self.target_column == "alertable":
                # fallback extra por si el mapping guardó strings
                alt = str(bool(key)).capitalize()
                if alt in self.label_mapping:
                    return int(self.label_mapping[alt])

            raise KeyError(f"No se pudo mapear la etiqueta '{raw}' con target='{self.target_column}'.")

        # fallback si no hay mapping y ya viene codificado
        return int(raw)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # 🔥 OPTIMIZACIÓN: Carga MEL de forma eficiente
        mel = self._fix_time_axis(
            self._load_npy(row[self.mel_column]),
            self.target_frames,
        )

        sample = {
            "mel": torch.from_numpy(mel).unsqueeze(0),
            "label": self._load_label(row),
            "filename": str(row.get("filename", idx)),
        }

        # 🔥 OPTIMIZACIÓN: MFCC solo si existe
        if (
            self.use_mfcc
            and self.mfcc_column is not None
            and pd.notna(row.get(self.mfcc_column))
        ):
            try:
                mfcc = self._fix_time_axis(
                    self._load_npy(row[self.mfcc_column]),
                    self.target_frames,
                )
                sample["mfcc"] = torch.from_numpy(mfcc).unsqueeze(0)
            except Exception:
                pass  # Saltarse si el archivo no existe

        # 🔥 OPTIMIZACIÓN: Escalares solo si se usan
        if self.use_scalars and self.available_scalar_columns:
            sample["scalars"] = torch.tensor(
                [float(row[c]) for c in self.available_scalar_columns],
                dtype=torch.float32,
            )

        return sample


def crnn_collate_fn(batch):
    """
    🔥 OPTIMIZACIÓN: Collate function eficiente sin operaciones innecesarias.
    """
    mel = torch.stack([b["mel"] for b in batch])

    out = {"mel": mel}

    if "mfcc" in batch[0]:
        out["mfcc"] = torch.stack([b["mfcc"] for b in batch])

    if "scalars" in batch[0]:
        out["scalars"] = torch.stack([b["scalars"] for b in batch])

    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    filenames = [b["filename"] for b in batch]

    return out, labels, filenames
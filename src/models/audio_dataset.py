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
    """
    Modos de mapping (mapping_mode):
      - "total"        : todas las clases human_label (sin filtro de alertable/emergency)
      - "human"        : solo filas con alertable == True
      - "alertable"    : target es la columna alertable (clasificación binaria)
      - "emergency"    : solo filas con alertable == True y emergency == True
      - "no_alertable" : solo filas con alertable == False

    Si se pasa label_mapping_path, el modo se autoinfiere del campo 'target_col'
    guardado en el pickle. Se puede sobreescribir con mapping_mode.
    """

    # Modos que filtran filas y usan human_label como target
    _HUMAN_LABEL_MODES = {"total", "human", "emergency", "no_alertable"}

    def __init__(
        self,
        metadata_csv: PathLike,
        label_mapping_path: Optional[PathLike] = None,
        split: Optional[str] = None,
        use_mfcc: bool = True,
        use_scalars: bool = True,
        use_waveform: bool = True,  # ⬇️ NUEVO
        target_frames: Optional[int] = None,
        scalar_columns: Optional[Sequence[str]] = None,
        mel_column_candidates: Sequence[str] = ("mel_spec_path", "mel_path"),
        mfcc_column_candidates: Sequence[str] = ("mfcc_path",),
        waveform_column_candidates: Sequence[str] = ("audio_normalized_path", "waveform_path"),  # ⬇️ NUEVO
        target_column: str = "human_label",
        mapping_mode: Optional[str] = None,  # "total" | "human" | "alertable" | "emergency"
    ) -> None:

        self.metadata_csv = Path(metadata_csv)
        self.df = pd.read_csv(self.metadata_csv)

        # ── Filtro por split ──────────────────────────────────────────────────
        if split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        # ── Carga del label mapping y detección del modo ──────────────────────
        self.label_mapping: Optional[Dict[Any, int]] = None
        self.num_classes: Optional[int] = None
        self._raw_payload: Optional[Dict] = None

        if label_mapping_path is not None and Path(label_mapping_path).exists():
            with open(label_mapping_path, "rb") as f:
                self._raw_payload = pickle.load(f)

            self.label_mapping = self._extract_label_mapping(self._raw_payload)
            self.num_classes = len(self.label_mapping)

            # Autoinfiere el modo desde el campo target_col del pickle
            if mapping_mode is None and isinstance(self._raw_payload, dict):
                target_col_in_pickle = self._raw_payload.get("target_col", "")
                mapping_mode = self._infer_mode_from_pkl_path(
                    str(label_mapping_path), target_col_in_pickle
                )

        self.mapping_mode = mapping_mode or "total"

        # ── target_column se ajusta automáticamente según el modo ─────────────
        if self.mapping_mode == "alertable":
            self.target_column = "alertable"
        else:
            self.target_column = target_column  # "human_label" por defecto

        if self.target_column not in self.df.columns:
            raise KeyError(f"No existe la columna target '{self.target_column}' en el CSV.")

        # ── Filtro de filas según el modo ─────────────────────────────────────
        self.df = self._filter_by_mode(self.df, self.mapping_mode)

        if self.df.empty:
            raise ValueError(
                f"No hay muestras tras aplicar mapping_mode={self.mapping_mode!r} "
                f"en {self.metadata_csv} para split={split!r}."
            )

        self.use_mfcc = use_mfcc
        self.use_scalars = use_scalars
        self.use_waveform = use_waveform  # ⬇️ NUEVO
        self.scalar_columns = list(scalar_columns or DEFAULT_SCALAR_COLUMNS)

        self.mel_column = self._resolve_existing_column(self.df.columns, mel_column_candidates)
        self.mfcc_column = self._resolve_existing_column(self.df.columns, mfcc_column_candidates, required=False)
        self.waveform_column = self._resolve_existing_column(self.df.columns, waveform_column_candidates, required=False)  # ⬇️ NUEVO

        self.available_scalar_columns = [c for c in self.scalar_columns if c in self.df.columns]

        # 🔥 OPTIMIZACIÓN: Normalizar paths UNA SOLA VEZ
        self._normalize_dataframe_paths()

        # 🔥 OPTIMIZACIÓN: Inferir dimensiones sin loops iterrows
        self.target_frames = target_frames or self._infer_target_frames_fast()
        self.mel_bins, self.mfcc_bins = self._infer_feature_bins_fast()
        self.waveform_samples = self._infer_waveform_samples_fast() if self.use_waveform and self.waveform_column else None  # ⬇️ NUEVO

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
    def _infer_mode_from_pkl_path(pkl_path: str, target_col_in_pickle: str) -> str:
        """
        Autoinfiere el modo a partir del nombre del archivo pickle y/o el
        campo target_col guardado en su interior.

        Prioridad: nombre del archivo > target_col del pickle.
        """
        name = Path(pkl_path).stem.lower()

        if "emergency" in name:
            return "emergency"
        if "no_alertable" in name:
            return "no_alertable"
        if "total" in name:
            return "total"
        if "human" in name:
            return "human"
        if "alertable" in name:
            return "alertable"

        # Fallback: usar target_col guardado en el pickle
        if target_col_in_pickle == "alertable":
            return "alertable"

        return "total"  # default seguro

    def _filter_by_mode(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        Filtra las filas del dataframe según el modo de mapping:
          - "total"        : sin filtro
          - "human"        : alertable == True
          - "alertable"    : sin filtro (target es la propia columna alertable)
          - "emergency"    : alertable == True AND emergency == True
          - "no_alertable" : alertable == False
        """
        if mode == "total" or mode == "alertable":
            return df.reset_index(drop=True)

        if mode in ("human", "emergency", "no_alertable"):
            if "alertable" not in df.columns:
                raise KeyError(
                    f"mapping_mode={mode!r} requiere la columna 'alertable' en el CSV."
                )

        if mode == "no_alertable":
            mask = df["alertable"].apply(self._normalize_alertable_value) == False  # noqa: E712
            return df[mask].reset_index(drop=True)

        if mode in ("human", "emergency"):
            alertable_mask = df["alertable"].apply(self._normalize_alertable_value) == True  # noqa: E712
            df = df[alertable_mask]

        if mode == "emergency":
            if "emergency" not in df.columns:
                raise KeyError(
                    "mapping_mode='emergency' requiere la columna 'emergency' en el CSV."
                )
            emergency_mask = df["emergency"].apply(self._normalize_alertable_value) == True  # noqa: E712
            df = df[emergency_mask]

        return df.reset_index(drop=True)

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

        for col in [self.mel_column, self.mfcc_column, self.waveform_column, "audio_normalized_path"]:
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

    def _infer_waveform_samples_fast(self) -> Optional[int]:
        """
        🔥 NUEVO: Inferir número de samples en waveform.
        """
        if not self.use_waveform or self.waveform_column is None:
            return None

        for waveform_path in self.df[self.waveform_column]:
            if isinstance(waveform_path, str) and Path(waveform_path).exists():
                try:
                    shape, _ = self._load_npy_header(waveform_path)
                    # Waveform es 1D: (num_samples,)
                    if len(shape) == 1:
                        return int(shape[0])
                    # O 2D: (1, num_samples) o (num_samples, 1)
                    elif len(shape) == 2:
                        return int(shape[-1]) if shape[0] == 1 else int(shape[0])
                except Exception:
                    continue

        return None

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

    def _fix_waveform_length(self, waveform: np.ndarray, target_samples: int) -> np.ndarray:
        """
        🔥 NUEVO: Ajusta la longitud del waveform a target_samples.
        """
        waveform = waveform.flatten()  # Asegurar 1D
        
        if len(waveform) > target_samples:
            waveform = waveform[:target_samples]
        elif len(waveform) < target_samples:
            pad = target_samples - len(waveform)
            waveform = np.pad(waveform, (0, pad), mode="constant")

        return waveform.astype(np.float32)

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

        # 🔥 NUEVO: Cargar waveform si existe y se solicita
        if (
            self.use_waveform
            and self.waveform_column is not None
            and pd.notna(row.get(self.waveform_column))
            and self.waveform_samples is not None
        ):
            try:
                waveform = self._load_npy(row[self.waveform_column])
                waveform = self._fix_waveform_length(waveform, self.waveform_samples)
                sample["waveform"] = torch.from_numpy(waveform).unsqueeze(0)
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
    Maneja mel, mfcc, waveform y scalars.
    """
    mel = torch.stack([b["mel"] for b in batch])

    out = {"mel": mel}

    if "mfcc" in batch[0]:
        out["mfcc"] = torch.stack([b["mfcc"] for b in batch])

    # 🔥 NUEVO: Manejo de waveform
    if "waveform" in batch[0]:
        out["waveform"] = torch.stack([b["waveform"] for b in batch])

    if "scalars" in batch[0]:
        out["scalars"] = torch.stack([b["scalars"] for b in batch])

    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    filenames = [b["filename"] for b in batch]

    return out, labels, filenames
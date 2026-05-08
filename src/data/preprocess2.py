
"""preprocess.py

Preprocesado de audio centrado en PyTorch / torchaudio.

Objetivos:
- minimizar dependencia de librosa
- empujar al máximo posible el trabajo útil a GPU cuando exista CUDA/ROCm
- mantener una API simple con una clase Preprocess
- guardar artefactos y un CSV final listo para entrenar

Qué hace:
- carga audios desde un CSV
- resample en GPU si está disponible
- mono / normalización / padding-trim
- augmentaciones tensoriales
- mel spectrogram y MFCC en torchaudio
- features escalares calculadas con torch
- guardado de waveform normalizada, mel, mfcc y metadata
- label encoding
- Dataset y collate_fn para entrenamiento posterior

Notas:
- La lectura del archivo de audio sigue ocurriendo en CPU porque así funciona torchaudio.load.
- Para exprimir GPU de verdad, este pipeline evita librosa y usa transformaciones tensoriales.
- En modo GPU, por defecto se desactiva multiprocessing porque varias workers peleando por una sola GPU suele empeorar el rendimiento.
"""

from __future__ import annotations

import math
import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

PathLike = Union[str, Path]


@dataclass
class PreprocessConfig:
    sample_rate: int = 22050
    n_mels: int = 128
    n_mfcc: int = 40
    n_fft: int = 2048
    hop_length: int = 512
    target_duration: Optional[float] = 5.0  # segundos; None = no fijar longitud
    normalize_peak: bool = True
    peak_target: float = 0.99
    augment: bool = False
    save_waveform: bool = True
    save_mel: bool = True
    save_mfcc: bool = True
    save_metadata_csv: bool = True
    save_label_mapping: bool = True
    file_path_candidates: Tuple[str, ...] = ("file_path", "path", "filepath", "audio_path", "filename")
    label_candidates: Tuple[str, ...] = ("label", "keywords", "class", "target", "category")
    split_candidates: Tuple[str, ...] = ("split", "dataset", "subset", "partition")
    output_subdir: str = "processed_dataset"
    audio_subdir: str = "audio"
    mel_subdir: str = "mel"
    mfcc_subdir: str = "mfcc"
    metadata_filename: str = "processed_metadata.csv"
    label_mapping_filename: str = "label_mapping.pkl"
    use_multiprocessing: bool = False
    num_workers: Optional[int] = None
    chunksize: int = 32
    feature_format: str = "pt"  # "pt" recomendado; "npy" también soportado para mel/mfcc


class Preprocess:
    def __init__(
        self,
        csv_path: Optional[PathLike] = None,
        raw_dir: Optional[PathLike] = None,
        interim_dir: Optional[PathLike] = None,
        config: Optional[PreprocessConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or PreprocessConfig()
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.raw_dir = Path(raw_dir) if raw_dir is not None else None
        self.interim_dir = Path(interim_dir) if interim_dir is not None else None

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.device_name = (
            torch.cuda.get_device_name(0)
            if self.device.type == "cuda" and torch.cuda.is_available()
            else "CPU"
        )

        self.df: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.label_mapping: Optional[Dict[str, int]] = None
        self.path_column: Optional[str] = None
        self.label_column: Optional[str] = None
        self.split_column: Optional[str] = None

        self.output_dir = self._resolve_output_dir()
        self.audio_dir = self.output_dir / self.config.audio_subdir
        self.mel_dir = self.output_dir / self.config.mel_subdir
        self.mfcc_dir = self.output_dir / self.config.mfcc_subdir
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.mel_dir.mkdir(parents=True, exist_ok=True)
        self.mfcc_dir.mkdir(parents=True, exist_ok=True)

        self.num_workers = self._resolve_num_workers(self.config.num_workers)
        self.use_multiprocessing = bool(self.config.use_multiprocessing and self.device.type != "cuda" and self.num_workers > 1)

        self._resampler_cache: Dict[int, torchaudio.transforms.Resample] = {}
        self._window_cache: Dict[Tuple[torch.device, int], torch.Tensor] = {}

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            power=2.0,
        ).to(self.device)

        self.amplitude_to_db = T.AmplitudeToDB(stype="power").to(self.device)

        self.mfcc_transform = T.MFCC(
            sample_rate=self.config.sample_rate,
            n_mfcc=self.config.n_mfcc,
            melkwargs={
                "n_fft": self.config.n_fft,
                "hop_length": self.config.hop_length,
                "n_mels": self.config.n_mels,
                "center": True,
                "power": 2.0,
            },
        ).to(self.device)

    # ------------------------------------------------------------------
    # Resolución de rutas y columnas
    # ------------------------------------------------------------------
    def _resolve_output_dir(self) -> Path:
        if self.interim_dir is not None:
            out = self.interim_dir / self.config.output_subdir
        else:
            out = Path(self.config.output_subdir)
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _resolve_num_workers(self, requested: Optional[int]) -> int:
        if requested is not None:
            return max(0, int(requested))
        cpu = os.cpu_count() or 1
        return max(1, min(8, cpu - 1))

    def _infer_column(self, df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
        lower_map = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        if required:
            raise ValueError(
                f"No pude inferir una columna válida. Candidatas: {candidates}. Columnas disponibles: {list(df.columns)}"
            )
        return None

    def _load_dataframe(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is not None:
            self.df = df.copy()
            return self.df

        if self.csv_path is None:
            raise ValueError("Debes pasar csv_path o un dataframe.")
        self.df = pd.read_csv(self.csv_path)
        return self.df

    def _prepare_columns(self, df: pd.DataFrame) -> None:
        self.path_column = self._infer_column(df, self.config.file_path_candidates, required=True)
        self.label_column = self._infer_column(df, self.config.label_candidates, required=False)
        self.split_column = self._infer_column(df, self.config.split_candidates, required=False)

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------
    def _audio_path_from_row(self, row: pd.Series) -> Path:
        assert self.path_column is not None

        raw_value = str(row[self.path_column])
        candidate = Path(raw_value)

        if candidate.is_absolute() and candidate.exists():
            return candidate

        if self.raw_dir is not None:
            joined = self.raw_dir / candidate
            if joined.exists():
                return joined

            basename_joined = self.raw_dir / candidate.name
            if basename_joined.exists():
                return basename_joined

        if candidate.exists():
            return candidate

        # último intento: interpretar como relativo al cwd
        return candidate

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        if orig_sr == self.config.sample_rate:
            return None  # type: ignore[return-value]

        if orig_sr not in self._resampler_cache:
            self._resampler_cache[orig_sr] = T.Resample(
                orig_freq=orig_sr,
                new_freq=self.config.sample_rate,
            ).to(self.device)
        return self._resampler_cache[orig_sr]

    def _get_window(self, n_fft: int) -> torch.Tensor:
        key = (self.device, n_fft)
        if key not in self._window_cache:
            self._window_cache[key] = torch.hann_window(n_fft, device=self.device)
        return self._window_cache[key]

    @staticmethod
    def _ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    def _normalize_peak(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize_peak:
            return waveform
        peak = waveform.abs().max().clamp_min(1e-8)
        return waveform / peak * float(self.config.peak_target)

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.config.target_duration is None:
            return waveform
        target_length = int(self.config.sample_rate * float(self.config.target_duration))
        current = waveform.shape[-1]
        if current > target_length:
            return waveform[..., :target_length]
        if current < target_length:
            pad = target_length - current
            return F.pad(waveform, (0, pad))
        return waveform

    def _augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        # Todo se hace con tensores para que, si hay GPU, ocurra ahí.
        if torch.rand(1, device=waveform.device).item() < 0.55:
            noise = torch.randn_like(waveform) * 0.003
            waveform = waveform + noise

        if torch.rand(1, device=waveform.device).item() < 0.45:
            gain = torch.empty(1, device=waveform.device).uniform_(0.85, 1.15)
            waveform = waveform * gain

        if waveform.shape[-1] > 1 and torch.rand(1, device=waveform.device).item() < 0.35:
            max_shift = max(1, int(0.08 * waveform.shape[-1]))
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,), device=waveform.device).item())
            waveform = torch.roll(waveform, shifts=shift, dims=-1)

        return waveform

    def _save_tensor(self, tensor: torch.Tensor, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.feature_format.lower() == "npy":
            np.save(path.with_suffix(".npy"), tensor.detach().cpu().numpy())
            return str(path.with_suffix(".npy"))
        torch.save(tensor.detach().cpu(), path.with_suffix(".pt"))
        return str(path.with_suffix(".pt"))

    def _compute_stft_magnitude(self, waveform: torch.Tensor) -> torch.Tensor:
        window = self._get_window(self.config.n_fft)
        spec = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        return spec.abs().squeeze(0)

    def _compute_scalar_features(self, waveform: torch.Tensor) -> Dict[str, float]:
        # waveform: [1, T]
        eps = 1e-10
        x = waveform.squeeze(0)

        rms = torch.sqrt(torch.mean(x ** 2)).item()

        if x.numel() > 1:
            zcr = ((x[:-1] * x[1:]) < 0).float().mean().item()
        else:
            zcr = 0.0

        mag = self._compute_stft_magnitude(waveform)  # [freq, frames]
        mag_mean = mag.mean(dim=-1)  # [freq]

        freqs = torch.linspace(
            0.0,
            float(self.config.sample_rate) / 2.0,
            mag_mean.shape[0],
            device=mag_mean.device,
        )

        mag_sum = mag_mean.sum().clamp_min(eps)
        centroid = (freqs * mag_mean).sum() / mag_sum
        bandwidth = torch.sqrt((((freqs - centroid) ** 2) * mag_mean).sum() / mag_sum)

        cumulative = torch.cumsum(mag_mean, dim=0)
        threshold = 0.85 * mag_sum
        rolloff_idx = torch.searchsorted(cumulative, threshold).clamp(max=mag_mean.shape[0] - 1)
        rolloff = freqs[rolloff_idx]

        flatness = torch.exp(torch.mean(torch.log(mag_mean.clamp_min(eps)))) / mag_mean.mean().clamp_min(eps)

        return {
            "rms": float(rms),
            "zcr": float(zcr),
            "spectral_centroid": float(centroid.item()),
            "spectral_bandwidth": float(bandwidth.item()),
            "spectral_rolloff": float(rolloff.item()),
            "spectral_flatness": float(flatness.item()),
        }

    # ------------------------------------------------------------------
    # Procesamiento individual
    # ------------------------------------------------------------------
    def process_single_audio(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        try:
            assert self.path_column is not None

            audio_path = self._audio_path_from_row(row)
            if not audio_path.exists():
                return None

            waveform, sr = torchaudio.load(str(audio_path))
            waveform = self._ensure_mono(waveform).to(self.device)

            if sr != self.config.sample_rate:
                resampler = self._get_resampler(sr)
                if resampler is not None:
                    waveform = resampler(waveform)

            waveform = self._fix_length(waveform)
            waveform = self._normalize_peak(waveform)

            if self.config.augment:
                waveform = self._augment_waveform(waveform)

            with torch.inference_mode():
                mel = self.mel_transform(waveform)
                mel_db = self.amplitude_to_db(mel)
                mfcc = self.mfcc_transform(waveform)

            scalar_features = self._compute_scalar_features(waveform)

            stem = audio_path.stem
            wav_out = self.audio_dir / f"{stem}_normalized.wav"
            mel_out = self.mel_dir / f"{stem}_mel"
            mfcc_out = self.mfcc_dir / f"{stem}_mfcc"

            result: Dict[str, Any] = row.to_dict()

            if self.config.save_waveform:
                wav_out.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(
                    str(wav_out),
                    waveform.detach().cpu(),
                    sample_rate=self.config.sample_rate,
                )
                result["waveform_path"] = str(wav_out)

            if self.config.save_mel:
                result["mel_path"] = self._save_tensor(mel_db, mel_out)

            if self.config.save_mfcc:
                result["mfcc_path"] = self._save_tensor(mfcc, mfcc_out)

            result.update(
                {
                    "sample_rate": self.config.sample_rate,
                    "num_channels": 1,
                    "duration_sec": float(waveform.shape[-1] / self.config.sample_rate),
                    "device_used": self.device.type,
                }
            )
            result.update(scalar_features)

            return result

        except Exception as e:
            print(f"❌ Error procesando {row.get(self.path_column, 'unknown')}: {e}")
            return None

    # ------------------------------------------------------------------
    # Procesamiento de dataframe
    # ------------------------------------------------------------------
    def process_dataframe(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        data = self._load_dataframe(df)
        self._prepare_columns(data)

        rows = [row for _, row in data.iterrows()]
        results: List[Dict[str, Any]] = []

        print(f"📁 Output dir: {self.output_dir}")
        print(f"🚀 Procesando {len(rows)} audios")
        print(f"Dispositivo: {self.device} | {self.device_name}")
        print(f"Augmentaciones: {self.config.augment}")
        print(f"Multiprocessing: {self.use_multiprocessing}")

        # En GPU, secuencial suele ser mejor para este tipo de pipeline.
        iterator = rows
        for row in tqdm(iterator, total=len(rows), desc="Procesando audios"):
            processed = self.process_single_audio(row)
            if processed is not None:
                results.append(processed)

        self.df_processed = pd.DataFrame(results)
        return self.df_processed

    # ------------------------------------------------------------------
    # Label encoding
    # ------------------------------------------------------------------
    def encode_labels(self, df: pd.DataFrame, label_column: Optional[str] = None) -> pd.DataFrame:
        label_column = label_column or self.label_column
        if label_column is None or label_column not in df.columns:
            return df

        le = LabelEncoder()
        encoded = df.copy()
        encoded["label_encoded"] = le.fit_transform(encoded[label_column].astype(str))

        self.label_encoder = le
        self.label_mapping = {label: int(idx) for idx, label in enumerate(le.classes_)}

        return encoded

    def save_label_mapping(self) -> None:
        if not self.config.save_label_mapping or self.label_mapping is None:
            return
        out = self.output_dir / self.config.label_mapping_filename
        with open(out, "wb") as f:
            pickle.dump(self.label_mapping, f)

    # ------------------------------------------------------------------
    # Guardado final
    # ------------------------------------------------------------------
    def save_processed_dataframe(self, df: Optional[pd.DataFrame] = None) -> Path:
        if not self.config.save_metadata_csv:
            raise RuntimeError("save_metadata_csv está desactivado.")
        data = df if df is not None else self.df_processed
        if data is None:
            raise ValueError("No hay dataframe procesado para guardar.")
        out = self.output_dir / self.config.metadata_filename
        data.to_csv(out, index=False)
        return out

    def run(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        processed = self.process_dataframe(df)
        processed = self.encode_labels(processed)

        self.df_processed = processed

        if self.config.save_metadata_csv:
            self.save_processed_dataframe(processed)

        if self.config.save_label_mapping and self.label_mapping is not None:
            self.save_label_mapping()

        return processed

    # ------------------------------------------------------------------
    # Dataset y collate
    # ------------------------------------------------------------------
    class AudioFeatureDataset(Dataset):
        def __init__(
            self,
            dataframe: pd.DataFrame,
            mel_column: str = "mel_path",
            mfcc_column: str = "mfcc_path",
            label_column: str = "label_encoded",
            transform: Optional[Any] = None,
        ) -> None:
            self.df = dataframe.reset_index(drop=True)
            self.mel_column = mel_column
            self.mfcc_column = mfcc_column
            self.label_column = label_column
            self.transform = transform

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            row = self.df.iloc[idx]
            sample: Dict[str, Any] = {}

            mel_path = row.get(self.mel_column)
            mfcc_path = row.get(self.mfcc_column)

            if pd.notna(mel_path) and str(mel_path):
                sample["mel"] = torch.load(str(mel_path), map_location="cpu")
            if pd.notna(mfcc_path) and str(mfcc_path):
                sample["mfcc"] = torch.load(str(mfcc_path), map_location="cpu")

            sample["label"] = torch.tensor(int(row[self.label_column]), dtype=torch.long)

            if self.transform is not None:
                sample = self.transform(sample)

            return sample

    @staticmethod
    def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = torch.stack([item["label"] for item in batch], dim=0)

        collated: Dict[str, Any] = {"label": labels}
        if "mel" in batch[0]:
            collated["mel"] = torch.stack([item["mel"] for item in batch], dim=0)
        if "mfcc" in batch[0]:
            collated["mfcc"] = torch.stack([item["mfcc"] for item in batch], dim=0)

        for key in batch[0].keys():
            if key in {"mel", "mfcc", "label"}:
                continue
            collated[key] = [item.get(key) for item in batch]

        return collated

    def build_dataset(
        self,
        df: Optional[pd.DataFrame] = None,
        mel_column: str = "mel_path",
        mfcc_column: str = "mfcc_path",
        label_column: str = "label_encoded",
        transform: Optional[Any] = None,
    ) -> "Preprocess.AudioFeatureDataset":
        data = df if df is not None else self.df_processed
        if data is None:
            raise ValueError("No hay dataframe procesado disponible.")
        return Preprocess.AudioFeatureDataset(
            data,
            mel_column=mel_column,
            mfcc_column=mfcc_column,
            label_column=label_column,
            transform=transform,
        )

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "device_name": self.device_name,
            "output_dir": str(self.output_dir),
            "sample_rate": self.config.sample_rate,
            "target_duration": self.config.target_duration,
            "augment": self.config.augment,
            "use_multiprocessing": self.use_multiprocessing,
            "num_workers": self.num_workers,
        }


if __name__ == "__main__":
    print("Este módulo está pensado para importarse.")

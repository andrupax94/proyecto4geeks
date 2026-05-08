
"""preprocess.py

Pipeline de preprocesado de audio basado en el notebook 02_preprocessing.ipynb.

Incluye:
- carga de audios
- resampling
- normalización / mono / padding-trim
- extracción de mel spectrogram y MFCC
- features escalares
- augmentaciones
- guardado de features y metadatos
- label encoding
- Dataset y collate_fn para PyTorch

Pensado para maximizar el uso de GPU en las partes que sí se benefician de ella
(MelSpectrogram, MFCC, resampling, operaciones tensoriales).
"""

from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")


PathLike = Union[str, Path]


@dataclass
class PreprocessConfig:
    sample_rate: int = 44100
    n_mels: int = 128
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512
    target_duration: Optional[float] = None  # seconds; None means no fixed length
    batch_size: int = 32
    num_workers: Optional[int] = None
    augment: bool = False
    use_multiprocessing: bool = True
    save_audio_normalized: bool = True
    save_mel: bool = True
    save_mfcc: bool = True
    save_metadata_csv: bool = True
    save_label_mapping: bool = True
    file_path_candidates: Tuple[str, ...] = ("file_path", "path", "filepath", "audio_path")
    label_candidates: Tuple[str, ...] = ("label", "keywords", "class", "target")
    split_candidates: Tuple[str, ...] = ("split", "dataset", "subset")
    output_subdir: str = "processed_dataset"


class Preprocess:
    """
    Clase principal de preprocesado.

    Uso típico:
        pp = Preprocess(csv_path="data/raw/dataset_final.csv", raw_dir="data/raw", interim_dir="data/interim")
        df_processed = pp.run()

    También puedes usar:
        results = pp.process_dataframe(df)
        ds = pp.build_dataset()
        train_loader, test_loader = pp.build_dataloaders()
    """

    def __init__(
        self,
        csv_path: Optional[PathLike] = None,
        raw_dir: Optional[PathLike] = None,
        interim_dir: Optional[PathLike] = None,
        config: Optional[PreprocessConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or PreprocessConfig()
        self.csv_path = Path(csv_path) if csv_path else None
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self.interim_dir = Path(interim_dir) if interim_dir else None

        self.device = torch.device(
            device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.device_name = (
            torch.cuda.get_device_name(0)
            if self.device.type == "cuda" and torch.cuda.is_available()
            else "CPU"
        )

        self.num_workers = self._resolve_num_workers(self.config.num_workers)

        self.df: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.label_mapping: Optional[Dict[str, int]] = None
        self.label_encoder: Optional[LabelEncoder] = None

        self.output_dir = self._resolve_output_dir()

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

        self._resampler_cache: Dict[int, T.Resample] = {}

    # --------------------------
    # Paths / config
    # --------------------------
    def _resolve_num_workers(self, requested: Optional[int]) -> int:
        if requested is not None:
            return max(0, int(requested))

        # En GPU conviene no pelear con varios procesos CUDA.
        if self.device.type == "cuda":
            return 2
        return max(1, cpu_count() - 1)

    def _resolve_output_dir(self) -> Path:
        if self.interim_dir is not None:
            out = self.interim_dir / self.config.output_subdir
        elif self.raw_dir is not None:
            out = self.raw_dir.parent / "interim" / self.config.output_subdir
        else:
            out = Path("processed_dataset")

        out.mkdir(parents=True, exist_ok=True)
        return out

    def _resolve_audio_path(self, row: pd.Series | Dict[str, Any], audio_dir: Optional[PathLike]) -> Optional[Path]:
        audio_dir_path = Path(audio_dir) if audio_dir else self.raw_dir

        if audio_dir_path is None:
            raise ValueError("Debes pasar `audio_dir` o inicializar `raw_dir`.")

        if isinstance(row, pd.Series):
            keys = row.index.tolist()
            getter = row.get
        else:
            keys = list(row.keys())
            getter = row.get

        rel_path = None
        for candidate in self.config.file_path_candidates:
            if candidate in keys and pd.notna(getter(candidate)):
                rel_path = str(getter(candidate))
                break

        if rel_path is None:
            raise KeyError(
                f"No encontré una columna de ruta válida. Probé: {self.config.file_path_candidates}"
            )

        rel_path = str(rel_path).replace("\\", os.sep).replace("/", os.sep)
        audio_path = audio_dir_path / rel_path
        return audio_path

    def _resolve_column_value(
        self,
        row: pd.Series | Dict[str, Any],
        candidates: Sequence[str],
        default: Any = None,
    ) -> Any:
        if isinstance(row, pd.Series):
            getter = row.get
            keys = row.index.tolist()
        else:
            getter = row.get
            keys = list(row.keys())

        for candidate in candidates:
            if candidate in keys and pd.notna(getter(candidate)):
                return getter(candidate)
        return default

    # --------------------------
    # Core audio operations
    # --------------------------
    def get_resampler(self, orig_sr: int) -> T.Resample:
        if orig_sr not in self._resampler_cache:
            self._resampler_cache[orig_sr] = T.Resample(
                orig_freq=orig_sr,
                new_freq=self.config.sample_rate,
            ).to(self.device)
        return self._resampler_cache[orig_sr]

    @staticmethod
    def _ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            return waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    def _normalize_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        peak = waveform.abs().max().clamp_min(1e-8)
        return waveform / peak

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.config.target_duration is None:
            return waveform

        target_length = int(self.config.sample_rate * self.config.target_duration)
        current_length = waveform.shape[-1]

        if current_length > target_length:
            return waveform[..., :target_length]
        if current_length < target_length:
            pad_size = target_length - current_length
            return F.pad(waveform, (0, pad_size))
        return waveform

    def load_audio(self, filepath: PathLike) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        """
        Carga audio en tensor. Mantiene el flujo listo para GPU.
        """
        try:
            waveform, sr = torchaudio.load(str(filepath))
            waveform = self._ensure_mono(waveform)
            waveform = self._normalize_waveform(waveform)
            return waveform, sr
        except Exception:
            try:
                y, sr = librosa.load(str(filepath), sr=None, mono=True)
                waveform = torch.from_numpy(y).float().unsqueeze(0)
                waveform = self._normalize_waveform(waveform)
                return waveform, sr
            except Exception as e:
                print(f"Error cargando {filepath}: {e}")
                return None, None

    def prepare_waveform(self, filepath: PathLike) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        waveform, sr = self.load_audio(filepath)
        if waveform is None or sr is None:
            return None, None

        waveform = waveform.to(self.device, non_blocking=True)

        if sr != self.config.sample_rate:
            resampler = self.get_resampler(sr)
            waveform = resampler(waveform)

        waveform = self._fix_length(waveform)
        waveform = self._normalize_waveform(waveform)
        return waveform, self.config.sample_rate

    def get_melspectrogram(self, waveform: torch.Tensor, as_uint8: bool = True) -> np.ndarray | torch.Tensor:
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        if not as_uint8:
            return mel_spec_db

        mel_np = mel_spec_db.detach().cpu().squeeze().numpy()
        min_v = mel_np.min()
        max_v = mel_np.max()
        mel_np = (mel_np - min_v) / (max_v - min_v + 1e-8)
        return (mel_np * 255).astype(np.uint8)

    def get_mfcc(self, waveform: torch.Tensor) -> np.ndarray:
        mfcc = self.mfcc_transform(waveform)
        return mfcc.detach().cpu().squeeze().numpy().astype(np.float32)

    def get_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Features escalarmente baratas pero útiles. Estas siguen en CPU porque
        librosa no las ejecuta en GPU de forma práctica.
        """
        features: Dict[str, float] = {}

        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=self.config.sample_rate)[0]
        features["spectral_centroid_mean"] = float(np.mean(spec_cent))
        features["spectral_centroid_std"] = float(np.std(spec_cent))

        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.config.sample_rate)[0]
        features["spectral_rolloff_mean"] = float(np.mean(spec_rolloff))
        features["spectral_rolloff_std"] = float(np.std(spec_rolloff))

        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))

        rms = librosa.feature.rms(y=audio)[0]
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))

        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.config.sample_rate)
        features["chroma_mean"] = float(np.mean(chroma))
        features["chroma_std"] = float(np.std(chroma))

        return features

    def waveform_to_numpy(self, waveform: torch.Tensor) -> np.ndarray:
        return waveform.detach().cpu().squeeze().numpy().astype(np.float32)

    def save_audio(self, waveform: torch.Tensor, filepath: PathLike, sample_rate: int) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(filepath), self.waveform_to_numpy(waveform), sample_rate)

    # --------------------------
    # Augmentations
    # --------------------------
    def gpu_add_noise(self, waveform: torch.Tensor, noise_factor: float = 0.005) -> torch.Tensor:
        noise = torch.randn_like(waveform) * noise_factor
        return waveform + noise

    def gpu_gain(self, waveform: torch.Tensor, min_gain: float = 0.8, max_gain: float = 1.2) -> torch.Tensor:
        gain = torch.empty(1, device=waveform.device).uniform_(min_gain, max_gain)
        return waveform * gain

    def gpu_time_shift(self, waveform: torch.Tensor, max_shift_ratio: float = 0.1) -> torch.Tensor:
        shift = int(waveform.shape[-1] * max_shift_ratio)
        if shift <= 0:
            return waveform
        offset = int(torch.randint(-shift, shift + 1, (1,), device=waveform.device).item())
        return torch.roll(waveform, shifts=offset, dims=-1)

    def cpu_time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        return librosa.effects.time_stretch(audio, rate=rate)

    def cpu_pitch_shift(self, audio: np.ndarray, semitones: int) -> np.ndarray:
        return librosa.effects.pitch_shift(audio, sr=self.config.sample_rate, n_steps=semitones)

    def augment_pipeline(
        self,
        waveform: torch.Tensor,
        include_cpu_augments: bool = True,
    ) -> List[torch.Tensor]:
        """
        Devuelve el original + augmentaciones.
        Las augmentaciones GPU-friendly se hacen en tensor.
        Las de time stretch / pitch shift siguen en CPU porque librosa no las
        acelera realmente con CUDA.
        """
        augmented = [waveform]

        # GPU-friendly
        augmented.append(self.gpu_add_noise(waveform))
        augmented.append(self.gpu_gain(waveform))
        augmented.append(self.gpu_time_shift(waveform))

        if include_cpu_augments:
            audio_np = self.waveform_to_numpy(waveform)

            for rate in (0.9, 1.1):
                try:
                    stretched = self.cpu_time_stretch(audio_np, rate=rate)
                    t = torch.from_numpy(stretched).float().unsqueeze(0).to(self.device)
                    t = self._fix_length(t)
                    augmented.append(self._normalize_waveform(t))
                except Exception:
                    pass

            for semitones in (-2, 2):
                try:
                    shifted = self.cpu_pitch_shift(audio_np, semitones=semitones)
                    t = torch.from_numpy(shifted).float().unsqueeze(0).to(self.device)
                    t = self._fix_length(t)
                    augmented.append(self._normalize_waveform(t))
                except Exception:
                    pass

        return augmented

    # --------------------------
    # Single file processing
    # --------------------------
    def process_single_audio(
        self,
        row: pd.Series | Dict[str, Any],
        audio_dir: Optional[PathLike] = None,
        output_dir: Optional[PathLike] = None,
        augment: Optional[bool] = None,
        include_cpu_augments: bool = True,
    ) -> Optional[Dict[str, Any]]:
        try:
            audio_path = self._resolve_audio_path(row, audio_dir)
            if audio_path is None or not audio_path.exists():
                return None

            out_dir = Path(output_dir) if output_dir else self.output_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            filename_base = audio_path.stem
            file_output = out_dir / filename_base
            file_output.mkdir(parents=True, exist_ok=True)

            waveform, sr = self.prepare_waveform(audio_path)
            if waveform is None or sr is None:
                return None

            row_label = self._resolve_column_value(row, self.config.label_candidates, default="unknown")
            row_split = self._resolve_column_value(row, self.config.split_candidates, default="unknown")

            base_result: Dict[str, Any] = {
                "filename": filename_base,
                "original_path": str(audio_path),
                "is_augmented": False,
                "aug_index": 0,
                "label": row_label,
                "split": row_split,
                "duration": float(waveform.shape[-1] / sr),
                "sample_rate": int(sr),
            }

            audio_np = self.waveform_to_numpy(waveform)
            base_result.update(self.get_spectral_features(audio_np))

            if self.config.save_mel:
                mel_spec = self.get_melspectrogram(waveform)
                mel_path = file_output / f"{filename_base}_mel.npy"
                np.save(mel_path, mel_spec)
                base_result["mel_spec_path"] = str(mel_path)

            if self.config.save_mfcc:
                mfcc = self.get_mfcc(waveform)
                mfcc_path = file_output / f"{filename_base}_mfcc.npy"
                np.save(mfcc_path, mfcc)
                base_result["mfcc_path"] = str(mfcc_path)

            if self.config.save_audio_normalized:
                normalized_path = file_output / f"{filename_base}_normalized.wav"
                self.save_audio(waveform, normalized_path, sr)
                base_result["audio_normalized_path"] = str(normalized_path)

            if augment if augment is not None else self.config.augment:
                augmented_waveforms = self.augment_pipeline(
                    waveform,
                    include_cpu_augments=include_cpu_augments,
                )

                # El original ya está representado por base_result.
                for aug_idx, aug_waveform in enumerate(augmented_waveforms[1:], start=1):
                    aug_result = dict(base_result)
                    aug_result["is_augmented"] = True
                    aug_result["aug_index"] = aug_idx

                    if self.config.save_mel:
                        mel_aug = self.get_melspectrogram(aug_waveform)
                        mel_aug_path = file_output / f"{filename_base}_mel_aug{aug_idx}.npy"
                        np.save(mel_aug_path, mel_aug)
                        aug_result["mel_spec_path"] = str(mel_aug_path)

                    if self.config.save_mfcc:
                        mfcc_aug = self.get_mfcc(aug_waveform)
                        mfcc_aug_path = file_output / f"{filename_base}_mfcc_aug{aug_idx}.npy"
                        np.save(mfcc_aug_path, mfcc_aug)
                        aug_result["mfcc_path"] = str(mfcc_aug_path)

                    if self.config.save_audio_normalized:
                        normalized_aug_path = file_output / f"{filename_base}_normalized_aug{aug_idx}.wav"
                        self.save_audio(aug_waveform, normalized_aug_path, sr)
                        aug_result["audio_normalized_path"] = str(normalized_aug_path)

                    # Guardamos cada augment como una fila adicional
                    self._append_augmented_result(aug_result)

            return base_result

        except Exception as e:
            print(f"Error procesando {self._resolve_column_value(row, self.config.file_path_candidates, default='unknown')}: {e}")
            return None

    def _append_augmented_result(self, result: Dict[str, Any]) -> None:
        """
        Almacena augmentaciones generadas por process_single_audio.
        Se usan al final del procesamiento.
        """
        if not hasattr(self, "_augmented_results"):
            self._augmented_results: List[Dict[str, Any]] = []
        self._augmented_results.append(result)

    # --------------------------
    # Dataset processing
    # --------------------------
    def load_dataframe(self, csv_path: Optional[PathLike] = None) -> pd.DataFrame:
        path = Path(csv_path) if csv_path else self.csv_path
        if path is None:
            raise ValueError("Debes indicar `csv_path` o pasar uno al constructor.")
        self.df = pd.read_csv(path)
        return self.df

    def process_dataframe(
        self,
        df: Optional[pd.DataFrame] = None,
        audio_dir: Optional[PathLike] = None,
        output_dir: Optional[PathLike] = None,
        augment: Optional[bool] = None,
        use_multiprocessing: Optional[bool] = None,
        chunksize: int = 32,
    ) -> pd.DataFrame:
        if df is None:
            if self.df is None:
                raise ValueError("No hay dataframe cargado.")
            df = self.df

        audio_dir_path = Path(audio_dir) if audio_dir else self.raw_dir
        if audio_dir_path is None:
            raise ValueError("Debes pasar `audio_dir` o inicializar `raw_dir`.")

        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        augment_flag = self.config.augment if augment is None else augment
        mp_flag = self.config.use_multiprocessing if use_multiprocessing is None else use_multiprocessing

        rows = df.to_dict("records")
        results: List[Dict[str, Any]] = []
        self._augmented_results = []

        print(f"📁 Audio dir: {audio_dir_path}")
        print(f"📁 Output dir: {out_dir}")
        print(f"🚀 Procesando {len(rows)} audios")
        print(f"Dispositivo: {self.device} | {self.device_name}")
        print(f"Augmentaciones: {augment_flag}")
        print(f"Multiprocessing: {mp_flag and self.num_workers > 0}")

        # Si estamos en GPU, mejor evitar multiproceso por conflictos de CUDA.
        if self.device.type == "cuda":
            mp_flag = False

        if mp_flag and self.num_workers > 0:
            fn = partial(
                _process_row_worker,
                audio_dir=str(audio_dir_path),
                output_dir=str(out_dir),
                augment=augment_flag,
                include_cpu_augments=True,
                config=self.config,
            )
            with Pool(self.num_workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(fn, rows, chunksize=chunksize),
                    total=len(rows),
                    desc="Procesando audios",
                ):
                    if result is not None:
                        results.append(result)
        else:
            for row in tqdm(rows, total=len(rows), desc="Procesando audios"):
                result = self.process_single_audio(
                    row=row,
                    audio_dir=audio_dir_path,
                    output_dir=out_dir,
                    augment=augment_flag,
                    include_cpu_augments=True,
                )
                if result is not None:
                    results.append(result)

        if hasattr(self, "_augmented_results") and self._augmented_results:
            results.extend(self._augmented_results)

        df_processed = pd.DataFrame(results)
        self.df_processed = df_processed
        return df_processed

    def encode_labels(self, df_processed: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
        if df_processed is None:
            if self.df_processed is None:
                raise ValueError("No hay dataframe procesado.")
            df_processed = self.df_processed

        self.label_encoder = LabelEncoder()
        df_processed = df_processed.copy()
        df_processed["label_encoded"] = self.label_encoder.fit_transform(df_processed["label"])

        self.label_mapping = dict(
            zip(
                self.label_encoder.classes_,
                self.label_encoder.transform(self.label_encoder.classes_),
            )
        )

        self.df_processed = df_processed
        return df_processed, self.label_mapping

    def save_label_mapping(self, output_dir: Optional[PathLike] = None) -> Optional[Path]:
        if not self.config.save_label_mapping:
            return None
        if self.label_mapping is None:
            raise ValueError("No hay label mapping disponible. Ejecuta `encode_labels()` primero.")

        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        mapping_path = out_dir / "label_mapping.pkl"
        with open(mapping_path, "wb") as f:
            pickle.dump(self.label_mapping, f)
        return mapping_path

    def save_metadata_csv(
        self,
        df_processed: Optional[pd.DataFrame] = None,
        output_dir: Optional[PathLike] = None,
        filename: str = "processed_metadata.csv",
    ) -> Path:
        if df_processed is None:
            if self.df_processed is None:
                raise ValueError("No hay dataframe procesado.")
            df_processed = self.df_processed

        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        feature_cols = [
            "filename",
            "label",
            "label_encoded",
            "split",
            "mel_spec_path",
            "mfcc_path",
            "audio_normalized_path",
            "duration",
            "sample_rate",
            "is_augmented",
            "aug_index",
        ]
        extra_cols = [
            c for c in df_processed.columns
            if any(x in c for x in ("spectral", "rms", "zcr", "chroma"))
        ]
        cols = [c for c in feature_cols + extra_cols if c in df_processed.columns]

        df_features = df_processed[cols].copy()
        csv_out = out_dir / filename
        df_features.to_csv(csv_out, index=False)
        return csv_out

    def run(
        self,
        csv_path: Optional[PathLike] = None,
        df: Optional[pd.DataFrame] = None,
        audio_dir: Optional[PathLike] = None,
        output_dir: Optional[PathLike] = None,
        augment: Optional[bool] = None,
        use_multiprocessing: Optional[bool] = None,
        chunksize: int = 32,
        save_outputs: bool = True,
    ) -> pd.DataFrame:
        if df is None:
            df = self.load_dataframe(csv_path)

        df_processed = self.process_dataframe(
            df=df,
            audio_dir=audio_dir,
            output_dir=output_dir,
            augment=augment,
            use_multiprocessing=use_multiprocessing,
            chunksize=chunksize,
        )

        df_processed, _ = self.encode_labels(df_processed)

        if save_outputs:
            self.save_label_mapping(output_dir=output_dir)
            self.save_metadata_csv(df_processed, output_dir=output_dir)

        return df_processed

    # --------------------------
    # Dataset / loaders
    # --------------------------
    def build_dataset(
        self,
        metadata_csv: Optional[PathLike] = None,
        use_mel: bool = True,
        use_mfcc: bool = True,
    ) -> "AudioFeatureDataset":
        if metadata_csv is None:
            metadata_csv = self.save_metadata_csv()
        return AudioFeatureDataset(str(metadata_csv), use_mel=use_mel, use_mfcc=use_mfcc)

    def custom_collate_fn(self, batch):
        features_list, labels, filenames = zip(*batch)

        batch_dict: Dict[str, torch.Tensor] = {}
        labels_tensor = torch.LongTensor(labels)

        if "mel_spectrogram" in features_list[0]:
            mels = [f["mel_spectrogram"] for f in features_list]
            batch_dict["mel_spectrogram"] = torch.stack(mels)

        if "scalar_features" in features_list[0]:
            scalars = [f["scalar_features"] for f in features_list]
            batch_dict["scalar_features"] = torch.stack(scalars)

        if "mfcc" in features_list[0]:
            mfccs = [f["mfcc"] for f in features_list]
            max_time = max(m.shape[1] for m in mfccs)
            mfccs_padded = [
                F.pad(m, (0, max_time - m.shape[1]))
                for m in mfccs
            ]
            batch_dict["mfcc"] = torch.stack(mfccs_padded)

        return batch_dict, labels_tensor, filenames

    def build_dataloaders(
        self,
        metadata_csv: Optional[PathLike] = None,
        use_mel: bool = True,
        use_mfcc: bool = True,
        batch_size: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        if self.df_processed is None:
            if metadata_csv is None:
                raise ValueError("Necesitas `df_processed` o `metadata_csv`.")
            df_meta = pd.read_csv(metadata_csv)
        else:
            df_meta = self.df_processed.copy()

        if "split" not in df_meta.columns:
            raise KeyError("El dataframe no tiene columna `split`.")

        dataset = AudioFeatureDataset(str(metadata_csv) if metadata_csv else str(self.save_metadata_csv()), use_mel=use_mel, use_mfcc=use_mfcc)

        train_idx = df_meta[df_meta["split"] == "train"].index.tolist()
        test_idx = df_meta[df_meta["split"] == "test"].index.tolist()

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)

        bs = batch_size or self.config.batch_size
        pin = self.device.type == "cuda"

        train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=0,
            collate_fn=self.custom_collate_fn,
            pin_memory=pin,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            collate_fn=self.custom_collate_fn,
            pin_memory=pin,
        )
        return train_loader, test_loader


def _process_row_worker(
    row: Dict[str, Any],
    audio_dir: str,
    output_dir: str,
    augment: bool,
    include_cpu_augments: bool,
    config: PreprocessConfig,
) -> Optional[Dict[str, Any]]:
    """
    Worker para multiproceso CPU. No comparte GPU entre procesos.
    """
    try:
        # En worker se usa una instancia ligera y local.
        pp = Preprocess(
            raw_dir=audio_dir,
            interim_dir=Path(output_dir).parent,
            config=config,
            device="cpu",
        )
        return pp.process_single_audio(
            row=row,
            audio_dir=audio_dir,
            output_dir=output_dir,
            augment=augment,
            include_cpu_augments=include_cpu_augments,
        )
    except Exception as e:
        print(f"Worker error: {e}")
        return None


class AudioFeatureDataset(Dataset):
    """Dataset para consumir el CSV de metadatos generado por Preprocess."""

    def __init__(self, metadata_csv: str, use_mel: bool = True, use_mfcc: bool = True):
        self.df = pd.read_csv(metadata_csv)
        self.use_mel = use_mel
        self.use_mfcc = use_mfcc

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features_dict: Dict[str, torch.Tensor] = {}

        if self.use_mel and "mel_spec_path" in row and pd.notna(row["mel_spec_path"]):
            mel = np.load(row["mel_spec_path"]).astype(np.float32)
            features_dict["mel_spectrogram"] = torch.from_numpy(mel).unsqueeze(0)

        if self.use_mfcc and "mfcc_path" in row and pd.notna(row["mfcc_path"]):
            mfcc = np.load(row["mfcc_path"]).astype(np.float32)
            features_dict["mfcc"] = torch.from_numpy(mfcc)

        scalar_features = [
            "spectral_centroid_mean", "spectral_centroid_std",
            "spectral_rolloff_mean", "spectral_rolloff_std",
            "zcr_mean", "zcr_std",
            "rms_mean", "rms_std",
            "chroma_mean", "chroma_std",
        ]

        scalar_values: List[float] = []
        for feat in scalar_features:
            if feat in row and pd.notna(row[feat]):
                scalar_values.append(float(row[feat]))

        if scalar_values:
            features_dict["scalar_features"] = torch.FloatTensor(scalar_values)

        label = int(row["label_encoded"])
        return features_dict, label, row["filename"]


def main():
    """
    Ejemplo de uso. Ajusta las rutas según tu proyecto.
    """
    root = Path().resolve()
    data_dir = root / "data"
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    csv_path = raw_dir / "dataset_final.csv"

    pp = Preprocess(
        csv_path=csv_path,
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        config=PreprocessConfig(
            sample_rate=44100,
            n_mels=128,
            n_mfcc=13,
            target_duration=None,
            batch_size=32,
            augment=False,
            use_multiprocessing=True,
        ),
    )

    df_processed = pp.run(
        audio_dir=raw_dir,
        output_dir=interim_dir / "processed_dataset",
        augment=False,
        use_multiprocessing=False if pp.device.type == "cuda" else True,
        chunksize=32,
        save_outputs=True,
    )

    print(df_processed.head())
    print(f"Procesados: {len(df_processed)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import re
import warnings
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

from src.utils.config import RAW_DIR, INTERIM_DIR

warnings.filterwarnings("ignore")

PathLike = Union[str, Path]


@dataclass
class PreprocessConfig:
    # Estandarización de audio
    sample_rate: int = 44100
    target_duration: Optional[float] = 5.0
    normalize_peak: bool = True
    peak_target: float = 0.99
    augment: bool = False

    # Formato de salida estandarizado
    save_audio: bool = True
    output_audio_filename: str = "audio_standardized.wav"
    output_audio_format: str = "WAV"
    output_audio_subtype: str = "PCM_16"

    # Features
    n_mels: int = 128
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512

    save_waveform: bool = False
    save_mel: bool = True
    save_mfcc: bool = True

    # Salida
    output_subdir: str = "processed_dataset"
    metadata_filename: str = "processed_metadataV2.csv"

    label_mapping_human_filename: str = "label_mapping_human_labelV2.pkl"
    label_mapping_alertable_filename: str = "label_mapping_alertableV2.pkl"
    label_mapping_emergency_filename: str = "label_mapping_emergencyV2.pkl"
    label_mapping_total: str = "label_mapping_totalV2.pkl"
    label_mapping_human_no_alertable_filename: str = "label_mapping_human_no_alertableV2.pkl"

    file_path_candidates: Sequence[str] = ("file_path", "path", "filepath", "audio_path", "filename", "audio")
    label_candidates: Sequence[str] = ("human_label", "human_labels", "label", "keywords", "class", "target", "category")
    dataset_source_candidates: Sequence[str] = ("dataset_source", "source", "dataset")
    split_candidates: Sequence[str] = ("split", "subset", "partition")

    use_multiprocessing: bool = False
    num_workers: Optional[int] = None


class Preprocess:
    def __init__(
        self,
        csv_paths: Optional[Sequence[PathLike]] = None,
        raw_dir: Optional[PathLike] = None,
        interim_dir: Optional[PathLike] = None,
        config: Optional[PreprocessConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or PreprocessConfig()
        self.csv_paths = [Path(p) for p in csv_paths] if csv_paths is not None else []
        self.raw_dir = Path(raw_dir) if raw_dir is not None else None
        self.interim_dir = Path(interim_dir) if interim_dir is not None else None

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device_name = (
            torch.cuda.get_device_name(0)
            if self.device.type == "cuda" and torch.cuda.is_available()
            else "CPU"
        )

        self.output_dir = self._resolve_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.label_mapping_human_path = self.output_dir / self.config.label_mapping_human_filename
        self.label_mapping_alertable_path = self.output_dir / self.config.label_mapping_alertable_filename
        self.label_mapping_emergency_path = self.output_dir / self.config.label_mapping_emergency_filename
        self.label_mapping_total_path = self.output_dir / self.config.label_mapping_total
        self.label_mapping_human_no_alertable_path = self.output_dir / self.config.label_mapping_human_no_alertable_filename
        self.metadata_path = self.output_dir / self.config.metadata_filename

        self._resampler_cache: Dict[int, torchaudio.transforms.Resample] = {}
        self._window_cache: Dict[tuple, torch.Tensor] = {}

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

    def _save_pickle(self, obj: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

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

    def _build_label_mapping(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        if target_col not in df.columns:
            raise ValueError(f"No existe la columna '{target_col}' en el dataframe.")

        series = df[target_col].dropna()

        if target_col == "alertable":
            values = []
            for v in series.tolist():
                norm = self._normalize_alertable_value(v)
                if norm is not None:
                    values.append(norm)

            unique_values = sorted(set(values))
            label2idx = {bool(v): i for i, v in enumerate(unique_values)}
            idx2label = {i: bool(v) for v, i in label2idx.items()}
        else:
            values = [str(v).strip() for v in series.tolist() if str(v).strip()]
            unique_values = sorted(set(values), key=lambda x: x.lower())
            label2idx = {label: i for i, label in enumerate(unique_values)}
            idx2label = {i: label for label, i in label2idx.items()}

        return {
            "target_col": target_col,
            "label2idx": label2idx,
            "idx2label": idx2label,
            "num_classes": len(label2idx),
        }

    def _generate_label_mappings(self, df: pd.DataFrame) -> None:
        if "human_label" in df.columns:
            total_map = self._build_label_mapping(df, "human_label")
            self._save_pickle(total_map, self.label_mapping_total_path)
            print(f"🧩 Label mapping guardado (total): {self.label_mapping_total_path}")

        if "human_label" in df.columns and "alertable" in df.columns:
            alertable_mask = df["alertable"].apply(self._normalize_alertable_value) == True  # noqa: E712
            df_alertable = df[alertable_mask]
            if not df_alertable.empty:
                human_map = self._build_label_mapping(df_alertable, "human_label")
            else:
                human_map = {"target_col": "human_label", "label2idx": {}, "idx2label": {}, "num_classes": 0}
            self._save_pickle(human_map, self.label_mapping_human_path)
            print(f"🧩 Label mapping guardado (human alertable): {self.label_mapping_human_path}")

        if "alertable" in df.columns:
            alert_map = self._build_label_mapping(df, "alertable")
            self._save_pickle(alert_map, self.label_mapping_alertable_path)
            print(f"🧩 Label mapping guardado (alertable): {self.label_mapping_alertable_path}")

        if "human_label" in df.columns and "alertable" in df.columns and "emergency" in df.columns:
            alertable_mask = df["alertable"].apply(self._normalize_alertable_value) == True  # noqa: E712
            emergency_mask = df["emergency"].apply(self._normalize_alertable_value) == True  # noqa: E712
            df_emergency = df[alertable_mask & emergency_mask]
            if not df_emergency.empty:
                emergency_map = self._build_label_mapping(df_emergency, "human_label")
            else:
                emergency_map = {"target_col": "human_label", "label2idx": {}, "idx2label": {}, "num_classes": 0}
            self._save_pickle(emergency_map, self.label_mapping_emergency_path)
            print(f"🧩 Label mapping guardado (emergency): {self.label_mapping_emergency_path}")
        elif "human_label" in df.columns and "emergency" not in df.columns:
            print("⚠️  Columna 'emergency' no encontrada; label_mapping_emergency no generado.")

        if "human_label" in df.columns and "alertable" in df.columns:
            no_alertable_mask = df["alertable"].apply(self._normalize_alertable_value) == False  # noqa: E712
            df_no_alertable = df[no_alertable_mask]
            if not df_no_alertable.empty:
                no_alertable_map = self._build_label_mapping(df_no_alertable, "human_label")
            else:
                no_alertable_map = {"target_col": "human_label", "label2idx": {}, "idx2label": {}, "num_classes": 0}
            self._save_pickle(no_alertable_map, self.label_mapping_human_no_alertable_path)
            print(f"🧩 Label mapping guardado (human no alertable): {self.label_mapping_human_no_alertable_path}")

    def _resolve_output_dir(self) -> Path:
        base = self.interim_dir if self.interim_dir is not None else Path(".")
        return base / self.config.output_subdir

    def _infer_column(self, df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
        lower_map = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        if required:
            raise ValueError(
                f"No pude inferir una columna válida. Candidatas: {list(candidates)}. "
                f"Columnas disponibles: {list(df.columns)}"
            )
        return None

    @staticmethod
    def _sanitize_folder_name(value: Any) -> str:
        text = str(value).strip()
        if not text:
            return "unknown"
        text = text.replace("\\", "_").replace("/", "_")
        text = re.sub(r'[<>:"|?*]', "_", text)
        text = re.sub(r"\s+", "_", text)
        return text

    def _load_csvs(self, csv_paths: Sequence[PathLike]) -> pd.DataFrame:
        frames = []
        for csv_path in csv_paths:
            path = Path(csv_path)
            if not path.exists():
                raise FileNotFoundError(f"No existe el CSV: {path}")
            frames.append(pd.read_csv(path))
        if not frames:
            raise ValueError("No se recibió ningún CSV.")
        return pd.concat(frames, ignore_index=True)

    def _prepare_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        return {
            "path": self._infer_column(df, self.config.file_path_candidates, required=True),
            "label": self._infer_column(df, self.config.label_candidates, required=True),
            "dataset_source": self._infer_column(df, self.config.dataset_source_candidates, required=True),
            "split": self._infer_column(df, self.config.split_candidates, required=False) or "split",
        }

    def _normalize_path_text(self, value: Any) -> str:
        s = str(value).strip().strip('"').strip("'")
        s = s.replace("\\", "/")
        s = re.sub(r"/+", "/", s)
        return s

    def _audio_path_from_row(self, row: pd.Series, path_col: str) -> Path:
        raw_value = row[path_col]
        normalized = self._normalize_path_text(raw_value)

        candidate = Path(normalized)

        if candidate.is_absolute() and candidate.exists():
            return candidate.resolve()

        if candidate.exists():
            return candidate.resolve()

        if self.raw_dir is not None:
            raw_dir = Path(self.raw_dir)

            p1 = raw_dir / candidate
            if p1.exists():
                return p1.resolve()

            candidate_parts = list(candidate.parts)
            raw_name = raw_dir.name.lower()

            for idx, part in enumerate(candidate_parts):
                if part.lower() == raw_name:
                    trimmed = Path(*candidate_parts[idx + 1 :])
                    p2 = raw_dir / trimmed
                    if p2.exists():
                        return p2.resolve()

            p3 = raw_dir / candidate.name
            if p3.exists():
                return p3.resolve()

            try:
                matches = list(raw_dir.rglob(candidate.name))
                if matches:
                    return matches[0].resolve()
            except Exception:
                pass

        return candidate

    def _get_resampler(self, orig_sr: int) -> Optional[torchaudio.transforms.Resample]:
        if orig_sr == self.config.sample_rate:
            return None
        if orig_sr not in self._resampler_cache:
            self._resampler_cache[orig_sr] = T.Resample(
                orig_freq=orig_sr,
                new_freq=self.config.sample_rate,
            ).to(self.device)
        return self._resampler_cache[orig_sr]

    def _get_window(self, n_fft: int) -> torch.Tensor:
        key = (self.device.type, str(self.device), n_fft)
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
        target_len = int(self.config.sample_rate * float(self.config.target_duration))
        current = waveform.shape[-1]
        if current > target_len:
            return waveform[..., :target_len]
        if current < target_len:
            return F.pad(waveform, (0, target_len - current))
        return waveform

    def _augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=waveform.device).item() < 0.55:
            waveform = waveform + torch.randn_like(waveform) * 0.003
        if torch.rand(1, device=waveform.device).item() < 0.45:
            gain = torch.empty(1, device=waveform.device).uniform_(0.85, 1.15)
            waveform = waveform * gain
        if waveform.shape[-1] > 1 and torch.rand(1, device=waveform.device).item() < 0.35:
            max_shift = max(1, int(0.08 * waveform.shape[-1]))
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,), device=waveform.device).item())
            waveform = torch.roll(waveform, shifts=shift, dims=-1)
        return waveform

    def _save_npy(self, tensor: torch.Tensor, path_without_suffix: Path) -> str:
        out = path_without_suffix.with_suffix(".npy")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, tensor.detach().cpu().numpy())
        return str(out)

    def _save_standardized_audio(self, waveform: torch.Tensor, path_without_suffix: Path) -> str:
        out = path_without_suffix.with_suffix(".wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        audio_np = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)
        sf.write(
            str(out),
            audio_np,
            self.config.sample_rate,
            subtype=self.config.output_audio_subtype,
            format=self.config.output_audio_format,
        )
        return str(out)

    @staticmethod
    def _subtype_to_bits_per_sample(subtype: Optional[str]) -> Optional[int]:
        if not subtype:
            return None
        subtype = str(subtype).upper()
        mapping = {
            "PCM_U8": 8,
            "PCM_16": 16,
            "PCM_24": 24,
            "PCM_32": 32,
            "FLOAT": 32,
            "DOUBLE": 64,
            "ULAW": 8,
            "ALAW": 8,
        }
        return mapping.get(subtype)

    @staticmethod
    def _estimate_bitrate_kbps(sample_rate: Optional[int], channels: Optional[int], bits_per_sample: Optional[int]) -> Optional[float]:
        if sample_rate is None or channels is None or bits_per_sample is None:
            return None
        return (float(sample_rate) * float(channels) * float(bits_per_sample)) / 1000.0

    def _read_audio_info(self, audio_path: Path) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "original_sample_rate": None,
            "original_channels": None,
            "original_frames": None,
            "original_duration_sec": None,
            "original_subtype": None,
            "original_bit_depth": None,
            "original_bitrate_kbps": None,
        }
        try:
            audio_info = sf.info(str(audio_path))
            info["original_sample_rate"] = int(audio_info.samplerate) if audio_info.samplerate else None
            info["original_channels"] = int(audio_info.channels) if audio_info.channels else None
            info["original_frames"] = int(audio_info.frames) if audio_info.frames else None
            if audio_info.samplerate and audio_info.frames:
                info["original_duration_sec"] = float(audio_info.frames / audio_info.samplerate)
            info["original_subtype"] = getattr(audio_info, "subtype", None)
            info["original_bit_depth"] = self._subtype_to_bits_per_sample(info["original_subtype"])
            info["original_bitrate_kbps"] = self._estimate_bitrate_kbps(
                info["original_sample_rate"],
                info["original_channels"],
                info["original_bit_depth"],
            )
        except Exception:
            pass
        return info

    def _output_base(self, dataset_source: str, human_label: str, audio_stem: str) -> Path:
        return (
            self.output_dir
            / self._sanitize_folder_name(dataset_source)
            / self._sanitize_folder_name(human_label)
            / self._sanitize_folder_name(audio_stem)
        )

    def _existing_outputs(self, base: Path) -> Dict[str, Path]:
        return {
            "audio": base / self.config.output_audio_filename,
            "waveform": base / "waveform.npy",
            "mel": base / "mel.npy",
            "mfcc": base / "mfcc.npy",
        }

    def _outputs_exist(self, base: Path) -> bool:
        outs = self._existing_outputs(base)
        checks = []
        if self.config.save_audio:
            checks.append(outs["audio"].exists())
        if self.config.save_waveform:
            checks.append(outs["waveform"].exists())
        if self.config.save_mel:
            checks.append(outs["mel"].exists())
        if self.config.save_mfcc:
            checks.append(outs["mfcc"].exists())
        return bool(checks) and all(checks)

    def process_single_audio(self, row: pd.Series, cols: Dict[str, str]) -> Optional[Dict[str, Any]]:
        try:
            audio_path = self._audio_path_from_row(row, cols["path"])
            if not audio_path.exists():
                print(f"⚠️ No existe el audio: {audio_path}")
                return None

            dataset_source = str(row[cols["dataset_source"]])
            human_label = str(row[cols["label"]])

            base_dir = self._output_base(dataset_source, human_label, audio_path.stem)
            outputs = self._existing_outputs(base_dir)

            result: Dict[str, Any] = row.to_dict()
            result["resolved_audio_path"] = str(audio_path)
            result["output_base_dir"] = str(base_dir)

            audio_info = self._read_audio_info(audio_path)
            result.update(audio_info)

            target_channels = 1
            target_sample_rate = self.config.sample_rate
            target_bit_depth = self._subtype_to_bits_per_sample(self.config.output_audio_subtype)
            target_bitrate_kbps = self._estimate_bitrate_kbps(target_sample_rate, target_channels, target_bit_depth)

            result.update({
                "final_sample_rate": target_sample_rate,
                "final_channels": target_channels,
                "final_bit_depth": target_bit_depth,
                "final_bitrate_kbps": target_bitrate_kbps,
                "standardized_audio_path": str(outputs["audio"]) if self.config.save_audio else None,
            })

            if self._outputs_exist(base_dir):
                result.update({
                    "processed": False,
                    "skipped_existing": True,
                    "waveform_path": str(outputs["waveform"]) if self.config.save_waveform else None,
                    "mel_path": str(outputs["mel"]) if self.config.save_mel else None,
                    "mfcc_path": str(outputs["mfcc"]) if self.config.save_mfcc else None,
                })
                result["sample_rate"] = target_sample_rate
                result["num_channels"] = target_channels
                result["bitrate_kbps"] = target_bitrate_kbps
                return result

            audio_np, sr = sf.read(str(audio_path))

            waveform = torch.tensor(audio_np, dtype=torch.float32)

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.transpose(0, 1)

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

            base_dir.mkdir(parents=True, exist_ok=True)

            if self.config.save_audio:
                result["standardized_audio_path"] = self._save_standardized_audio(waveform, outputs["audio"])
            if self.config.save_waveform:
                result["waveform_path"] = self._save_npy(waveform, outputs["waveform"])
            if self.config.save_mel:
                result["mel_path"] = self._save_npy(mel_db, outputs["mel"])
            if self.config.save_mfcc:
                result["mfcc_path"] = self._save_npy(mfcc, outputs["mfcc"])

            result.update({
                "processed": True,
                "skipped_existing": False,
                "sample_rate": self.config.sample_rate,
                "num_channels": 1,
                "bitrate_kbps": target_bitrate_kbps,
                "duration_sec": float(waveform.shape[-1] / self.config.sample_rate),
                "device_used": self.device.type,
            })
            result.update(scalar_features)
            return result

        except Exception as e:
            print(f"❌ Error procesando {row.get(cols['path'], 'unknown')}: {e}")
            return None

    def process_csvs(self, csv_paths: Optional[Sequence[PathLike]] = None) -> pd.DataFrame:
        paths = [Path(p) for p in (csv_paths or self.csv_paths)]
        if not paths:
            raise ValueError("Debes pasar una lista de CSVs.")

        incoming = self._load_csvs(paths)
        cols = self._prepare_columns(incoming)

        print(f"📁 Output dir: {self.output_dir}")
        print(f"🚀 Audios a evaluar: {len(incoming)}")
        print(f"Dispositivo: {self.device} | {self.device_name}")

        results: List[Dict[str, Any]] = []
        for _, row in tqdm(incoming.iterrows(), total=len(incoming), desc="Procesando audios"):
            processed = self.process_single_audio(row, cols)
            if processed is not None:
                results.append(processed)

        df_new = pd.DataFrame(results)

        if self.metadata_path.exists() and self.metadata_path.stat().st_size > 0:
            try:
                df_old = pd.read_csv(self.metadata_path)

                if (
                    cols["dataset_source"] in df_old.columns
                    and cols["dataset_source"] in df_new.columns
                    and not df_new.empty
                ):
                    dataset_sources = (
                        df_new[cols["dataset_source"]]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    df_old = df_old[
                        ~df_old[cols["dataset_source"]]
                        .astype(str)
                        .isin(dataset_sources)
                    ]

                df_final = pd.concat([df_old, df_new], ignore_index=True)

            except pd.errors.EmptyDataError:
                df_final = df_new
        else:
            df_final = df_new

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(self.metadata_path, index=False)
        print(f"\n✅ CSV guardado en: {self.metadata_path}")
        print(f"📊 Total de registros en histórico: {len(df_final)}")
        self._generate_label_mappings(df_final)
        return df_final

    def run(self, csv_paths: Optional[Sequence[PathLike]] = None) -> pd.DataFrame:
        return self.process_csvs(csv_paths)

    def summary(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "device_name": self.device_name,
            "output_dir": str(self.output_dir),
            "sample_rate": self.config.sample_rate,
            "target_duration": self.config.target_duration,
            "augment": self.config.augment,
            "save_audio": self.config.save_audio,
            "audio_subtype": self.config.output_audio_subtype,
        }

    def _compute_scalar_features(self, waveform: torch.Tensor) -> Dict[str, float]:
        eps = 1e-10

        x = waveform.squeeze(0)

        # RMS
        rms = torch.sqrt(torch.mean(x ** 2)).item()

        # Zero Crossing Rate
        zcr = (
            ((x[:-1] * x[1:]) < 0).float().mean().item()
            if x.numel() > 1
            else 0.0
        )

        # STFT
        window = self._get_window(self.config.n_fft)

        spec = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=window,
            center=True,
            return_complex=True,
        ).abs().squeeze(0)

        mag_mean = spec.mean(dim=-1)

        freqs = torch.linspace(
            0.0,
            float(self.config.sample_rate) / 2.0,
            mag_mean.shape[0],
            device=mag_mean.device,
        )

        mag_sum = mag_mean.sum().clamp_min(eps)

        # Spectral centroid
        centroid = (freqs * mag_mean).sum() / mag_sum

        # Spectral bandwidth
        bandwidth = torch.sqrt(
            (((freqs - centroid) ** 2) * mag_mean).sum() / mag_sum
        )

        # Spectral rolloff
        cumulative = torch.cumsum(mag_mean, dim=0)
        threshold = 0.85 * mag_sum

        rolloff_idx = torch.searchsorted(
            cumulative,
            threshold
        ).clamp(max=mag_mean.shape[0] - 1)

        rolloff = freqs[rolloff_idx]

        # Spectral flatness
        flatness = (
            torch.exp(torch.mean(torch.log(mag_mean.clamp_min(eps))))
            / mag_mean.mean().clamp_min(eps)
        )

        return {
            "rms": float(rms),
            "zcr": float(zcr),
            "spectral_centroid": float(centroid.item()),
            "spectral_bandwidth": float(bandwidth.item()),
            "spectral_rolloff": float(rolloff.item()),
            "spectral_flatness": float(flatness.item()),
        }
    def process_audio_file(
        self,
        audio_path: PathLike,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Procesa un único archivo de audio y devuelve (mel, mfcc) listos para inferencia.

        A diferencia de ``process_single_audio``, este método no necesita un CSV ni
        guarda nada en disco: sólo carga el audio, aplica el pipeline estándar
        (mono → resampleo → fix_length → normalización de pico → features) y
        devuelve los tensores.

        Parameters
        ----------
        audio_path : str | Path
            Ruta al archivo de audio (.wav, .mp3, …).

        Returns
        -------
        mel  : torch.Tensor  — shape [1, n_mels, T],  float32, en ``self.device``
        mfcc : torch.Tensor  — shape [1, n_mfcc, T],  float32, en ``self.device``
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio no encontrado: {path}")

        audio_np, sr = sf.read(str(path))

        waveform = torch.tensor(audio_np, dtype=torch.float32)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)

        waveform = self._ensure_mono(waveform).to(self.device)

        resampler = self._get_resampler(sr)
        if resampler is not None:
            waveform = resampler(waveform)

        waveform = self._fix_length(waveform)
        waveform = self._normalize_peak(waveform)

        with torch.inference_mode():
            mel  = self.amplitude_to_db(self.mel_transform(waveform))  # [1, n_mels, T]
            mfcc = self.mfcc_transform(waveform)                        # [1, n_mfcc, T]

        return mel.float(), mfcc.float()


def main():
    """
    Ejecución del preprocesado incremental.
    """
    csv_filenames = ["dataset_final_resampled.csv"]
    csv_paths = [RAW_DIR / f for f in csv_filenames]

    config = PreprocessConfig(
        sample_rate=44100,
        target_duration=5.0,
        augment=False,
        save_audio=True,
        output_audio_subtype="PCM_16",
        output_audio_format="WAV",
        n_mels=128,
        n_mfcc=13,
        save_waveform=False,
        save_mel=True,
        save_mfcc=True,
        use_multiprocessing=True,
    )

    pp = Preprocess(
        csv_paths=csv_paths,
        raw_dir=RAW_DIR,
        interim_dir=INTERIM_DIR,
        config=config,
    )

    df_processed = pp.run()

    print("\n--- Resumen del proceso ---")
    print(df_processed.head())
    print(f"Total de registros en el histórico: {len(df_processed)}")
    return df_processed


if __name__ == "__main__":
    main()
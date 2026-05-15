"""
Preprocesado incremental de audio.

Cambios respecto al script original:
- Acepta una lista de CSVs en lugar de un único CSV.
- Usa dataset_source + human_label para organizar la salida:
  processed_dataset/<dataset_source>/<human_label>/<audio_stem>/
- Si los espectrogramas ya existen, no recalcula ese audio.
- Si el CSV final ya existe, elimina las filas del dataset_source que se esté procesando
  y añade las nuevas filas sin borrar el resto del histórico.
- Guarda features en .npy.
"""

from __future__ import annotations
import soundfile as sf
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import pickle
from src.utils.config import RAW_DIR,INTERIM_DIR
warnings.filterwarnings("ignore")

PathLike = Union[str, Path]


@dataclass
class PreprocessConfig:
    sample_rate: int = 22050
    n_mels: int = 128
    n_mfcc: int = 40
    n_fft: int = 2048
    hop_length: int = 512
    target_duration: Optional[float] = 5.0
    normalize_peak: bool = True
    peak_target: float = 0.99
    augment: bool = False

    save_waveform: bool = True
    save_mel: bool = True
    save_mfcc: bool = True

   
    output_subdir: str = "processed_dataset"
    metadata_filename: str = "processed_metadata.csv"

    label_mapping_human_filename: str = "label_mapping_human_label.pkl"
    label_mapping_alertable_filename: str = "label_mapping_alertable.pkl"

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
        # Mapeo para human_label
        if "human_label" in df.columns:
            human_map = self._build_label_mapping(df, "human_label")
            self._save_pickle(human_map, self.label_mapping_human_path)
            print(f"🧩 Label mapping guardado: {self.label_mapping_human_path}")

        # Mapeo para alertable
        if "alertable" in df.columns:
            alert_map = self._build_label_mapping(df, "alertable")
            self._save_pickle(alert_map, self.label_mapping_alertable_path)
            print(f"🧩 Label mapping guardado: {self.label_mapping_alertable_path}")
    # ---------------------------
    # Resolución de rutas/columnas
    # ---------------------------
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

    # ---------------------------
    # Audio helpers
    # ---------------------------
    def _audio_path_from_row(self, row: pd.Series, path_col: str) -> Path:
        raw_value = str(row[path_col])
        candidate = Path(raw_value)

        if candidate.is_absolute() and candidate.exists():
            return candidate

        if self.raw_dir is not None:
            joined = self.raw_dir / candidate
            if joined.exists():
                return joined

            joined_name = self.raw_dir / candidate.name
            if joined_name.exists():
                return joined_name

        if candidate.exists():
            return candidate

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

    def _compute_scalar_features(self, waveform: torch.Tensor) -> Dict[str, float]:
        eps = 1e-10
        x = waveform.squeeze(0)

        rms = torch.sqrt(torch.mean(x ** 2)).item()
        zcr = ((x[:-1] * x[1:]) < 0).float().mean().item() if x.numel() > 1 else 0.0

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
        freqs = torch.linspace(0.0, float(self.config.sample_rate) / 2.0, mag_mean.shape[0], device=mag_mean.device)

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

    def _output_base(self, dataset_source: str, human_label: str, audio_stem: str) -> Path:
        return (
            self.output_dir
            / self._sanitize_folder_name(dataset_source)
            / self._sanitize_folder_name(human_label)
            / self._sanitize_folder_name(audio_stem)
        )

    def _existing_outputs(self, base: Path) -> Dict[str, Path]:
        return {
            "waveform": base / "waveform.npy",
            "mel": base / "mel.npy",
            "mfcc": base / "mfcc.npy",
        }

    def _outputs_exist(self, base: Path) -> bool:
        outs = self._existing_outputs(base)
        checks = []
        if self.config.save_waveform:
            checks.append(outs["waveform"].exists())
        if self.config.save_mel:
            checks.append(outs["mel"].exists())
        if self.config.save_mfcc:
            checks.append(outs["mfcc"].exists())
        return bool(checks) and all(checks)

    # ---------------------------
    # Procesamiento
    # ---------------------------
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

            if self._outputs_exist(base_dir):
                result.update({
                    "processed": False,
                    "skipped_existing": True,
                    "waveform_path": str(outputs["waveform"]) if self.config.save_waveform else None,
                    "mel_path": str(outputs["mel"]) if self.config.save_mel else None,
                    "mfcc_path": str(outputs["mfcc"]) if self.config.save_mfcc else None,
                })
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

        # Actualiza el CSV final sin perder el histórico de otros dataset_source.
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

        # 🔥 GUARDAR EL CSV FINAL (LÍNEA CRÍTICA QUE FALTABA)
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
        }



def main():
    """
    Ejecución del preprocesado incremental.
    """
    # 1. Definir rutas base
    
  
    # Definimos los nombres de los archivos
    csv_filenames = ["UrbanSound8k.csv","audioset.csv","ESC50.csv","zenodo.csv","Guns_DS.csv","VOICe.csv"]


    # Mapeamos para agregar el raw_dir usando una list comprehension
    csv_paths = [RAW_DIR / f for f in csv_filenames]
    # 2. Configurar los parámetros (usando solo campos existentes en PreprocessConfig)
    # Nota: Se eliminó 'batch_size' ya que no existe en tu dataclass.
    config = PreprocessConfig(
        sample_rate=44100,
        n_mels=128,
        n_mfcc=13,
        target_duration=None,
        augment=False,
        use_multiprocessing=True  # Se define aquí, no en el run()
    )

    # 3. Instanciar la clase Preprocess
    # El parámetro correcto es 'csv_paths' (en plural)
    pp = Preprocess(
        csv_paths=csv_paths,
        raw_dir=RAW_DIR,
        interim_dir=INTERIM_DIR,
        config=config
    )

    # 4. Ejecutar el proceso
    # El método run() en tu clase solo acepta csv_paths opcionales.
    # Los demás parámetros (audio_dir, output_dir, etc.) ya se pasaron en el __init__.
    df_processed = pp.run()

    # 5. Resultados
    print("\n--- Resumen del proceso ---")
    print(df_processed.head())
    print(f"Total de registros en el histórico: {len(df_processed)}")
    

if __name__ == "__main__":
    main()
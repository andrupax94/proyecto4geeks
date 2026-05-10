import wave
from pathlib import Path
from datetime import datetime

import pandas as pd
import soundfile as sf


class MetadataEX:
    def __init__(self, csv_path, dataset_name, folder):
        self.csv_path = Path(csv_path)
        self.dataset_name = dataset_name
        self.folder = Path(folder)

    @staticmethod
    def _channels_label(n_channels: int):
        if n_channels == 1:
            return "Mono"
        if n_channels == 2:
            return "Stereo"
        return f"{n_channels} canales"

    @staticmethod
    def _bitdepth_from_subtype(subtype: str):
        subtype = (subtype or "").upper()

        mapping = {
            "PCM_S8": 8,
            "PCM_U8": 8,
            "PCM_16": 16,
            "PCM_24": 24,
            "PCM_32": 32,
            "FLOAT": 32,
            "DOUBLE": 64,
        }

        for key, value in mapping.items():
            if key in subtype:
                return value

        return None

    def _metadata_with_wave(self, file_path: Path):
        """
        Lee metadatos de WAV usando wave.
        Muy fiable para .wav PCM.
        """
        with wave.open(str(file_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            sampwidth_bytes = wf.getsampwidth()

            duration = round(n_frames / sample_rate, 3) if sample_rate else None
            bit_depth = sampwidth_bytes * 8 if sampwidth_bytes else None
            bit_velocity = (
                sample_rate * n_channels * bit_depth
                if sample_rate and n_channels and bit_depth
                else None
            )

            return {
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": self._channels_label(n_channels),
                "bit_depth": bit_depth,
                "bit_velocity": bit_velocity,
                "audio_format": "wav",
            }

    def _metadata_with_soundfile(self, file_path: Path):
        """
        Fallback para WAV u otros formatos usando soundfile.
        """
        info = sf.info(str(file_path))

        duration = round(info.frames / info.samplerate, 3) if info.samplerate else None
        sample_rate = info.samplerate
        channels_num = info.channels
        bit_depth = self._bitdepth_from_subtype(info.subtype)

        bit_velocity = (
            sample_rate * channels_num * bit_depth
            if sample_rate and channels_num and bit_depth
            else None
        )

        return {
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": self._channels_label(channels_num),
            "bit_depth": bit_depth,
            "bit_velocity": bit_velocity,
            "audio_format": info.format.lower() if info.format else None,
        }

    def generate_metadata_audio(self):
        """
        Lee el CSV, extrae metadatos de los audios y sobrescribe el archivo.

        Añade:
        - file_exists
        - audio_format
        - duration
        - sample_rate
        - channels
        - bit_depth
        - bit_velocity
        - size
        - date_modification
        - dataset_source
        """
        if not self.csv_path.exists():
            print(f"Error: No se encontró el CSV en {self.csv_path}")
            return

        if not self.folder.exists():
            print(f"Error: No se encontró la carpeta de audios en {self.folder}")
            return

        df = pd.read_csv(self.csv_path)

        if "path" not in df.columns:
            print("Error: El CSV no contiene la columna obligatoria 'path'")
            return

        file_exists_list = []
        audio_format_list = []
        durations = []
        sample_rates = []
        channels_list = []
        bit_depths = []
        bit_velocities = []
        sizes_kb = []
        date_modifications = []

        print(f"Procesando metadatos para {len(df)} archivos...")

        for _, row in df.iterrows():
            ruta_audio = self.folder / str(row["path"])

            if not ruta_audio.exists():
                file_exists_list.append(False)
                audio_format_list.append(None)
                durations.append(None)
                sample_rates.append(None)
                channels_list.append(None)
                bit_depths.append(None)
                bit_velocities.append(None)
                sizes_kb.append(None)
                date_modifications.append(None)
                continue

            file_exists_list.append(True)

            try:
                if ruta_audio.suffix.lower() == ".wav":
                    try:
                        meta = self._metadata_with_wave(ruta_audio)
                    except Exception:
                        meta = self._metadata_with_soundfile(ruta_audio)
                else:
                    meta = self._metadata_with_soundfile(ruta_audio)

                stat = ruta_audio.stat()
                size_kb = round(stat.st_size / 1024, 2)
                date_modification = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

                audio_format_list.append(meta.get("audio_format"))
                durations.append(meta.get("duration"))
                sample_rates.append(meta.get("sample_rate"))
                channels_list.append(meta.get("channels"))
                bit_depths.append(meta.get("bit_depth"))
                bit_velocities.append(meta.get("bit_velocity"))
                sizes_kb.append(size_kb)
                date_modifications.append(date_modification)

            except Exception as e:
                print(f"Error leyendo {ruta_audio}: {e}")
                audio_format_list.append(None)
                durations.append(None)
                sample_rates.append(None)
                channels_list.append(None)
                bit_depths.append(None)
                bit_velocities.append(None)
                sizes_kb.append(None)
                date_modifications.append(None)

        df["file_exists"] = file_exists_list
        df["audio_format"] = audio_format_list
        df["duration"] = durations
        df["sample_rate"] = sample_rates
        df["channels"] = channels_list
        df["bit_depth"] = bit_depths
        df["bit_velocity"] = bit_velocities
        df["size"] = sizes_kb
        df["date_modification"] = date_modifications
        df["dataset_source"] = self.dataset_name

        try:
            df.to_csv(self.csv_path, index=False, quoting=1)
            print("--- CSV actualizado con éxito ---")
            print(f"Ruta: {self.csv_path.resolve()}")
        except PermissionError:
            print("Error: No se pudo sobrescribir el CSV. Asegúrate de que no esté abierto en Excel.")
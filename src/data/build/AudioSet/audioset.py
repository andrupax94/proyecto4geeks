
import pandas as pd
import os
from pathlib import Path

from src.data.build.metadata import MetadataEX

from src.utils.config import RAW_DIR, BUILD_DIR
# --- EJEMPLO DE CONFIGURACIÓN ---

import soundfile as sf
import librosa
import os
from pathlib import Path
from datetime import datetime
import sys
# =====================================
# IMPORT CONFIG
# =====================================

sys.path.append(str(Path(__file__).resolve().parents[4]))

from src.utils.config import (
    AUDIOSET_SONIDOS_DIR,
    SINONIMOS_V3_PATH,
    CANONICAL_CLASSES_PATH,
    AUDIOSET_DATASET_PATH,
)

class AudioSet:
    AudioSet.build_dataset()
    def obtener_metadata(audio_path):
        try:
            info = sf.info(audio_path)
            duration = info.duration
            sample_rate = info.samplerate
            channels = "Mono" if info.channels == 1 else "Stereo"
            bit_depth = info.subtype

        except Exception:
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            sample_rate = sr
            channels = "Mono"
            bit_depth = "Unknown"

        size_kb = os.path.getsize(audio_path) / 1024

        modification_date = datetime.fromtimestamp(
            os.path.getmtime(audio_path)
        )

        audio_format = audio_path.suffix.replace(".", "").lower()

        return (
            duration,
            sample_rate,
            channels,
            bit_depth,
            size_kb,
            modification_date,
            audio_format,
        )


    def obtener_clase(folder):

        folder = folder.lower()

        canonical = AudioSet.synonym_to_canonical.get(folder, folder)

        human_label = canonical

        info = AudioSet.canonical_info.get(canonical, {})

        return (
            canonical,
            human_label,
            info.get("environment", "unknown"),
            info.get("alertable", False),
            info.get("emergency", False),
        )

    # =====================================
    # BUILD DATASET
    # =====================================
    def build_dataset():
        dataset = []

        print("\nProcesando audios...\n")

        for folder in os.listdir(AUDIOSET_SONIDOS_DIR):

            folder_path = AUDIOSET_SONIDOS_DIR / folder

            if not folder_path.is_dir():
                continue

            print(f"Carpeta: {folder}")

            for audio_file in folder_path.glob("*.*"):

                if audio_file.suffix.lower() not in [
                    ".wav", ".mp3", ".flac", ".ogg"
                ]:
                    continue

                (
                    duration,
                    sample_rate,
                    channels,
                    bit_depth,
                    size_kb,
                    modification_date,
                    audio_format,
                ) = AudioSet.obtener_metadata(audio_file)

                (
                    canonical,
                    human_label,
                    env,
                    alertable,
                    emergency,
                ) = AudioSet.obtener_clase(folder)

                fila = {
                    "audio": audio_file.name,
                    "label": canonical,
                    "labels": [canonical],
                    "human_label": human_label,
                    "human_labels": [human_label],
                    "path": str(audio_file.relative_to(AUDIOSET_SONIDOS_DIR.parent)),
                    "dataset_source": "AudioSet",
                    "split": "train",
                    "env": env,
                    "alertable": alertable,
                    "emergency": emergency,
                    "file_exists": True,
                    "audio_format": audio_format,
                    "duration": round(duration, 3),
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "bit_depth": bit_depth,
                    "bit_velocity": "unknown",
                    "size": round(size_kb, 2),
                    "date_modification": modification_date,
                }

                dataset.append(fila)

        # =====================================
        # SAVE DATASET
        # =====================================

        df = pd.DataFrame(dataset)

        # ORDEN EXACTO DE COLUMNAS
        df = df[
            [
                "audio",
                "label",
                "labels",
                "human_label",
                "human_labels",
                "path",
                "dataset_source",
                "split",
                "env",
                "alertable",
                "emergency",
                "file_exists",
                "audio_format",
                "duration",
                "sample_rate",
                "channels",
                "bit_depth",
                "bit_velocity",
                "size",
                "date_modification",
            ]
        ]

        AUDIOSET_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(AUDIOSET_DATASET_PATH, index=False)

        print("\n✅ DATASET AUDIOSET CREADO")
        print(f"Audios totales: {len(df)}")
        print(f"Guardado en: {AUDIOSET_DATASET_PATH}")


if __name__ == "__main__":
    script_dir =  BUILD_DIR / "audioset"
    file_list = {"audioset.csv":["audioset.csv",0.8]}

    print("Cargando sinonimos...")
    sinonimos = pd.read_csv(SINONIMOS_V3_PATH)

    print("Cargando canonical clases...")
    canonical_df = pd.read_csv(CANONICAL_CLASSES_PATH)

    sinonimos["canonical"] = sinonimos["canonical"].str.lower()
    sinonimos["synonym"] = sinonimos["synonym"].str.lower()
    canonical_df["canonical"] = canonical_df["canonical"].str.lower()

    synonym_to_canonical = dict(
        zip(sinonimos["synonym"], sinonimos["canonical"])
    )

    canonical_info = canonical_df.set_index("canonical").to_dict("index")
    AudioSet.build_dataset()
    nombre_salida = os.path.join(RAW_DIR, "audioset.csv")
    metadata = MetadataEX(
        csv_path=nombre_salida,
        dataset_name="audioset",
        folder=RAW_DIR
    )
    metadata.generate_metadata_audio()
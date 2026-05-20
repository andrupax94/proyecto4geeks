import shutil
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import kurtosis
from tqdm import tqdm
from src.utils.config import *

# ============================================
# CONFIG
# ============================================
INPUT_DIR = RAW_DIR / "youtube" / "gunshot"
CSV_OUTPUT = BUILD_DIR /"youtube" /"youtube_gunshot" / "gunshot.csv"

SR = 22050

# CONFIG AGRESIVA
MIN_PEAK = 0.45
MIN_CENTROID = 2200
MAX_DURATION = 1.2
MIN_KURTOSIS = 12.0
MIN_ATTACK_RATIO = 5.0

DELETE_NON_GUNSHOT = True  # Eliminar en lugar de mover

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# ============================================
# FEATURES
# ============================================
def extract_features(path: Path):
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)

        if len(y) == 0:
            return None

        duration = librosa.get_duration(y=y, sr=sr)

        peak = float(np.max(np.abs(y)))
        rms = float(np.sqrt(np.mean(y**2)))

        centroid = float(
            librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        )

        zcr = float(
            librosa.feature.zero_crossing_rate(y).mean()
        )

        # Kurtosis ayuda mucho para detectar impulsos
        k = float(kurtosis(y))

        # Energía inicial vs resto
        split = max(1, int(len(y) * 0.1))

        attack_energy = np.mean(np.abs(y[:split])) + 1e-8
        rest_energy = np.mean(np.abs(y[split:])) + 1e-8

        attack_ratio = float(attack_energy / rest_energy)

        return {
            "duration": duration,
            "peak": peak,
            "rms": rms,
            "centroid": centroid,
            "zcr": zcr,
            "kurtosis": k,
            "attack_ratio": attack_ratio,
        }

    except Exception as e:
        print(f"ERROR {path}: {e}")
        return None


# ============================================
# HEURÍSTICA GUNSHOT
# ============================================
def is_probable_gunshot(features):
    if features is None:
        return False

    score = 0

    # Disparos suelen tener transientes fuertes
    if features["peak"] >= MIN_PEAK:
        score += 1

    # Centroides relativamente altos
    if features["centroid"] >= MIN_CENTROID:
        score += 1

    # Duración corta
    if features["duration"] <= MAX_DURATION:
        score += 1

    # Impulsividad
    if features["kurtosis"] >= MIN_KURTOSIS:
        score += 1

    # Ataque fuerte al inicio
    if features["attack_ratio"] >= MIN_ATTACK_RATIO:
        score += 1

    return score >= 3


# ============================================
# MAIN
# ============================================
def main():
    files = [
        p for p in INPUT_DIR.rglob("*")
        if p.suffix.lower() in AUDIO_EXTS
    ]

    print(f"Audios encontrados: {len(files)}")

    rows = []

    kept = 0
    removed = 0

    for path in tqdm(files):
        feats = extract_features(path)

        prediction = is_probable_gunshot(feats)

        row = {
            "file": str(path),
            "probable_gunshot": prediction,
        }

        if feats is not None:
            row.update(feats)

        rows.append(row)

        if prediction:
            kept += 1
        else:
            removed += 1

            if DELETE_NON_GUNSHOT:
                try:
                    path.unlink()  # Eliminar archivo
                    print(f"Eliminado: {path.name}")
                except Exception as e:
                    print(f"No se pudo eliminar {path}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUTPUT, index=False)

    print("\n========== RESUMEN ==========")
    print(f"Probables disparos: {kept}")
    print(f"No disparos (eliminados): {removed}")
    print(f"CSV: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
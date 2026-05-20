from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from tqdm import tqdm

from src.utils.config import *

# ============================================
# CONFIG
# ============================================
INPUT_DIR = RAW_DIR / "youtube" / "car_crash"
CSV_OUTPUT = BUILD_DIR / "youtube" / "youtube_car_crashs" / "car_crashs.csv"

SR = 22050

# Ajustes pensados para car crashes:
# - impacto fuerte
# - broadband
# - posible cola metálica / vidrio
# - videos con ruido de fondo
MIN_PEAK = 0.18
MIN_CENTROID = 900
MAX_CENTROID = 7500
MIN_ROLLOFF = 1800
MAX_DURATION = 8.0

MIN_KURTOSIS = 2.0
MIN_ATTACK_RATIO = 1.15
MIN_FLATNESS = 0.04
MIN_CREST_FACTOR = 3.0
MIN_RMS = 0.008

DELETE_NON_CAR_CRASH = True
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".mp4", ".webm"}


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
        rms = float(np.sqrt(np.mean(y**2)) + 1e-12)

        centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean())
        flatness = float(librosa.feature.spectral_flatness(y=y).mean())
        zcr = float(librosa.feature.zero_crossing_rate(y).mean())

        # Kurtosis para impulsos: en crash suele haber picos bruscos,
        # pero no tan “limpios” como en una explosión pura.
        k = float(kurtosis(y, fisher=True, bias=False))

        split = max(1, int(len(y) * 0.12))
        attack_energy = np.mean(np.abs(y[:split])) + 1e-8
        rest_energy = np.mean(np.abs(y[split:])) + 1e-8
        attack_ratio = float(attack_energy / rest_energy)

        crest_factor = float(peak / (rms + 1e-12))

        return {
            "duration": duration,
            "peak": peak,
            "rms": rms,
            "centroid": centroid,
            "rolloff": rolloff,
            "flatness": flatness,
            "zcr": zcr,
            "kurtosis": k,
            "attack_ratio": attack_ratio,
            "crest_factor": crest_factor,
        }

    except Exception as e:
        print(f"ERROR {path}: {e}")
        return None


# ============================================
# HEURÍSTICA CAR CRASH
# ============================================
def is_probable_car_crash(features):
    if features is None:
        return False

    score = 0

    # Impacto inicial fuerte
    if features["peak"] >= MIN_PEAK:
        score += 1

    # Energía suficiente
    if features["rms"] >= MIN_RMS:
        score += 1

    # Broadband razonable
    if features["centroid"] >= MIN_CENTROID:
        score += 1
    if features["centroid"] <= MAX_CENTROID:
        score += 1

    # Rolloff útil para golpes/vidrio/metal
    if features["rolloff"] >= MIN_ROLLOFF:
        score += 1

    # Video real con crash suele tener contenido más "rugoso"
    if features["flatness"] >= MIN_FLATNESS:
        score += 1

    # Impulsividad moderada
    if features["kurtosis"] >= MIN_KURTOSIS:
        score += 1

    # Ataque más marcado al inicio
    if features["attack_ratio"] >= MIN_ATTACK_RATIO:
        score += 1

    # Picos pronunciados respecto al RMS
    if features["crest_factor"] >= MIN_CREST_FACTOR:
        score += 1

    # Duración no demasiado larga
    if features["duration"] <= MAX_DURATION:
        score += 1

    return score >= 6


# ============================================
# MAIN
# ============================================
def main():
    files = [p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in AUDIO_EXTS]

    print(f"Audios encontrados: {len(files)}")

    rows = []
    kept = 0
    removed = 0

    for path in tqdm(files):
        feats = extract_features(path)
        prediction = is_probable_car_crash(feats)

        row = {
            "file": str(path),
            "probable_car_crash": prediction,
        }

        if feats is not None:
            row.update(feats)

        rows.append(row)

        if prediction:
            kept += 1
        else:
            removed += 1
            if DELETE_NON_CAR_CRASH:
                try:
                    path.unlink()
                    print(f"Eliminado: {path.name}")
                except Exception as e:
                    print(f"No se pudo eliminar {path}: {e}")

    df = pd.DataFrame(rows)
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUTPUT, index=False)

    print("\n========== RESUMEN ==========")
    print(f"Probables car_crash: {kept}")
    print(f"No car_crash (eliminadas): {removed}")
    print(f"CSV: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
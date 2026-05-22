"""
Regenera crying.csv desde los WAVs procesados, sin Excel.
Ejecutar en la carpeta donde está crying_processed/
"""
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kurtosis as scipy_kurtosis
from tqdm import tqdm

PROCESSED_DIR = Path("crying_processed") # carpeta con los WAVs procesados
OUTPUT_CSV    = Path("crying.csv")

def extract_features(clip_path):
    try:
        y, sr = librosa.load(clip_path, sr=None, mono=True)
        if len(y) == 0:
            return None
        duration     = float(len(y) / sr)
        peak         = float(np.max(np.abs(y)))
        rms          = float(np.sqrt(np.mean(y ** 2)))
        centroid     = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        zcr          = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        kurt         = float(scipy_kurtosis(y))
        split        = max(1, int(0.2 * len(y)))
        energy_total = np.mean(y ** 2) + 1e-10
        attack_ratio = float(np.mean(y[:split] ** 2) / energy_total)
        return {
            "file":             str(clip_path),
            "probable_crying":  True,
            "duration":         duration,
            "peak":             peak,
            "rms":              rms,
            "centroid":         centroid,
            "zcr":              zcr,
            "kurtosis":         kurt,
            "attack_ratio":     attack_ratio,
        }
    except Exception as e:
        print(f"  Error en {clip_path.name}: {e}")
        return None

clips = list(PROCESSED_DIR.glob("*.wav"))
print(f"Encontrados {len(clips)} clips en {PROCESSED_DIR}")

rows = []
for clip in tqdm(clips, desc="Extrayendo features"):
    row = extract_features(clip)
    if row:
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)   # CSV limpio, sin Excel
print(f"\nGuardado: {OUTPUT_CSV}  ({len(df)} filas)")
print(df[['peak','rms','centroid','zcr','kurtosis','attack_ratio']].describe().round(4))

"""
Pipeline completo:
1. Descarga audio de YouTube (primeros 30 min)
2. Segmenta en clips de 5s y filtra por energía
3. Extrae features acústicas (igual que los otros CSVs)
4. Genera crying.csv
5. Une todos los CSVs en unified_dataset.csv
"""

import subprocess
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kurtosis as scipy_kurtosis
from tqdm import tqdm
import sys

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
URLS_FILE       = Path("./crying_urls.txt")
RAW_DIR         = Path("./crying_raw")
PROCESSED_DIR   = Path("./crying_processed")
OUTPUT_CSV      = Path("./crying.csv")
UNIFIED_CSV     = Path("./unified_dataset.csv")

CLIP_DURATION       = 5     # segundos
ENERGY_THRESHOLD    = 0.001 # igual que el script (ajustable)
MAX_URLS            = None  # None = todos; ponemos un número para probar rápido

EXISTING_CSVS = {
    "fire":       Path("./fire.csv"),
    "gunshot":    Path("./gunshot.csv"),
    "car_crash":  Path("./car_crash.csv"),
    "explosion":  Path("./explosions.csv"),
}

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: DESCARGA
# ─────────────────────────────────────────────
def download_audio(urls_file: Path, out_dir: Path, max_urls=None):
    with open(urls_file) as f:
        urls = [u.strip() for u in f if u.strip()]
    if max_urls:
        urls = urls[:max_urls]

    print(f"\n{'='*50}")
    print(f"STEP 1: Descargando {len(urls)} URLs → {out_dir}")
    print(f"{'='*50}")

    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] {url}")
        cmd = [
            "yt-dlp", "-x", "--audio-format", "wav",
            "--download-sections", "*00:00:00-00:30:00",
            "--no-playlist",
            "-o", str(out_dir / "%(id)s.%(ext)s"),
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr.strip()[-200:]}")
        else:
            print(f"OK")

    wavs = list(out_dir.glob("*.wav"))
    print(f"\n→ {len(wavs)} archivos WAV descargados")
    return wavs

# ─────────────────────────────────────────────
# STEP 2: SEGMENTAR + FILTRAR POR ENERGÍA
# ─────────────────────────────────────────────
def segment_audio(raw_dir: Path, proc_dir: Path, clip_dur=5, energy_thr=0.001):
    files = list(raw_dir.glob("*.wav"))
    print(f"\n{'='*50}")
    print(f"STEP 2: Segmentando {len(files)} archivos → clips de {clip_dur}s")
    print(f"{'='*50}")

    total_clips = 0
    for file in tqdm(files, desc="Segmentando"):
        try:
            y, sr = librosa.load(file, sr=None, mono=True)
        except Exception as e:
            print(f"No se pudo cargar {file.name}: {e}")
            continue

        samples_per_clip = sr * clip_dur
        n_clips = len(y) // samples_per_clip

        for i in range(n_clips):
            clip = y[i * samples_per_clip:(i + 1) * samples_per_clip]
            energy = float(np.mean(clip ** 2))
            if energy < energy_thr:
                continue
            out_path = proc_dir / f"{file.stem}_seg{i}.wav"
            sf.write(out_path, clip, sr)
            total_clips += 1

    print(f"→ {total_clips} clips guardados en {proc_dir}")
    return list(proc_dir.glob("*.wav"))

# ─────────────────────────────────────────────
# STEP 3: EXTRAER FEATURES (mismo esquema que los otros CSVs)
# ─────────────────────────────────────────────
def extract_features(clip_path: Path) -> dict | None:
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

        # attack_ratio: energía primer 20% vs total
        split        = max(1, int(0.2 * len(y)))
        energy_total = np.mean(y ** 2) + 1e-10
        attack_ratio = float(np.mean(y[:split] ** 2) / energy_total)

        return {
            "file":         str(clip_path),
            "probable_crying": True,   # todos son positivos (descargados de crying videos)
            "duration":     duration,
            "peak":         peak,
            "rms":          rms,
            "centroid":     centroid,
            "zcr":          zcr,
            "kurtosis":     kurt,
            "attack_ratio": attack_ratio,
        }
    except Exception as e:
        print(f"Error en {clip_path.name}: {e}")
        return None

def build_crying_csv(clips: list[Path], output: Path) -> pd.DataFrame:
    print(f"\n{'='*50}")
    print(f"STEP 3: Extrayendo features de {len(clips)} clips")
    print(f"{'='*50}")

    rows = []
    for clip in tqdm(clips, desc="Extrayendo features"):
        row = extract_features(clip)
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"→ crying.csv guardado: {len(df)} filas → {output}")
    return df

# ─────────────────────────────────────────────
# STEP 4: UNIR TODOS LOS CSVs
# ─────────────────────────────────────────────
def unify_datasets(existing: dict, crying_csv: Path, output: Path) -> pd.DataFrame:
    print(f"\n{'='*50}")
    print(f"STEP 4: Unificando todos los datasets")
    print(f"{'='*50}")

    dfs = []

    for class_name, csv_path in existing.items():
        df = pd.read_csv(csv_path)
        # Estandarizar: renombrar columna probable_X → label_positive
        prob_col = [c for c in df.columns if c.startswith("probable_")][0]
        df = df.rename(columns={prob_col: "label_positive"})
        df["class"] = class_name
        # Quedarse solo con columnas comunes + extras si existen
        dfs.append(df)
        print(f"  {class_name:12s}: {len(df):5d} filas | positivos: {df['label_positive'].sum()}")

    # Crying
    df_cry = pd.read_csv(crying_csv)
    df_cry = df_cry.rename(columns={"probable_crying": "label_positive"})
    df_cry["class"] = "crying"
    dfs.append(df_cry)
    print(f"  {'crying':12s}: {len(df_cry):5d} filas | positivos: {df_cry['label_positive'].sum()}")

    unified = pd.concat(dfs, ignore_index=True)

    # Reordenar columnas: class y label_positive al frente
    front = ["file", "class", "label_positive"]
    rest  = [c for c in unified.columns if c not in front]
    unified = unified[front + rest]

    unified.to_csv(output, index=False)
    print(f"\n→ Dataset unificado: {len(unified)} filas, {len(unified.columns)} columnas → {output}")
    print(f"  Clases: {unified['class'].value_counts().to_dict()}")
    return unified

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Paso 1: Descargar
    download_audio(URLS_FILE, RAW_DIR, max_urls=MAX_URLS)

    # Paso 2: Segmentar
    clips = segment_audio(RAW_DIR, PROCESSED_DIR, CLIP_DURATION, ENERGY_THRESHOLD)

    if not clips:
        print("\nNo se generaron clips. Revisa las URLs o el umbral de energía.")
        sys.exit(1)

    # Paso 3: Extraer features → crying.csv
    df_crying = build_crying_csv(clips, OUTPUT_CSV)

    # Paso 4: Unificar todo
    df_unified = unify_datasets(EXISTING_CSVS, OUTPUT_CSV, UNIFIED_CSV)

    print("\nPipeline completo.")
    print(f"   crying.csv      → {OUTPUT_CSV}")
    print(f"   unified_dataset → {UNIFIED_CSV}")

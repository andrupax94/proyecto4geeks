"""
Método alternativo para detectar duplicados:
En lugar de MFCCs, usa un "hash" basado en:
- Peak (amplitud máxima)
- Duration (duración exacta)
- RMS (energía media)
- Spectral shape (primeros 4 MFCCs normalizados)

Más específico para detectar "el MISMO clip repetido" que "sonidos parecidos"
"""
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

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
INPUT_DIR = RAW_DIR / "youtube" / "fire"
CSV_OUTPUT = BUILD_DIR /"youtube" /"youtube_fire" / "fire.csv"

SR = 22050

MIN_PEAK = 0.20
MIN_CENTROID = 1000
MAX_DURATION = 5.0
MIN_KURTOSIS = 2.0
MIN_RMS = 0.01

DELETE_NON_FIRE = False
DELETE_DUPLICATES = True

# Tolerancia para considerarse "el mismo" clip
# Estos valores determinan qué tan exacto debe ser el match
PEAK_TOLERANCE = 0.02          # Peak dentro de ±0.02 (2%)
DURATION_TOLERANCE = 0.05      # Duración dentro de ±0.05 segundos
RMS_TOLERANCE = 0.002          # RMS dentro de ±0.002
SPECTRAL_SIM_THRESHOLD = 0.96  # Similitud espectral >= 96%

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
N_MFCC_SHORT = 4  # Solo primeros 4 coeficientes para "forma" espectral


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
        centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        zcr = float(librosa.feature.zero_crossing_rate(y).mean())
        k = float(kurtosis(y))

        split = max(1, int(len(y) * 0.2))
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


def is_probable_fire(features):
    if features is None:
        return False

    score = 0
    if features["peak"] >= MIN_PEAK:
        score += 1
    if features["centroid"] >= MIN_CENTROID:
        score += 1
    if features["duration"] <= MAX_DURATION:
        score += 1
    if features["kurtosis"] >= MIN_KURTOSIS:
        score += 1
    if features["rms"] >= MIN_RMS:
        score += 1

    return score >= 3


# ============================================
# DETECCIÓN DE DUPLICADOS (MÉTODO ALTERNATIVO)
# ============================================
def extract_spectral_signature(path: Path) -> np.ndarray:
    """Extrae firma espectral (primeros 4 MFCCs)"""
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
        if len(y) == 0:
            return None
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC_SHORT)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_normalized = mfcc_mean / (np.linalg.norm(mfcc_mean) + 1e-8)
        
        return mfcc_normalized
    except Exception as e:
        return None


def create_fingerprint(path: Path, features: dict) -> Tuple[dict, np.ndarray]:
    """Crea un fingerprint único para cada archivo"""
    spectral_sig = extract_spectral_signature(path)
    
    fingerprint = {
        "peak": round(features["peak"], 3),
        "duration": round(features["duration"], 2),
        "rms": round(features["rms"], 4),
    }
    
    return fingerprint, spectral_sig


def are_duplicates(fp1: dict, sig1: np.ndarray, 
                   fp2: dict, sig2: np.ndarray) -> bool:
    """Determina si dos archivos son duplicados basado en múltiples criterios"""
    
    # Criterio 1: Peak similar
    if abs(fp1["peak"] - fp2["peak"]) > PEAK_TOLERANCE:
        return False
    
    # Criterio 2: Duración casi idéntica
    if abs(fp1["duration"] - fp2["duration"]) > DURATION_TOLERANCE:
        return False
    
    # Criterio 3: RMS similar
    if abs(fp1["rms"] - fp2["rms"]) > RMS_TOLERANCE:
        return False
    
    # Criterio 4: Espectral similar (si ambos tienen firma)
    if sig1 is not None and sig2 is not None:
        distance = np.linalg.norm(sig1 - sig2)
        spectral_sim = 1 - (distance / 2)  # Normalizar a [0,1]
        
        if spectral_sim < SPECTRAL_SIM_THRESHOLD:
            return False
    
    return True


def find_duplicate_groups_advanced(fingerprints: Dict[Path, dict], 
                                   spectral_sigs: Dict[Path, np.ndarray]) -> List[List[Path]]:
    """Agrupa duplicados usando múltiples criterios"""
    files = list(fingerprints.keys())
    n = len(files)
    
    visited = set()
    groups = []
    
    for i in range(n):
        if i in visited:
            continue
        
        group = [files[i]]
        visited.add(i)
        
        for j in range(i + 1, n):
            if j in visited:
                continue
            
            if are_duplicates(
                fingerprints[files[i]], spectral_sigs.get(files[i]),
                fingerprints[files[j]], spectral_sigs.get(files[j])
            ):
                group.append(files[j])
                visited.add(j)
        
        if len(group) > 1:
            groups.append(group)
    
    return groups


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
    fingerprints = {}
    spectral_sigs = {}
    
    kept = 0
    removed = 0

    # Fase 1: Extracción de características y filtrado básico
    print("\n[FASE 1] Extrayendo características...")
    for path in tqdm(files):
        feats = extract_features(path)
        prediction = is_probable_fire(feats)

        row = {
            "file": str(path),
            "probable_fire": prediction,
        }

        if feats is not None:
            row.update(feats)

        rows.append(row)

        if prediction:
            kept += 1
            if DELETE_DUPLICATES:
                fp, sig = create_fingerprint(path, feats)
                fingerprints[path] = fp
                if sig is not None:
                    spectral_sigs[path] = sig
        else:
            removed += 1

            if DELETE_NON_FIRE:
                try:
                    path.unlink()
                    print(f"Eliminado (no-fuego): {path.name}")
                except Exception as e:
                    print(f"No se pudo eliminar {path}: {e}")

    # Fase 2: Detección de duplicados (método avanzado)
    if DELETE_DUPLICATES and len(fingerprints) > 1:
        print("\n[FASE 2] Detectando duplicados (método avanzado)...")
        print(f"Criterios:")
        print(f"  Peak: ±{PEAK_TOLERANCE} ({PEAK_TOLERANCE*100:.1f}%)")
        print(f"  Duration: ±{DURATION_TOLERANCE}s")
        print(f"  RMS: ±{RMS_TOLERANCE}")
        print(f"  Spectral: {SPECTRAL_SIM_THRESHOLD*100:.0f}%+ similitud")
        
        duplicate_groups = find_duplicate_groups_advanced(fingerprints, spectral_sigs)
        
        duplicates_deleted = 0
        for group in duplicate_groups:
            for dup_file in group[1:]:
                try:
                    dup_file.unlink()
                    duplicates_deleted += 1
                    print(f"Eliminado (duplicado): {dup_file.name}")
                except Exception as e:
                    print(f"No se pudo eliminar duplicado {dup_file}: {e}")
        
        kept -= duplicates_deleted
        removed += duplicates_deleted
        print(f"\nDuplicados eliminados: {duplicates_deleted}")
        print(f"Grupos de duplicados encontrados: {len(duplicate_groups)}")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUTPUT, index=False)

    print("\n========== RESUMEN ==========")
    print(f"Probables fuegos (después de dedup): {kept}")
    print(f"No fuegos/duplicados (eliminados): {removed}")
    print(f"CSV: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
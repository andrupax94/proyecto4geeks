"""
Script para analizar qué archivos se rechazan y probar diferentes parámetros
sin tener que reanalizar todo.

Uso:
    python fire_analyze.py --threshold 0.5 --show-rejected 20
    python fire_analyze.py --list-criteria
"""

import pandas as pd
from pathlib import Path
from src.utils.config import *

# Cambiar esto a la ruta donde guardaste tu CSV
CSV_FILE = BUILD_DIR / "youtube" / "youtube_fire" / "fire.csv"


def analyze_csv(csv_path=CSV_FILE):
    """Analiza el CSV generado para entender qué se rechaza"""
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*60)
    print("ANÁLISIS DEL CSV")
    print("="*60)
    
    print(f"\nTotal de archivos: {len(df)}")
    print(f"Fuegos detectados: {df['probable_fire'].sum()}")
    print(f"Rechazados: {(~df['probable_fire']).sum()}")
    print(f"Ratio: {df['probable_fire'].sum() / len(df) * 100:.1f}% aceptados")
    
    # Estadísticas de archivos aceptados vs rechazados
    print("\n" + "-"*60)
    print("ESTADÍSTICAS - ACEPTADOS vs RECHAZADOS")
    print("-"*60)
    
    accepted = df[df['probable_fire'] == True]
    rejected = df[df['probable_fire'] == False]
    
    features = ['peak', 'rms', 'centroid', 'zcr', 'kurtosis', 'duration', 'attack_ratio']
    
    for feat in features:
        if feat in df.columns:
            acc_mean = accepted[feat].mean()
            rej_mean = rejected[feat].mean()
            acc_std = accepted[feat].std()
            rej_std = rejected[feat].std()
            
            print(f"\n{feat.upper()}:")
            print(f"  Aceptados: {acc_mean:.4f} ± {acc_std:.4f}")
            print(f"  Rechazados: {rej_mean:.4f} ± {rej_std:.4f}")
            print(f"  Diferencia: {abs(acc_mean - rej_mean):.4f}")


def show_rejected_details(csv_path=CSV_FILE, n=20):
    """Muestra archivos rechazados con sus características"""
    
    df = pd.read_csv(csv_path)
    rejected = df[df['probable_fire'] == False].head(n)
    
    print("\n" + "="*60)
    print(f"PRIMEROS {n} ARCHIVOS RECHAZADOS")
    print("="*60)
    
    for idx, row in rejected.iterrows():
        print(f"\n{idx+1}. {Path(row['file']).name}")
        print(f"   Peak: {row['peak']:.3f}")
        print(f"   RMS: {row['rms']:.4f}")
        print(f"   Centroid: {row['centroid']:.0f} Hz")
        print(f"   Duration: {row['duration']:.2f}s")
        print(f"   Kurtosis: {row['kurtosis']:.2f}")


def test_params(csv_path=CSV_FILE, min_peak=0.20, min_centroid=1000, 
                max_duration=5.0, min_kurtosis=2.0, min_rms=0.01):
    """Simula diferentes parámetros y muestra cuántos pasarían"""
    
    df = pd.read_csv(csv_path)
    
    # Aplicar criterios
    mask = (
        (df['peak'] >= min_peak) &
        (df['centroid'] >= min_centroid) &
        (df['duration'] <= max_duration) &
        (df['kurtosis'] >= min_kurtosis) &
        (df['rms'] >= min_rms)
    )
    
    score_mask = (
        (df['peak'] >= min_peak).astype(int) +
        (df['centroid'] >= min_centroid).astype(int) +
        (df['duration'] <= max_duration).astype(int) +
        (df['kurtosis'] >= min_kurtosis).astype(int) +
        (df['rms'] >= min_rms).astype(int)
    )
    
    # Contar archivos que pasan con score >= 3
    would_pass = (score_mask >= 3).sum()
    would_reject = (score_mask < 3).sum()
    
    print("\n" + "="*60)
    print("SIMULACIÓN CON PARÁMETROS PERSONALIZADOS")
    print("="*60)
    print(f"MIN_PEAK: {min_peak}")
    print(f"MIN_CENTROID: {min_centroid}")
    print(f"MAX_DURATION: {max_duration}")
    print(f"MIN_KURTOSIS: {min_kurtosis}")
    print(f"MIN_RMS: {min_rms}")
    
    print(f"\nResultados:")
    print(f"  Pasarían: {would_pass} ({would_pass/len(df)*100:.1f}%)")
    print(f"  Rechazarían: {would_reject} ({would_reject/len(df)*100:.1f}%)")
    
    return would_pass, would_reject


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("HERRAMIENTAS DE ANÁLISIS DE FIRE DETECTOR")
        print("\nUsos:")
        print("  python fire_analyze.py --analyze     # Análisis completo")
        print("  python fire_analyze.py --show 50     # Mostrar 50 rechazados")
        print("  python fire_analyze.py --test 0.15 900 4.0 1.5 0.005  # Probar parámetros")
        print("\n")
        analyze_csv()
    else:
        cmd = sys.argv[1]
        
        if cmd == "--analyze":
            analyze_csv()
        
        elif cmd == "--show":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            show_rejected_details(n=n)
        
        elif cmd == "--test":
            if len(sys.argv) < 7:
                print("Uso: python fire_analyze.py --test MIN_PEAK MIN_CENTROID MAX_DURATION MIN_KURTOSIS MIN_RMS")
                print("Ejemplo: python fire_analyze.py --test 0.15 900 4.0 1.5 0.005")
            else:
                min_peak = float(sys.argv[2])
                min_centroid = float(sys.argv[3])
                max_duration = float(sys.argv[4])
                min_kurtosis = float(sys.argv[5])
                min_rms = float(sys.argv[6])
                
                test_params(min_peak=min_peak, min_centroid=min_centroid,
                           max_duration=max_duration, min_kurtosis=min_kurtosis,
                           min_rms=min_rms)
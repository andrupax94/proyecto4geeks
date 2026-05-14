from pathlib import Path
from src.utils.config import RAW_DIR, BUILD_DIR
import subprocess

import pandas as pd
import os


from src.data.build.dataset import Dataset
from src.data.build.metadata import MetadataEX
from src.utils.config import RAW_DIR, BUILD_DIR
def split_voice_dataset(
    list_file: str | Path,
    audio_dir: str | Path,
    annotation_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
):
    list_file = Path(list_file)
    audio_dir = Path(audio_dir)
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with list_file.open("r", encoding="utf-8") as f:
        wav_names = [line.strip() for line in f if line.strip()]

    total_saved = 0

    for wav_name in wav_names:
        wav_path = audio_dir / wav_name
        ann_path = annotation_dir / f"{Path(wav_name).stem}.txt"

        if not wav_path.exists():
            print(f"[SKIP] No existe el audio: {wav_path}")
            continue

        if not ann_path.exists():
            print(f"[SKIP] No existe la anotación: {ann_path}")
            continue

        print(f"\nProcesando: {wav_name}")

        with ann_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 3:
                    print(f"  [WARN] Línea inválida {ann_path.name}:{i} -> {line}")
                    continue

                try:
                    start_sec = float(parts[0])
                    end_sec = float(parts[1])
                    label = parts[2].strip()
                except ValueError:
                    print(f"  [WARN] No se pudo leer la línea {i}: {line}")
                    continue

                if end_sec <= start_sec:
                    print(f"  [WARN] Segmento inválido {start_sec}-{end_sec} en {ann_path.name}:{i}")
                    continue

                class_dir = output_dir / label
                class_dir.mkdir(parents=True, exist_ok=True)

                out_name = f"{Path(wav_name).stem}_{start_sec:.2f}_{end_sec:.2f}.wav"
                out_path = class_dir / out_name

                if out_path.exists() and not overwrite:
                    print(f"  [SKIP] Ya existe: {out_path.name}")
                    continue

                duration = end_sec - start_sec

                cmd = [
                    "ffmpeg",
                    "-y" if overwrite else "-n",
                    "-i", str(wav_path),
                    "-ss", f"{start_sec:.3f}",
                    "-t", f"{duration:.3f}",
                    "-acodec", "pcm_s16le",
                    str(out_path),
                ]

                try:
                    subprocess.run(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True,
                    )
                    total_saved += 1
                    print(f"  [OK] {label}: {out_path.name}")
                except subprocess.CalledProcessError:
                    print(f"  [ERROR] No se pudo exportar: {out_path.name}")

    print(f"\nListo. Segmentos guardados: {total_saved}")


if __name__ == "__main__":
    
    # VOICE_DIR=BUILD_DIR / "VOICe"
    # split_voice_dataset(
    #     list_file=VOICE_DIR/ "target"/ "synthetic_target_adaptation.txt",
    #     audio_dir=VOICE_DIR / "audio",
    #     annotation_dir=VOICE_DIR / "annotation",
    #     output_dir=RAW_DIR / "VOICe",
    #     overwrite=False,
    # )
    script_dir =  BUILD_DIR / "VOICe"
    file_list = {"VOICe.csv":["VOICe.csv",0.8]}
    df_res={}
    for key,val in file_list.items():
      
        # Clases que quieres conservar (ahora puedes usar los nombres oficiales)
        
        nombre_salida = os.path.join(RAW_DIR, key)
        df_res[key] = Dataset.generar_csv_audio(
            df=None,
            col_labels='class',
            col_mids=RAW_DIR / "zenodo.csv",
            col_fname='slice_file_name',
            ruta_carpeta="VOICe/",
            sinonimos=None,
            nocsv=True,
            nombre_salida=nombre_salida,
            dataset_name="VOICe",
            split=val[1]
        )
        print(df_res[key].head())
    nombre_salida = os.path.join(RAW_DIR, "VOICe.csv")
    metadata = MetadataEX(
        csv_path=nombre_salida,
        dataset_name="VOICe",
        folder=RAW_DIR
    )
    metadata.generate_metadata_audio()
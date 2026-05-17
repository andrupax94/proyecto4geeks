import os
import pandas as pd
from pydub import AudioSegment
from src.utils.config import *

def extraer_eventos(
        carpeta_audio,
        df,
        col_file="filename",
        col_onset="onset",
        col_offset="offset",
        col_label="event_label",
        carpeta_salida="recortes",
        csv_salida_path=None,
    ):
    """
    Lee un CSV/TSV con filename, onset, offset, event_label y genera
    nuevos .wav recortados según los tiempos.
    Además genera un CSV con los nuevos nombres.
    """

    os.makedirs(carpeta_salida, exist_ok=True)



    # Lista para el CSV final
    registros_salida = []

    for idx, row in df.iterrows():
        nombre = str(row[col_file])
        onset = float(row[col_onset]) * 1000
        offset = float(row[col_offset]) * 1000
        label = str(row[col_label])

        ruta_audio = os.path.join(carpeta_audio, nombre)

        if not os.path.exists(ruta_audio):
            print(f"⚠️ Archivo no encontrado: {ruta_audio}")
            continue

        audio = AudioSegment.from_wav(ruta_audio)
        fragmento = audio[onset:offset]

        # Nombre nuevo
        new_name = f"{os.path.splitext(nombre)[0]}_{label}_{idx}.wav"
        ruta_salida = os.path.join(carpeta_salida, new_name)

        fragmento.export(ruta_salida, format="wav")

        print(f"✔️ Guardado: {ruta_salida}")

        # Registrar en el CSV final
        registros_salida.append({
            "old_filename": nombre,
            "new_filename": new_name,
            "event_label": label,
            "onset": row[col_onset],
            "offset": row[col_offset],
            "output_path": ruta_salida
        })

    # Guardar CSV final
    if csv_salida_path is not None:
        df_out = pd.DataFrame(registros_salida)
        df_out.to_csv(csv_salida_path, index=False)
        print(f"📄 CSV generado en: {csv_salida_path}")


def main():
    split = "train"
    dataset_name = "driver_safety"

    carpeta_audio = BUILD_DIR / dataset_name / "audios" / split
    df = pd.read_csv(BUILD_DIR / dataset_name / "train.tsv", delimiter="\t")

    carpeta_salida = RAW_DIR / dataset_name / split
    os.makedirs(carpeta_salida, exist_ok=True)

    csv_salida = carpeta_salida / "recortes.csv"

    extraer_eventos(
        carpeta_audio=carpeta_audio,
        df=df,
        col_file="filename",
        col_onset="onset",
        col_offset="offset",
        col_label="event_label",
        carpeta_salida=carpeta_salida,
        csv_salida_path=csv_salida
    )


if __name__ == "__main__":
    main()

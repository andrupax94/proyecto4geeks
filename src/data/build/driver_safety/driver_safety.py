import pandas as pd
import os
from src.data.build.dataset import Dataset
from src.data.build.metadata import MetadataEX
from src.utils.config import RAW_DIR, BUILD_DIR
from src.utils.wav_cutter import *
# --- EJEMPLO DE CONFIGURACIÓN ---
if __name__ == "__main__":
    dataset_name ="driver_safety"
    script_dir =  BUILD_DIR / dataset_name
    file_list = {f"{dataset_name}_train.csv":["train.tsv","train"],f"{dataset_name}_test.csv":["test.tsv","test"]}
    columns_to_filter=[
        "Drift",
        "Hail",
        "Hit",
        "Firefighters",
        "Dog",
        "CarHorn",
        "TruckHorn",
        "Cough",
        "Ambulance",
        "Cry",
        "Police",
        "BoatHorn",
        "Yawn",
        "Cat",
        "Notifications",
        "RingTone",
        "TrainHorn",
        "WarningBeeps",
        "Thunder",
        "Motorcyle",
        "Airplane",
        "Train",
        "Truck",
        "Ignition",
        "Door",
        "Scream",
        "Window",
        "Rain",
        "Crash",
        "ManipulatingObjects",
        "Car",
        "Helicopter",
    ]
    df_res={}
    
    for key,val in file_list.items():
        df_original = pd.read_csv(os.path.join(script_dir, val[0]),delimiter="\t")
        # Clases que quieres conservar (ahora puedes usar los nombres oficiales)
        df_filtrado = df_original[df_original["event_label"].isin(columns_to_filter)]
        nombre_salida = os.path.join(RAW_DIR, key)
        carpeta_audio = BUILD_DIR / dataset_name / "audios" / val[1]
        audio_salida = RAW_DIR / dataset_name / val[1]
        os.makedirs(carpeta_audio, exist_ok=True)
        val[0] = val[1]+".csv"
        print("-------------Iniciando corte de audios------------")
        extraer_eventos(
            carpeta_audio=carpeta_audio,
            df=df_filtrado,
            col_file="filename",
            col_onset="onset",
            col_offset="offset",
            col_label="event_label",
            carpeta_salida=audio_salida,
            csv_salida_path=script_dir / val[0]
        )
        df_filtrado = pd.read_csv(os.path.join(script_dir, val[0]))
        print("-------------Generando CSV audios------------")
        df_res[key] = Dataset.generar_csv_audio(
            df=df_filtrado,
            col_labels='event_label',
            col_mids=RAW_DIR / "zenodo.csv",
            col_fname='new_filename',
            ruta_carpeta=f"{dataset_name}/",
            sinonimos=None,
            nombre_salida=nombre_salida,
            dataset_name=dataset_name,
            split=val[1]
        )
        print(df_res[key].head())
    archivos=[RAW_DIR / f"{dataset_name}_train.csv", RAW_DIR / f"{dataset_name}_test.csv"]
    Dataset.concatenar_y_ordenar_csvs(archivos ,"audio" , RAW_DIR / f"{dataset_name}.csv",delete_old=True)
    
    nombre_salida = os.path.join(RAW_DIR, f"{dataset_name}.csv")
    metadata = MetadataEX(
        csv_path=nombre_salida,
        dataset_name=f"{dataset_name}",
        folder=RAW_DIR
    )
    metadata.generate_metadata_audio()

  
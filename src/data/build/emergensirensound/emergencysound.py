
import pandas as pd
import os
from pathlib import Path
from src.data.build.dataset import Dataset
from src.data.build.metadata import MetadataEX
from src.utils.config import RAW_DIR, BUILD_DIR
# --- EJEMPLO DE CONFIGURACIÓN ---
if __name__ == "__main__":
    script_dir =  BUILD_DIR / "emergencysound"
    file_list = {"emergencysound.csv":["emergencysound.csv",0.8]}
    df_res={}
    for key,val in file_list.items():
       
        # Clases que quieres conservar (ahora puedes usar los nombres oficiales)
        
        nombre_salida = os.path.join(RAW_DIR, key)
        df_res[key] = Dataset.generar_csv_audio(
            df=None,
            col_labels='category',
            col_mids=RAW_DIR / "zenodo.csv",
            col_fname='filename',
            ruta_carpeta="emergencysound/",
            sinonimos=None,
            nocsv=True,
            nombre_salida=nombre_salida,
            dataset_name="emergencysound",
            split=val[1]
        )
        print(df_res[key].head())
    nombre_salida = os.path.join(RAW_DIR, "emergencysound.csv")
    metadata = MetadataEX(
        csv_path=nombre_salida,
        dataset_name="emergencysound",
        folder=RAW_DIR
    )
    metadata.generate_metadata_audio()
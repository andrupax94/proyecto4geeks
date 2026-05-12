import pandas as pd
import os
import json
from pathlib import Path
import csv
from src.data.build.dataset import Dataset
from src.data.build.create_csv_from_raw import Create_csv_from_raw
import re
import hashlib


# --- EJEMPLO DE CONFIGURACIÓN ---
if __name__ == "__main__":
    script_end = Path(__file__).resolve().parents[4] / "data"
    script_end_raw = script_end / "raw"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cargar tu dataframe (asegúrate de que dev.csv esté en la misma carpeta)
    sinonimos = Dataset.cargar_sinonimos_csv(script_dir + "/../sinonimosV2.csv")
    file_list = {"zenodo_train.csv":["dev.csv","train"],"zenodo_test.csv":["eval.csv","test"]}
    df_res={}
    for key,val in file_list.items():
        df_original = pd.read_csv(os.path.join(script_dir, val[0]))
        # Clases que quieres conservar (ahora puedes usar los nombres oficiales)
        
        nombre_salida = os.path.join(script_end_raw, key)
        df_res[key] = Dataset.generar_csv_audio(
            df=df_original,
            col_labels='labels',
            col_mids='mids',
            col_fname='fname',
            ruta_carpeta="zenodo/",
            sinonimos=sinonimos,
            nombre_salida=nombre_salida,
            dataset_name="zenodo",
            split=val[1]
        )
        print(df_res[key].head())
    archivos=[script_end_raw / "zenodo_train.csv", script_end_raw / "zenodo_test.csv"]
    Dataset.concatenar_y_ordenar_csvs(archivos ,"audio" , script_end_raw / "zenodo.csv")
   
  
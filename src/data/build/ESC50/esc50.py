import src.data.build.zenodo_ds.zenodo_ds as zn
import pandas as pd
import os
from pathlib import Path
from src.data.build.dataset import Dataset
# --- EJEMPLO DE CONFIGURACIÓN ---
if __name__ == "__main__":
    script_end = Path(__file__).resolve().parents[4] / "data"
    script_end_raw = script_end / "raw"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cargar tu dataframe (asegúrate de que dev.csv esté en la misma carpeta)
    sinonimos = Dataset.cargar_sinonimos_csv(script_dir + "/../sinonimosV2.csv")
    file_list = {"ESC50.csv":["ESC50.csv",0.8]}
    df_res={}
    for key,val in file_list.items():
        df_original = pd.read_csv(os.path.join(script_dir, val[0]))
        # Clases que quieres conservar (ahora puedes usar los nombres oficiales)
        
        nombre_salida = os.path.join(script_end_raw, key)
        df_res[key] = zn.generar_csv_audio(
            df=df_original,
            col_labels='category',
            col_mids=script_end_raw / "zenodo.csv",
            col_fname='filename',
            ruta_carpeta="ESC50/",
            sinonimos=sinonimos,
            nombre_salida=nombre_salida,
            dataset_name="ESC50",
            split=val[1]
        )
        print(df_res[key].head())


import pandas as pd
import os
from pathlib import Path
from src.data.build.dataset import Dataset
from src.data.build.metadata import MetadataEX
from src.utils.config import RAW_DIR, BUILD_DIR
# --- EJEMPLO DE CONFIGURACIÓN ---
if __name__ == "__main__":
    script_dir =  BUILD_DIR / "audioset"
    file_list = {"audioset.csv":["audioset.csv",0.8]}
    # df_res={}
    # for key,val in file_list.items():
    #     df_original = pd.read_csv(os.path.join(script_dir, val[0]))
    #     # Clases que quieres conservar (ahora puedes usar los nombres oficiales)
        
    #     nombre_salida = os.path.join(RAW_DIR, key)
    #     df_res[key] = Dataset.generar_csv_audio(
    #         df=df_original,
    #         col_labels='category',
    #         col_mids=RAW_DIR / "zenodo.csv",
    #         col_fname='filename',
    #         ruta_carpeta="ESC50/",
    #         sinonimos=None,
    #         nombre_salida=nombre_salida,
    #         dataset_name="ESC50",
    #         split=val[1]
    #     )
    #     print(df_res[key].head())
    nombre_salida = os.path.join(RAW_DIR, "audioset.csv")
    metadata = MetadataEX(
        csv_path=nombre_salida,
        dataset_name="audioset",
        folder=RAW_DIR
    )
    metadata.generate_metadata_audio()
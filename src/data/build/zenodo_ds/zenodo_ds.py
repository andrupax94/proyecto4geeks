import pandas as pd
import os
from src.data.build.dataset import Dataset
from src.data.build.metadata import MetadataEX
from src.utils.config import RAW_DIR, BUILD_DIR
# --- EJEMPLO DE CONFIGURACIÓN ---
if __name__ == "__main__":
    script_dir =  BUILD_DIR / "zenodo_ds"
    file_list = {"zenodo_train.csv":["dev.csv","train"],"zenodo_test.csv":["eval.csv","test"]}
    df_res={}
    for key,val in file_list.items():
        df_original = pd.read_csv(os.path.join(script_dir, val[0]))
        # Clases que quieres conservar (ahora puedes usar los nombres oficiales)
        
        nombre_salida = os.path.join(RAW_DIR, key)
        df_res[key] = Dataset.generar_csv_audio(
            df=df_original,
            col_labels='labels',
            col_mids='mids',
            col_fname='fname',
            ruta_carpeta="zenodo/",
            sinonimos=None,
            nombre_salida=nombre_salida,
            dataset_name="zenodo",
            split=val[1]
        )
        print(df_res[key].head())
    archivos=[RAW_DIR / "zenodo_train.csv", RAW_DIR / "zenodo_test.csv"]
    Dataset.concatenar_y_ordenar_csvs(archivos ,"audio" , RAW_DIR / "zenodo.csv",delete_old=True)
    
    nombre_salida = os.path.join(RAW_DIR, "zenodo.csv")
    metadata = MetadataEX(
        csv_path=nombre_salida,
        dataset_name="zenodo",
        folder=RAW_DIR
    )
    metadata.generate_metadata_audio()

  
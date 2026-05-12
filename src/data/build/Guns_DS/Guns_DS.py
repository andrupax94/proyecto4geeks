import os
from pathlib import Path

from src.data.build.dataset import Dataset
from src.data.build.metadata import MetadataEX
from src.utils.config import RAW_DIR, BUILD_DIR


if __name__ == "__main__":
    # Carpeta raíz del dataset con estructura:
    # Guns_DS/
    #   gunshot/
    #   reload/
    #   ...
    dataset_root = BUILD_DIR / "Guns_DS"

    nombre_salida = RAW_DIR / "Guns_DS.csv"

    df_res = Dataset.generar_csv_audio(
        df=None,
        ruta_carpeta=dataset_root,
        nombre_salida=str(nombre_salida),
        dataset_name="Guns_DS",
        split="train",
        nocsv=True,
        # si tus CSV de mapeo están en otro sitio, cambia estas rutas:
        ruta_csv_sinonimos=RAW_DIR / "canonical_synonym.csv",
        ruta_csv_alert_env=RAW_DIR / "canonical_alertable_environment.csv",
    )

    print(df_res.head())

    metadata = MetadataEX(
        csv_path=str(nombre_salida),
        dataset_name="Guns_DS",
        folder=RAW_DIR
    )
    metadata.generate_metadata_audio()
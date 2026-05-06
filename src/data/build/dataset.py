import pandas as pd
import os
from pathlib import Path
import shutil
from tqdm import tqdm 
from src.data.build.metadata import MetadataEX
class Dataset:
    def generate_filter_audio(csv_path, ruta_origen, ruta_destino):
        """
        Copia archivos de audio filtrados en el CSV desde una ruta de origen a una de destino,
        respetando la estructura de subcarpetas definida en la columna 'folder'.
        """
        # 1. Cargar el CSV
        if not os.path.exists(csv_path):
            print(f"Error: No se encontró el CSV en {csv_path}")
            return

        df = pd.read_csv(csv_path)
        
        # Verificamos si las columnas necesarias existen
        if 'audio' not in df.columns or 'folder' not in df.columns:
            print("Error: El CSV debe tener las columnas 'audio' y 'folder'.")
            return

        print(f"Iniciando copia de {len(df)} archivos...")
        archivos_copiados = 0
        archivos_no_encontrados = 0

        # 2. Iterar sobre el DataFrame
        # Usamos tqdm para ver el progreso en la terminal
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Copiando audios"):
            nombre_archivo = str(row['audio'])
            subcarpeta = str(row['folder'])

            # Construir rutas completas
            # E:origen/{folder}/{audio}
            path_fuente = os.path.join(ruta_origen, subcarpeta, nombre_archivo)
            
            # E:destino/{folder}/
            carpeta_destino_completa = os.path.join(ruta_destino, subcarpeta)
            # E:destino/{folder}/{audio}
            path_destino_completo = os.path.join(carpeta_destino_completa, nombre_archivo)

            # 3. Lógica de copia
            if os.path.exists(path_fuente):
                # Crear la carpeta de destino si no existe (equivalente a mkdir -p)
                os.makedirs(carpeta_destino_completa, exist_ok=True)
                
                # Copiar archivo preservando metadatos
                try:
                    shutil.copy2(path_fuente, path_destino_completo)
                    archivos_copiados += 1
                except Exception as e:
                    print(f"Error al copiar {nombre_archivo}: {e}")
            else:
                archivos_no_encontrados += 1

        print(f"\n--- Proceso de filtrado físico finalizado ---")
        print(f"Exitosos: {archivos_copiados}")
        print(f"No encontrados en origen: {archivos_no_encontrados}")
        print(f"Ubicación destino: {os.path.abspath(ruta_destino)}")


    def generate_enviroment_columns():
        pass

    def cargar_sinonimos_csv(
        path_csv,
        policy="first"  # "error" | "warn" | "first"
    ):
        """
        Carga sinónimos desde CSV con validación de conflictos y
        añade atributos extra: emergency y environment.

        Devuelve un diccionario:

        {
            canon: {
                "synonyms": [...],
                "emergency": True/False,
                "environment": { "interior", "exterior", ... }
            }
        }
        """

        df = pd.read_csv(path_csv)

        sinonimos = {}
        sinonimo_a_canonico = {}
        conflictos = []

        for i, row in df.iterrows():
            canon = str(row["canonical"]).strip()
            syn = str(row["synonym"]).strip()
            env = str(row["environment"]).strip()
            emergency = str(row["emergency"]).strip()

            if not canon or not syn:
                continue

            # Inicializar estructura extendida
            if canon not in sinonimos:
                sinonimos[canon] = {
                    "synonyms": [],
                    "emergency": False,
                    "environment": set()
                }

            # --- Atributos extra ---
            # emergency: True/False
            if emergency.lower() == "true":
                sinonimos[canon]["emergency"] = True

            # environment: interior/exterior/indefinida
            if env:
                sinonimos[canon]["environment"] = env

            # --- Manejo de sinónimos con conflictos ---
            if syn in sinonimo_a_canonico:
                canon_existente = sinonimo_a_canonico[syn]

                if canon_existente != canon:
                    conflictos.append((syn, canon_existente, canon))

                    if policy == "error":
                        raise ValueError(
                            f"Conflicto: '{syn}' está en '{canon_existente}' y '{canon}'"
                        )

                    elif policy == "warn":
                        print(
                            f"[WARN] '{syn}' estaba en '{canon_existente}' → ahora en '{canon}'"
                        )
                        sinonimos[canon_existente]["synonyms"].remove(syn)
                        sinonimos[canon]["synonyms"].append(syn)
                        sinonimo_a_canonico[syn] = canon

                    elif policy == "first":
                        print(
                            f"[WARN] '{syn}' duplicado → se mantiene en '{canon_existente}', se ignora '{canon}'"
                        )
                        continue

            else:
                sinonimos[canon]["synonyms"].append(syn)
                sinonimo_a_canonico[syn] = canon

        # Limpieza final: quitar canónicos sin sinónimos
        sinonimos = {k: v for k, v in sinonimos.items() if v["synonyms"]}

        # Resumen
        if conflictos:
            print(f"\nResumen: {len(conflictos)} conflictos detectados")
        else:
            print("Sin conflictos detectados 👍")

        return sinonimos

    
    # --- EJEMPLO DE CONFIGURACIÓN ---
    if __name__ == "__main__":
        script_end = Path(__file__).resolve().parents[3] / "data"
        script_end_interim = script_end / "interim"
        script_end_raw = script_end / "raw"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        nombre_salida = os.path.join(script_end_interim, "zenodo.csv")
        
        metadata = MetadataEX(
            csv_path=nombre_salida,
            dataset_name="zenodo",
            folder=script_end_raw
        )

        metadata.generate_metadata_audio()
        # generate_filter_audio(
        #     csv_path=nombre_salida,
        #     ruta_origen=script_end_data, 
        #     ruta_destino=script_end_interim
        # )
    
        # generate_metadata_audio(
        #     csv_path=nombre_salida,
        #     dataset_name="zenodo"
        # )
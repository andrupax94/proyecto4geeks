import pandas as pd
import os
import json
from pathlib import Path
from tinytag import TinyTag
import shutil
from tqdm import tqdm # Opcional: para ver una barra de progreso    
import soundfile as sf
import numpy as np
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

    def generate_metadata_audio(csv_path, dataset_name, folder):
        """
        Lee un CSV, extrae metadatos de los audios y sobrescribe el archivo.
        
        Args:
            csv_path (str): Ruta al archivo CSV (ej: 'dataset_final.csv')
            dataset_name (str): Nombre de la fuente (ej: 'AudioSet_Zenodo')
        """
        if not os.path.exists(csv_path):
            print(f"Error: No se encontró el CSV en {csv_path}")
            return

        # 1. Cargar el dataframe
        df = pd.read_csv(csv_path)

        #----DECIBELES-----------------------------------------
        def max_dbfs(file_path, block_size=65536):
            max_val = 0.0
            
            with sf.SoundFile(file_path) as f:
                for block in f.blocks(blocksize=block_size):
                    max_val = max(max_val, np.max(np.abs(block)))
            
            if max_val == 0:
                return -np.inf
            
            return 20 * np.log10(max_val)
        
        
        # Listas para almacenar los metadatos temporalmente
        durations = []
        sample_rates = []
        channels = []
        bit_depths = []

        print(f"Procesando metadatos para {len(df)} archivos...")

        # 2. Iterar por cada fila para leer el archivo físico
        for index, row in df.iterrows():
            # Usamos la columna 'path' que creamos en el script anterior
            ruta_audio = folder / row['path']
            
            if os.path.exists(ruta_audio):
                try:
                    tag = TinyTag.get(ruta_audio)
                    durations.append(round(tag.duration, 3))
                    sample_rates.append(tag.samplerate)
                    channels.append("Stereo" if tag.channels > 1 else "Mono")
                    bit_depths.append(getattr(tag, 'bitdepth', 16)) # Por defecto 16 si no lo detecta
                except Exception as e:
                    print(f"Error leyendo {ruta_audio}: {e}")
                    durations.append(0.0)
                    sample_rates.append(0)
                    channels.append("Unknown")
                    bit_depths.append(0)
            else:
                # Si el archivo no existe en la ruta especificada
                durations.append(None)
                sample_rates.append(None)
                channels.append("File Not Found")
                bit_depths.append(None)

        # 3. Asignar las nuevas columnas
        df['duration'] = durations
        df['sample_rate'] = sample_rates
        df['channels'] = channels
        df['bit_depth'] = bit_depths
        df['dataset_source'] = dataset_name

        # 4. Sobrescribir el CSV original
        try:
            df.to_csv(csv_path, index=False, quoting=1)
            print(f"--- CSV actualizado con éxito ---")
            print(f"Ruta: {os.path.abspath(csv_path)}")
        except PermissionError:
            print("Error: No se pudo sobrescribir el CSV. Asegúrate de que no esté abierto en Excel.")

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
        script_end = Path(__file__).resolve().parents[2] / "data"
        script_end_interim = script_end / "interim"
        script_end_data = script_end / "data"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        nombre_salida = os.path.join(script_end, "zenodo.csv")
        
        
        # generate_filter_audio(
        #     csv_path=nombre_salida,
        #     ruta_origen=script_end_data, 
        #     ruta_destino=script_end_interim
        # )
    
        # generate_metadata_audio(
        #     csv_path=nombre_salida,
        #     dataset_name="zenodo"
        # )
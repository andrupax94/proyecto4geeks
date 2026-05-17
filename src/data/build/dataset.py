import pandas as pd
import os
from pathlib import Path
import shutil
from tqdm import tqdm 

from src.data.build.create_csv_from_raw import Create_csv_from_raw
from src.utils.config import RAW_DIR
class Dataset:
    generar_csv_audio = Create_csv_from_raw.generar_csv_audio
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
    def preparar_dataframe(dato):
        """
        Si es un string, lo lee como CSV. 
        Si ya es un DataFrame, lo devuelve tal cual.
        """
        if isinstance(dato, pd.DataFrame):
            return dato
        elif isinstance(dato, (str, Path)):
            return pd.read_csv(str(dato))
        else:
            raise ValueError(f"Formato no soportado: {type(dato)}. Debe ser una ruta (str) o un DataFrame.")
        
    def concatenar_y_ordenar_csvs(archivos, columna_orden, archivo_salida, delete_old=False, sum_data=False):
        """
        Concatena varios archivos CSV con las mismas columnas, los ordena,
        guarda el resultado y elimina los archivos de entrada.

        Parámetros:
            archivos: lista de rutas CSV de entrada
            columna_orden: columna por la que ordenar
            archivo_salida: ruta del CSV final
            delete_old: si True, elimina los CSV originales tras guardar
            sum_data: si True, agrega los datos al archivo_salida existente
        """
        try:
            dataframes = [Dataset.preparar_dataframe(archivo) for archivo in archivos]

            df_resultado = pd.concat(dataframes, ignore_index=True)

            # Si sum_data=True y ya existe el archivo final, se agrega su contenido
            if sum_data and os.path.exists(archivo_salida):
                df_existente = Dataset.preparar_dataframe(archivo_salida)
                df_resultado = pd.concat([df_existente, df_resultado], ignore_index=True)
                print(f"Datos previos cargados desde '{archivo_salida}' y añadidos al resultado.")

            if columna_orden in df_resultado.columns:
                df_resultado = df_resultado.sort_values(by=columna_orden)
                print(f"Ordenado por la columna: '{columna_orden}'")
            else:
                print(f"Advertencia: La columna '{columna_orden}' no existe. Se guardará sin ordenar.")

            df_resultado.to_csv(archivo_salida, index=False)
            print(f"Éxito: Archivo guardado como '{archivo_salida}'")

            # Eliminar archivos originales solo después de guardar correctamente
            if delete_old:
                for archivo in archivos:
                    if os.path.exists(archivo):
                        os.remove(archivo)
                        print(f"Eliminado: '{archivo}'")
                    else:
                        print(f"No existe para eliminar: '{archivo}'")

        except FileNotFoundError:
            print("Error: Uno de los archivos no fue encontrado.")
        except Exception as e:
            print(f"Ocurrió un error inesperado: {e}")

    # Ejemplo de uso:
    # concatenar_y_ordenar_csvs('datos_enero.csv', 'datos_febrero.csv', 'fecha', 'reporte_final.csv')
    # --- EJEMPLO DE CONFIGURACIÓN ---
if __name__ == "__main__":
    archivos=[
            RAW_DIR / "zenodo.csv", 
            RAW_DIR / "ESC50.csv",
            RAW_DIR / "UrbanSound8k.csv",
            RAW_DIR / "audioset.csv",
            RAW_DIR / "Guns_DS.csv",
            RAW_DIR / "VOICe.csv",
            RAW_DIR / "Enhanced_audio_of_accident.csv",
            RAW_DIR / "emergencysound.csv",
            RAW_DIR / "driver_safety.csv",
            ]
    Dataset.concatenar_y_ordenar_csvs(archivos ,"audio" , RAW_DIR / "dataset_final.csv")
        
    

       
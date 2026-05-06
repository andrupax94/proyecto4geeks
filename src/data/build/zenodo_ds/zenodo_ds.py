import pandas as pd
import os
import json
from pathlib import Path
import csv
from src.data.build.dataset import Dataset

def generar_csv_audio(
    df,
    col_labels,
    col_mids,
    col_fname,
    ruta_carpeta,
    sinonimos=None,
    nombre_salida="zenodo.csv",
    dataset_name="zenodo",
    split="train"):
    """
    Procesa un dataframe de audio y AGRUPA clases parecidas usando `sinonimos`.
    Ya no filtra por lista de clases.
    """
   
    # 0. Preparar mapa de sinónimos -> clase canónica
    mapa_sinonimos = {}

    if sinonimos:
        for nombre_oficial, info in sinonimos.items():
            nombre_oficial_limpio = str(nombre_oficial).strip()

            # Canonical → sí mismo
            mapa_sinonimos[nombre_oficial_limpio] = nombre_oficial_limpio

            # 👇 AQUÍ ESTÁ LA CLAVE
            for sinonimo in info["synonyms"]:
                mapa_sinonimos[str(sinonimo).strip()] = nombre_oficial_limpio

    # --- FUNCIONES DE APOYO ---
    def normalizar_etiqueta(etiqueta):
        etiqueta = str(etiqueta).strip()
        return mapa_sinonimos.get(etiqueta, etiqueta)

    def obtener_labels_limpias(row):
        """
        Devuelve la lista de etiquetas humanas ya agrupadas por sinónimos.
        Mantiene el orden y elimina duplicados.
        """
        etiquetas_orig = [e.strip() for e in str(row[col_labels]).split(",") if e.strip()]

        etiquetas_normalizadas = []
        vistos = set()

        for e in etiquetas_orig:
            canonica = normalizar_etiqueta(e)
            if canonica not in vistos:
                vistos.add(canonica)
                etiquetas_normalizadas.append(canonica)

        return etiquetas_normalizadas

    def obtener_clase_principal(etiquetas_fila):
        return etiquetas_fila[0] if etiquetas_fila else None
     

    def obtener_mid_singular(row):
        """
        Busca el MID correspondiente a la clase principal.
        Si no encuentra coincidencia, devuelve el primer MID disponible.
        """
        mids_lista = [m.strip() for m in str(row[col_mids]).split(",") if m.strip()]
        labels_orig = [e.strip() for e in str(row[col_labels]).split(",") if e.strip()]
        clase_detectada = row["human_label"]

        if not mids_lista:
            return None

        if clase_detectada is None:
            return mids_lista[0]

        for i, orig in enumerate(labels_orig):
            if normalizar_etiqueta(orig) == clase_detectada:
                return mids_lista[i] if i < len(mids_lista) else mids_lista[0]

        return mids_lista[0]
   
    def agregar_etiquetas_env(row):
        canon = row["human_label"]
        if canon in sinonimos:
            info = sinonimos[canon]
            return info["environment"]
    def agregar_etiquetas_emergency(row):
        canon = row["human_label"]
        if canon in sinonimos:
            info = sinonimos[canon]
            return info["emergency"]
           


    # 1. Procesar nombres humanos y agrupa
    df = df.copy()
    df["temp_human_list"] = df.apply(obtener_labels_limpias, axis=1)

    # 2. Crear columna principal agrupada
    df["human_label"] = df["temp_human_list"].apply(obtener_clase_principal)

    # 3. Crear MID principal
    df["label"] = df.apply(obtener_mid_singular, axis=1)

    # 4. Formatear columnas de AUDIO y PATH
    df["audio"] = df[col_fname].apply(
        lambda x: f"{x}.wav" if not str(x).lower().endswith(".wav") else str(x)
    )
    if dataset_name=="UrbanSound8k":
        aux = f"{ruta_carpeta}/fold{df["fold"]}"
        df["path"] = df["audio"].apply(lambda x: os.path.join(aux, x))
    else:
        df["path"] = df["audio"].apply(lambda x: os.path.join(ruta_carpeta, x))
    df["dataset_source"] = dataset_name
    if split not in ["train", "test"]:
        # Mezclamos y asignamos 80% a train y el resto a test
        # random_state=42 asegura que el split sea siempre el mismo si corres el código de nuevo
        train_df = df.sample(frac=split, random_state=42)
        df["split"] = "test"
        df.loc[train_df.index, "split"] = "train"
    else:
        df["split"] = split

    # 5. Columnas en lista
    df["labels"] = df[col_mids].apply(
        lambda x: json.dumps([i.strip() for i in str(x).split(",") if i.strip()])
    )
    df["human_labels"] = df["temp_human_list"].apply(json.dumps)
    df["env"] = df.apply(agregar_etiquetas_env, axis=1)
    df["emergency"] = df.apply(agregar_etiquetas_emergency, axis=1)
    # 6. Seleccionar columnas finales
    columnas_finales = [
        "audio",
        "label",
        "labels",
        "human_label",
        "human_labels",
        "path",
        "dataset_source",
        "split",
        "env",
        "emergency",
    ]

    resultado = df[columnas_finales]

    # 7. Guardar CSV
    resultado.to_csv(nombre_salida, index=False, quoting=csv.QUOTE_ALL)

    print(f"Éxito. Archivo guardado en: {os.path.abspath(nombre_salida)}")
    return resultado

# --- EJEMPLO DE CONFIGURACIÓN ---
if __name__ == "__main__":
    script_end = Path(__file__).resolve().parents[4] / "data"
    script_end_interim = script_end / "interim"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cargar tu dataframe (asegúrate de que dev.csv esté en la misma carpeta)

    df_original = pd.read_csv(os.path.join(script_dir, "dev.csv"))
    sinonimos = Dataset.cargar_sinonimos_csv(script_dir + "/../sinonimosV2.csv")

    # Clases que quieres conservar (ahora puedes usar los nombres oficiales)
    nombre_salida = os.path.join(script_end_interim, "zenodo.csv")
    df_res = generar_csv_audio(
        df=df_original,
        col_labels='labels',
        col_mids='mids',
        col_fname='fname',
        ruta_carpeta="zenodo/",
        sinonimos=sinonimos,
        nombre_salida=nombre_salida,
        dataset_name="zenodo"
    )
    print(df_res.head())
   
  
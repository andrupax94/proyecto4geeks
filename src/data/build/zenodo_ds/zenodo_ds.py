import pandas as pd
import os
import json
from pathlib import Path
import csv
from src.data.build.dataset import Dataset
import re
import hashlib

def clave_norm(texto):
    """
    Normaliza texto para comparaciones robustas:
    - lower
    - espacios -> _
    - trim
    """
    texto = str(texto).strip().lower()
    texto = re.sub(r"\s+", "_", texto)
    return texto

def generar_mid_custom(canonical: str, prefijo="/C/") -> str:
    base = clave_norm(canonical)
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
    return f"{prefijo}{base}_{digest}"

def generar_csv_audio(
    df,
    col_labels,
    col_mids,
    col_fname,
    ruta_carpeta,
    sinonimos=None,
    nombre_salida="zenodo.csv",
    dataset_name="zenodo",
    split="train"
):
    """
    Procesa un dataframe de audio y agrupa clases usando `sinonimos`.

    Comportamiento de `col_mids`:
    - Si es una columna existente en `df`, funciona como antes:
        * label sale de esa columna
        * labels y human_labels se rellenan normal
    - Si NO es una columna de `df`, se interpreta como:
        * un DataFrame externo, o
        * una ruta a CSV/XLSX
      y se busca coincidencia por `human_label` para recuperar `label`.
      En ese caso, `labels` y `human_labels` quedan en blanco.
    """

    # ------------------------------------------------------------
    # 0. Mapa de sinónimos -> canonical
    # ------------------------------------------------------------
    mapa_sinonimos = {}

    if sinonimos:
        for nombre_oficial, info in sinonimos.items():
            nombre_oficial_limpio = str(nombre_oficial).strip()

            mapa_sinonimos[clave_norm(nombre_oficial_limpio)] = nombre_oficial_limpio

            for sinonimo in info.get("synonyms", []):
                mapa_sinonimos[clave_norm(sinonimo)] = nombre_oficial_limpio

    def normalizar_etiqueta(etiqueta):
        clave = clave_norm(etiqueta)
        return mapa_sinonimos.get(clave, str(etiqueta).strip())

    def obtener_labels_limpias(row):
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

    # ------------------------------------------------------------
    # 1. Resolver si col_mids es columna local o dataset externo
    # ------------------------------------------------------------
    modo_mids = "columna"
    lookup_label_mid = None
    df_mids_externo = None

    if isinstance(col_mids, pd.DataFrame):
        modo_mids = "externo"
        df_mids_externo = col_mids.copy()

    elif isinstance(col_mids, (str, Path)):

        # Si coincide con una columna existente -> modo normal
        if str(col_mids) in df.columns:
            col_mids = str(col_mids)

        else:
            ruta_mids = Path(col_mids)

            if ruta_mids.exists():
                modo_mids = "externo"

                if ruta_mids.suffix.lower() in [".xlsx", ".xls"]:
                    df_mids_externo = pd.read_excel(ruta_mids)
                else:
                    df_mids_externo = pd.read_csv(ruta_mids)

            else:
                raise ValueError(
                    f"`col_mids`='{col_mids}' no existe como columna "
                    f"de `df` ni como ruta válida a un dataset externo."
                )

    if modo_mids == "externo":
        if "human_label" not in df_mids_externo.columns or "label" not in df_mids_externo.columns:
            raise ValueError(
                "El dataset externo indicado en `col_mids` debe tener columnas `human_label` y `label`."
            )

        # Primer match gana
        lookup_label_mid = {}
        for _, fila in df_mids_externo.iterrows():
            h = fila["human_label"]
            l = fila["label"]

            if pd.isna(h) or pd.isna(l):
                continue

            h_norm = clave_norm(normalizar_etiqueta(h))
            l_val = str(l).strip()

            if h_norm not in lookup_label_mid and l_val:
                lookup_label_mid[h_norm] = l_val

    # ------------------------------------------------------------
    # 2. Helpers para label / labels
    # ------------------------------------------------------------
    def obtener_mid_singular(row):
        """
        - Modo columna: busca el MID correspondiente a la clase principal como antes.
        - Modo externo: busca en el dataset externo por human_label y devuelve label.
        """
        if modo_mids == "externo":
            canon = row["human_label"]

            if canon is None:
                return ""

            clave = clave_norm(normalizar_etiqueta(canon))

            # Buscar MID existente
            mid = lookup_label_mid.get(clave, "")

            # Si no existe -> generar uno estable
            if not mid:
                mid = generar_mid_custom(canon)

            return mid

        mids_lista = [m.strip() for m in str(row[col_mids]).split(",") if m.strip()]
        labels_orig = [e.strip() for e in str(row[col_labels]).split(",") if e.strip()]
        clase_detectada = row["human_label"]

        if not mids_lista:
            return ""

        if clase_detectada is None:
            return mids_lista[0]

        for i, orig in enumerate(labels_orig):
            if normalizar_etiqueta(orig) == clase_detectada:
                return mids_lista[i] if i < len(mids_lista) else mids_lista[0]

        return mids_lista[0]

    def obtener_labels_mids_lista(row):
        """
        En modo columna: devuelve la lista normal de mids.
        En modo externo: queda vacía.
        """
        if modo_mids == "externo":
            return []

        return [m.strip() for m in str(row[col_mids]).split(",") if m.strip()]

    def obtener_human_labels_lista(row):
        """
        En modo columna: devuelve la lista normal de labels humanas.
        En modo externo: queda vacía.
        """
        if modo_mids == "externo":
            return []

        return row["temp_human_list"]

    def agregar_etiquetas_env(row):
        canon = row["human_label"]
        if sinonimos and canon in sinonimos:
            return sinonimos[canon].get("environment", None)
        return None

    def agregar_etiquetas_emergency(row):
        canon = row["human_label"]
        if sinonimos and canon in sinonimos:
            return sinonimos[canon].get("emergency", None)
        return None

    # ------------------------------------------------------------
    # 3. Procesar dataframe
    # ------------------------------------------------------------
    df = df.copy()

    df["temp_human_list"] = df.apply(obtener_labels_limpias, axis=1)
    df["human_label"] = df["temp_human_list"].apply(obtener_clase_principal)
    df["label"] = df.apply(obtener_mid_singular, axis=1)

    # ------------------------------------------------------------
    # 4. Audio y path
    # ------------------------------------------------------------
    df["audio"] = df[col_fname].apply(
        lambda x: f"{x}.wav" if not str(x).lower().endswith(".wav") else str(x)
    )

    if dataset_name == "UrbanSound8k":
        if "fold" not in df.columns:
            raise ValueError("Para dataset_name='UrbanSound8k' necesitas una columna llamada `fold`.")
        df["path"] = df.apply(
            lambda row: os.path.join(f"{ruta_carpeta}/fold{row['fold']}", row["audio"]),
            axis=1
        )
        
    else:
        df["path"] = df["audio"].apply(lambda x: os.path.join(ruta_carpeta, x))

    df["dataset_source"] = dataset_name

    if split not in ["train", "test"]:
        train_df = df.sample(frac=float(split), random_state=42)
        df["split"] = "test"
        df.loc[train_df.index, "split"] = "train"
    else:
        df["split"] = split

    # ------------------------------------------------------------
    # 5. Columnas finales según modo
    # ------------------------------------------------------------
    if modo_mids == "externo":
        df["labels"] = ""
        df["human_labels"] = ""
    else:
        df["labels"] = df.apply(
            lambda row: json.dumps(obtener_labels_mids_lista(row)),
            axis=1
        )
        df["human_labels"] = df.apply(
            lambda row: json.dumps(obtener_human_labels_lista(row)),
            axis=1
        )

    df["env"] = df.apply(agregar_etiquetas_env, axis=1)
    df["emergency"] = df.apply(agregar_etiquetas_emergency, axis=1)

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
    resultado.to_csv(nombre_salida, index=False, quoting=csv.QUOTE_ALL)

    print(f"Éxito. Archivo guardado en: {os.path.abspath(nombre_salida)}")
    return resultado

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
        df_res[key] = generar_csv_audio(
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
    Dataset.concatenar_y_ordenar_csvs(script_end_raw / "zenodo_train.csv",script_end_raw / "zenodo_test.csv","audio",script_end_raw / "zenodo.csv")
   
  
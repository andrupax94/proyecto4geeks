import os
import csv
import json
from pathlib import Path
import pandas as pd
import re
import hashlib
from src.utils.config import BUILD_DIR, RAW_DIR
class Create_csv_from_raw:
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
        base = Create_csv_from_raw.clave_norm(canonical)
        digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
        return f"{prefijo}{base}_{digest}"
    def generar_csv_audio(
        df=None,
        col_labels=None,
        col_mids=None,
        col_fname=None,
        ruta_carpeta=None,
        sinonimos=None,
        nombre_salida="zenodo.csv",
        dataset_name="zenodo",
        split="train",
        nocsv=False,
        ruta_csv_sinonimos=BUILD_DIR / "sinonimosV4.csv",
        ruta_csv_alert_env=BUILD_DIR / "canonical_clasesV2.csv",
    ):
        """
        Genera un CSV de audio en dos modos:

        1) Modo normal (nocsv=False):
        - Usa el dataframe `df` y las columnas `col_labels`, `col_mids`, `col_fname`.
        - Mantiene la lógica anterior.

        2) Modo carpeta (nocsv=True):
        - Ignora `df`.
        - Recorre `ruta_carpeta` y toma como clase la estructura de carpetas.
        - Ejemplo:
            ruta_carpeta = "mi_dataset"
            archivo = "mi_dataset/clase1/audio1.wav"
            human_label = "mi_dataset/clase1"
        - La canonical se resuelve con `canonical_synonym.csv`.
        - `label` se resuelve desde ese diccionario; si no existe, se genera un MID estable.
        - `alertable`, `emergency` y `env` se cargan desde `canonical_alertable_environment.csv`.
        """

 

        # ------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------
        def es_vacio(v):
            return v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v)

        def a_bool(v):
            if es_vacio(v):
                return False
            if isinstance(v, bool):
                return v
            s = str(v).strip().lower()
            return s in {"1", "true", "t", "yes", "y", "si", "sí", "on"}

        def parse_lista_valor(v):
            if es_vacio(v):
                return []
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            s = str(v).strip()
            if not s:
                return []
            # intenta JSON primero
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return [str(x).strip() for x in obj if str(x).strip()]
            except Exception:
                pass
            # fallback por separadores típicos
            for sep in ["|", ";", ","]:
                if sep in s:
                    return [x.strip() for x in s.split(sep) if x.strip()]
            return [s]

        def norm(x):
            return Create_csv_from_raw.clave_norm(str(x).strip())

        def cargar_sinonimos_desde_csv(path_csv):
            """
            Soporta CSV con columnas tipo:
            - canonical
            - synonym / synonyms / sinonym / syn
            - label / mid (opcional)
            """
            mapa_syn_to_canon = {}
            mapa_canon_to_mid = {}

            if not path_csv or not Path(path_csv).exists():
                return mapa_syn_to_canon, mapa_canon_to_mid

            df_syn = pd.read_csv(path_csv)

            cols = {c.lower().strip(): c for c in df_syn.columns}
            canonical_col = cols.get("canonical") or cols.get("canon") or cols.get("human_label") or cols.get("label")
            synonym_col = cols.get("synonym") or cols.get("synonyms") or cols.get("sinonym") or cols.get("syn")
            label_col = cols.get("label") or cols.get("mid") or cols.get("id")

            if canonical_col is None:
                raise ValueError(
                    f"`{path_csv}` debe tener al menos una columna `canonical` "
                    f"(o equivalente como `label`)."
                )

            for _, row in df_syn.iterrows():
                canonical = row.get(canonical_col, None)
                if es_vacio(canonical):
                    continue

                canonical = str(canonical).strip()
                if not canonical:
                    continue

                canon_key = norm(canonical)
                mapa_syn_to_canon[canon_key] = canonical

                if label_col is not None and not es_vacio(row.get(label_col, None)):
                    mapa_canon_to_mid[canon_key] = str(row.get(label_col)).strip()

                # el canonical también es su propio sinónimo
                mapa_syn_to_canon[canon_key] = canonical

                if synonym_col is not None and not es_vacio(row.get(synonym_col, None)):
                    for syn in parse_lista_valor(row.get(synonym_col)):
                        mapa_syn_to_canon[norm(syn)] = canonical

            return mapa_syn_to_canon, mapa_canon_to_mid

        def cargar_alertas_desde_csv(path_csv):
            """
            Espera columnas:
            - canonical
            - alertable
            - emergency
            - environment (o env)
            """
            mapa = {}

            if not path_csv or not Path(path_csv).exists():
                return mapa

            df_map = pd.read_csv(path_csv)
            cols = {c.lower().strip(): c for c in df_map.columns}
            canonical_col = cols.get("canonical") or cols.get("label")
            alertable_col = cols.get("alertable")
            emergency_col = cols.get("emergency")
            env_col = cols.get("environment") or cols.get("env")

            if canonical_col is None:
                raise ValueError(
                    f"`{path_csv}` debe tener una columna `canonical` (o equivalente `label`)."
                )

            for _, row in df_map.iterrows():
                canonical = row.get(canonical_col, None)
                if es_vacio(canonical):
                    continue

                canonical = str(canonical).strip()
                if not canonical:
                    continue

                key = norm(canonical)
                mapa[key] = {
                    "alertable": a_bool(row.get(alertable_col, False)) if alertable_col else False,
                    "emergency": a_bool(row.get(emergency_col, False)) if emergency_col else False,
                    "env": None if env_col is None or es_vacio(row.get(env_col, None)) else str(row.get(env_col)).strip(),
                }

            return mapa

        def resolver_canonica(texto, mapa_syn):
            """
            Intenta resolver por:
            1) texto completo
            2) último segmento del path
            """
            if es_vacio(texto):
                return str(texto).strip()

            raw = str(texto).strip()
            candidatos = [raw]

            # último segmento por si llega "mi_dataset/clase1"
            if "/" in raw:
                candidatos.append(raw.split("/")[-1].strip())
            if "\\" in raw:
                candidatos.append(raw.split("\\")[-1].strip())

            for cand in candidatos:
                key = norm(cand)
                if key in mapa_syn:
                    return mapa_syn[key]

            return raw

        # ------------------------------------------------------------
        # Cargar mapas externos
        # ------------------------------------------------------------
        mapa_sinonimos = {}
        mapa_canonico_mid = {}

        # si viene el diccionario viejo, lo respetamos
        if isinstance(sinonimos, dict) and sinonimos:
            for nombre_oficial, info in sinonimos.items():
                canon = str(nombre_oficial).strip()
                if not canon:
                    continue

                mapa_sinonimos[norm(canon)] = canon

                # si el dict trae label/mid, lo guardamos
                mid_val = None
                if isinstance(info, dict):
                    mid_val = info.get("label", None) or info.get("mid", None)

                    for syn in info.get("synonyms", []):
                        mapa_sinonimos[norm(syn)] = canon

                if mid_val is not None and str(mid_val).strip():
                    mapa_canonico_mid[norm(canon)] = str(mid_val).strip()

        # fallback / complemento desde CSV
        mapa_syn_csv, mapa_mid_csv = cargar_sinonimos_desde_csv(ruta_csv_sinonimos)
        mapa_sinonimos.update(mapa_syn_csv)
        mapa_canonico_mid.update(mapa_mid_csv)

        mapa_alertas = cargar_alertas_desde_csv(ruta_csv_alert_env)

        def normalizar_etiqueta(etiqueta):
            canon = resolver_canonica(etiqueta, mapa_sinonimos)
            return canon

        # ------------------------------------------------------------
        # Modo nocsv=True -> construir desde estructura de carpetas
        # ------------------------------------------------------------
        if nocsv:
            if ruta_carpeta is None:
                raise ValueError("`ruta_carpeta` es obligatorio.")

            ruta_carpeta = Path(RAW_DIR /ruta_carpeta)
            if not ruta_carpeta.exists():
                raise ValueError(f"`ruta_carpeta` no existe: {ruta_carpeta}")
            filas = []

            extensiones_validas = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

            for archivo in ruta_carpeta.rglob("*"):
                if not archivo.is_file():
                    continue
                if archivo.suffix.lower() not in extensiones_validas:
                    continue

                rel_parent = archivo.parent.relative_to(ruta_carpeta)
                normalized_label = rel_parent.as_posix().lower().replace(" ", "_")
                if not normalized_label in mapa_syn_csv.keys():
                    human_label = normalized_label
                else:
                    human_label = mapa_syn_csv[normalized_label]

                canonical = normalizar_etiqueta(human_label)

                # label/MID: primero intenta por canonical, luego genera uno estable
                mid = mapa_canonico_mid.get(norm(canonical), "")
                if not mid:
                    mid = Create_csv_from_raw.generar_mid_custom(canonical)

                info = mapa_alertas.get(norm(canonical), {})
                alertable = info.get("alertable", False)
                emergency = info.get("emergency", False)
                env = info.get("env", None)

                filas.append(
                    {
                        "audio": archivo.name,
                        "label": mid,
                        "labels": json.dumps([mid]),
                        "human_label": human_label,
                        "human_labels": json.dumps([human_label]),
                        "path": str(archivo),
                        "dataset_source": dataset_name,
                        "split": split if split in ["train", "test"] else "train",
                        "env": env,
                        "emergency": emergency,
                        "alertable": alertable,
                    }
                )

            resultado = pd.DataFrame(filas)

            if resultado.empty:
                raise ValueError(f"No se encontraron audios válidos dentro de {ruta_carpeta}")

            # split numérico opcional
            if split not in ["train", "test"]:
                train_df = resultado.sample(frac=float(split), random_state=42)
                resultado["split"] = "test"
                resultado.loc[train_df.index, "split"] = "train"

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
                "alertable",
                "emergency",
            ]

            resultado = resultado[columnas_finales]
            resultado.to_csv(nombre_salida, index=False, quoting=csv.QUOTE_ALL)
            print(f"Éxito. Archivo guardado en: {os.path.abspath(nombre_salida)}")
            return resultado

        # ------------------------------------------------------------
        # Modo normal -> usa df
        # ------------------------------------------------------------
        if df is None:
            raise ValueError("`df` es obligatorio cuando `nocsv=False`.")

        if col_labels is None or col_mids is None or col_fname is None:
            raise ValueError("Debes indicar `col_labels`, `col_mids` y `col_fname` cuando `nocsv=False`.")

        # ------------------------------------------------------------
        # 1. Procesar labels
        # ------------------------------------------------------------
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
        # 2. Resolver col_mids
        # ------------------------------------------------------------
        modo_mids = "columna"
        lookup_label_mid = None
        df_mids_externo = None

        if isinstance(col_mids, pd.DataFrame):
            modo_mids = "externo"
            df_mids_externo = col_mids.copy()

        elif isinstance(col_mids, (str, Path)):
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
                        f"`col_mids`='{col_mids}' no existe como columna de `df` ni como ruta válida."
                    )

        if modo_mids == "externo":
            if "human_label" not in df_mids_externo.columns or "label" not in df_mids_externo.columns:
                raise ValueError(
                    "El dataset externo indicado en `col_mids` debe tener columnas `human_label` y `label`."
                )

            lookup_label_mid = {}
            for _, fila in df_mids_externo.iterrows():
                h = fila["human_label"]
                l = fila["label"]

                if pd.isna(h) or pd.isna(l):
                    continue

                h_norm = norm(normalizar_etiqueta(h))
                l_val = str(l).strip()

                if h_norm not in lookup_label_mid and l_val:
                    lookup_label_mid[h_norm] = l_val

        def obtener_mid_singular(row):
            if modo_mids == "externo":
                canon = row["human_label"]
                if canon is None:
                    return ""
                clave = norm(normalizar_etiqueta(canon))
                mid = lookup_label_mid.get(clave, "")
                if not mid:
                    mid = Create_csv_from_raw.generar_mid_custom(canon)
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
            if modo_mids == "externo":
                return []
            return [m.strip() for m in str(row[col_mids]).split(",") if m.strip()]

        def obtener_human_labels_lista(row):
            if modo_mids == "externo":
                return []
            return row["temp_human_list"]

        def agregar_etiquetas_env(row):
            canon = row["human_label"]
            info = mapa_alertas.get(norm(canon), {})
            return info.get("env", None)

        def agregar_etiquetas_emergency(row):
            canon = row["human_label"]
            info = mapa_alertas.get(norm(canon), {})
            return info.get("emergency", False)

        def agregar_etiquetas_alertable(row):
            canon = row["human_label"]
            info = mapa_alertas.get(norm(canon), {})
            return info.get("alertable", False)

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
        elif dataset_name == "driver_safety":
            df["path"] = df["audio"].apply(lambda x: os.path.join(str(f"{ruta_carpeta}{split}/"), x))
        else:
            df["path"] = df["audio"].apply(lambda x: os.path.join(str(ruta_carpeta), x))

        df["dataset_source"] = dataset_name

        if split not in ["train", "test"]:
            train_df = df.sample(frac=float(split), random_state=42)
            df["split"] = "test"
            df.loc[train_df.index, "split"] = "train"
        else:
            df["split"] = split

        # ------------------------------------------------------------
        # 5. Columnas finales
        # ------------------------------------------------------------
        if modo_mids == "externo":
            df["labels"] = ""
            df["human_labels"] = ""
        else:
            df["labels"] = df.apply(lambda row: json.dumps(obtener_labels_mids_lista(row)), axis=1)
            df["human_labels"] = df.apply(lambda row: json.dumps(obtener_human_labels_lista(row)), axis=1)

        df["env"] = df.apply(agregar_etiquetas_env, axis=1)
        df["emergency"] = df.apply(agregar_etiquetas_emergency, axis=1)
        df["alertable"] = df.apply(agregar_etiquetas_alertable, axis=1)

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
            "alertable",
            "emergency",
        ]

        resultado = df[columnas_finales]
        resultado.to_csv(nombre_salida, index=False, quoting=csv.QUOTE_ALL)

        print(f"Éxito. Archivo guardado en: {os.path.abspath(nombre_salida)}")
        return resultado
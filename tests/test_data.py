import pandas as pd
from src.utils.config import RAW_DIR
def obtener_human_labels_env_vacio(ruta_csv, n_muestra=20):
    """
    Devuelve una muestra de human_label cuyo campo 'env' esté vacío,
    sea 'indefinido', 'indefinida' o NaN.
    """

    df = pd.read_csv(ruta_csv)

    # Normalizar columna env a string
    df["env"] = df["env"].astype(str).str.strip().str.lower()

    # Condiciones de "env vacío"
    condiciones = (
        (df["env"].isna()) |
        (df["env"] == "") |
        (df["env"] == "nan")
    )

    # Filtrar
    filtrado = df.loc[condiciones, "human_label"]

    # Quitar duplicados
    unicos = filtrado.drop_duplicates()

    # Devolver muestra
    return unicos.sample(min(n_muestra, len(unicos))).tolist()


# Ejemplo de uso:
if __name__ == "__main__":
    ruta = RAW_DIR / "ESC50.csv" 
    muestra = obtener_human_labels_env_vacio(ruta)
    print("Muestra de human_label con env vacío o indefinido:")
    print(muestra)

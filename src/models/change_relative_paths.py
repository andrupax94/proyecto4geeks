import pandas as pd
from pathlib import Path, PureWindowsPath
from src.utils.config import INTERIM_DIR

PROJECT_ROOT = Path.cwd().resolve()

PATH_COLUMNS = [
    "mel_spec_path",
    "mfcc_path",
    "audio_normalized_path",
]

PROCESSED_METADATA = INTERIM_DIR / "processed_dataset"
input_csv = PROCESSED_METADATA / "processed_metadata.csv"
output_csv = PROCESSED_METADATA / "metadata_relative.csv"

def to_relative(path):
    if pd.isna(path) or str(path).strip() == "":
        return path

    # Convierte "E:\Proyectos\..." a partes tipo Windows
    p = PureWindowsPath(str(path))

    # Si existe letra de unidad, nos quedamos solo con la parte después de la raíz del disco
    parts = p.parts
    if len(parts) >= 2 and parts[0].endswith(":\\"):
        # ejemplo: ('E:\\', 'Proyectos', 'proyecto4geeks', 'data', ...)
        try:
            idx = parts.index("proyecto4geeks")
            rel_parts = parts[idx:]
            return str(Path(*rel_parts)).replace("\\", "/")
        except ValueError:
            # si no encuentra el proyecto, quita la unidad y deja el resto
            return str(Path(*parts[1:])).replace("\\", "/")

    return str(path).replace("\\", "/")

df = pd.read_csv(input_csv)

for col in PATH_COLUMNS:
    if col in df.columns:
        df[col] = df[col].apply(to_relative)

df.to_csv(output_csv, index=False)

print(f"✔ Rutas convertidas a relativas en: {output_csv}")
print(df[PATH_COLUMNS].head())
import os
import pandas as pd
import shutil
import subprocess

# ==========================
# Carpeta donde está el script
# ==========================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# carpeta final de sonidos
SONIDOS_PATH = os.path.join(BASE_PATH, "sonidos")
os.makedirs(SONIDOS_PATH, exist_ok=True)

# ==========================
# Descargar ESC-50
# ==========================
ESC50_PATH = os.path.join(BASE_PATH, "ESC-50")

if not os.path.exists(ESC50_PATH):
    print("Descargando ESC-50...")
    subprocess.run([
        "git", "clone",
        "https://github.com/karolpiczak/ESC-50.git",
        ESC50_PATH
    ])

# ==========================
# Clases necesarias
# ==========================
CLASES = [
    "siren",
    "glass_breaking",
    "fireworks",
    "engine",
    "car_horn",
    "wind",
    "footsteps",
    "laughing",
    "breathing"
]

# ==========================
# Leer metadata
# ==========================
meta_path = os.path.join(ESC50_PATH, "meta", "esc50.csv")
meta = pd.read_csv(meta_path)

meta = meta[meta["category"].isin(CLASES)]

# ==========================
# Crear carpetas por categoría
# ==========================
for clase in CLASES:
    os.makedirs(os.path.join(SONIDOS_PATH, clase), exist_ok=True)

# ==========================
# Copiar audios
# ==========================
for _, row in meta.iterrows():

    archivo = row["filename"]
    clase = row["category"]

    origen = os.path.join(ESC50_PATH, "audio", archivo)
    destino = os.path.join(SONIDOS_PATH, clase, archivo)

    shutil.copy(origen, destino)

print("\n✅ Dataset creado correctamente en:")
print(SONIDOS_PATH)
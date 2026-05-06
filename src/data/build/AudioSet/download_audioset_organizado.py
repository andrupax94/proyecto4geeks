import os
import pandas as pd
import subprocess

# ==========================
# CONFIG RUTAS
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 📄 CSV (queda en AudioSet)
CSV_FILE = os.path.join(BASE_DIR, "balanced_train_segments.csv")

# 📁 Carpeta principal AudioSet
AUDIOSET_DIR = BASE_DIR

# 📁 Carpeta final donde se organizarán sonidos
OUTPUT_DIR = os.path.join(AUDIOSET_DIR, "sonidos")

print("CSV:", CSV_FILE)
print("OUTPUT:", OUTPUT_DIR)

MAX_DOWNLOADS = 40

# ==========================
# CLASES
# ==========================

CLASSES = {
    # 🚨 Emergencia
    "gunshot": "/m/09x0r",
    "scream": "/m/03qc9zr",
    "glass_breaking": "/m/07r04",
    "fireworks": "/m/01j3sz",
    "siren": "/m/012n7d",
    "fight": "/m/0jbk",

    # 🌆 Normal
    "engine": "/m/02mfyn",
    "car_horn": "/m/0284vy3",
    "wind": "/m/03l2n",
    "music": "/m/04rlf"
}

# ==========================
# CREAR CARPETAS
# ==========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for categoria in CLASSES:
    os.makedirs(
        os.path.join(OUTPUT_DIR, categoria),
        exist_ok=True
    )

# ==========================
# LEER CSV (FORMA ROBUSTA)
# ==========================

df_rows = []

with open(CSV_FILE, "r", encoding="utf-8") as f:
    for line in f:

        if line.startswith("#"):
            continue

        parts = line.strip().split(",")

        if len(parts) < 4:
            continue

        ytid = parts[0]
        start = parts[1]
        end = parts[2]
        labels = ",".join(parts[3:])  # evita errores por comas extra

        df_rows.append([ytid, start, end, labels])

df = pd.DataFrame(df_rows, columns=["YTID", "start", "end", "labels"])

print("Filas cargadas:", len(df))

# ==========================
# DESCARGA
# ==========================

contador = 0

for _, row in df.iterrows():

    if contador >= MAX_DOWNLOADS:
        break

    labels = str(row["labels"])

    for class_name, label_code in CLASSES.items():

        if label_code in labels:

            ytid = row["YTID"]

            try:
                start = float(row["start"])
                end = float(row["end"])
            except:
                continue

            url = f"https://youtube.com/watch?v={ytid}"

            output_file = os.path.join(
                OUTPUT_DIR,
                class_name,
                f"{ytid}.wav"
            )

            print("Descargando:", url, "->", class_name)

            cmd = [
                "python",
                "-m",
                "yt_dlp",
                "-x",
                "--audio-format", "wav",
                "--download-sections",
                f"*{start}-{end}",
                "-o", output_file,
                url,
            ]

            try:
                subprocess.run(cmd, check=True)
                contador += 1
            except Exception as e:
                print("Error con:", url, e)

            break

print("DESCARGA FINALIZADA")
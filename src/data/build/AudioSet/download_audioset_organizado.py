import os
import pandas as pd
import subprocess
import sys

# ==========================
# CONFIG RUTAS
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(BASE_DIR, "balanced_train_segments.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "sonidos")

MAX_DOWNLOADS = 40

print("CSV:", CSV_FILE)
print("OUTPUT:", OUTPUT_DIR)

# ==========================
# CLASES AUDIOSET
# ==========================

CLASSES = {

    # 🚨 EMERGENCIA
    "gunshot": "/m/09x0r",
    "scream": "/m/03qc9zr",
    "glass_breaking": "/m/07r04",
    "fireworks": "/m/01j3sz",
    "siren": "/m/012n7d",
    "fight": "/m/0jbk",

    # 🌆 NORMAL
    "engine": "/m/02mfyn",
    "car_horn": "/m/0284vy3",
    "wind": "/m/03l2n",
    "music": "/m/04rlf",
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
# LEER CSV CORRECTAMENTE
# ==========================

print("\nLeyendo CSV AudioSet...")

print("\nLeyendo CSV AudioSet (modo robusto)...")

rows = []

with open(CSV_FILE, "r", encoding="utf-8") as f:

    for line in f:

        # ignorar comentarios
        if line.startswith("#"):
            continue

        # eliminar comillas
        line = line.replace('"', '')

        parts = line.strip().split(",")

        # proteger líneas corruptas
        if len(parts) < 4:
            continue

        ytid = parts[0]
        start = parts[1]
        end = parts[2]

        # labels pueden tener MUCHAS comas
        labels = ",".join(parts[3:])

        rows.append([ytid, start, end, labels])

df = pd.DataFrame(
    rows,
    columns=["YTID", "start", "end", "labels"]
)

print("Filas cargadas:", len(df))
print(df.head())


# ==========================
# DESCARGA
# ==========================

contador = 0

for _, row in df.iterrows():

    if contador >= MAX_DOWNLOADS:
        break

    labels = str(row["labels"]).split(",")

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
                f"{ytid}.%(ext)s"
            )

            print(f"\nDescargando {class_name} -> {url}")

            cmd = [
                sys.executable,
                "-m",
                "yt_dlp",
                "-x",
                "--audio-format", "wav",
                "--download-sections",
                f"*{start}-{end}",
                "-o",
                output_file,
                url,
                "--quiet",
                "--no-warnings"
            ]

            try:
                subprocess.run(cmd, check=True)
                contador += 1
                print("✅ OK:", contador)

            except Exception as e:
                print("❌ Error:", e)

            break

print("\n🎯 DESCARGA FINALIZADA")
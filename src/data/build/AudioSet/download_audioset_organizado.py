import os
import pandas as pd
import subprocess
import sys

print("Cargando AudioSet...")

# ========================
# RUTAS
# ========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_AUDIOSET = os.path.join(
    BASE_DIR,
    "balanced_train_segments.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "sonidos")

# ========================
# CLASES
# ========================

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

TARGET_PER_CLASS = 80

# ========================
# CREAR CARPETAS
# ========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for c in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, c), exist_ok=True)

# ========================
# LEER AUDIOSET
# ========================

rows = []

with open(CSV_AUDIOSET, "r", encoding="utf-8") as f:
    for line in f:

        if line.startswith("#"):
            continue

        parts = line.strip().split(",")

        if len(parts) < 4:
            continue

        rows.append([
            parts[0],
            parts[1],
            parts[2],
            ",".join(parts[3:])
        ])

df = pd.DataFrame(
    rows,
    columns=["YTID", "start", "end", "labels"]
)

print("Total filas:", len(df))

# ========================
# CONTADORES
# ========================

downloaded = {c: 0 for c in CLASSES}

# ========================
# DESCARGA MASIVA INTELIGENTE
# ========================

for _, row in df.iterrows():

    # detener cuando TODAS estén completas
    if all(v >= TARGET_PER_CLASS for v in downloaded.values()):
        break

    labels = row["labels"]

    for class_name, label_code in CLASSES.items():

        if downloaded[class_name] >= TARGET_PER_CLASS:
            continue

        if label_code not in labels:
            continue

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

        if os.path.exists(output_file):
            downloaded[class_name] += 1
            continue

        print(f"⬇ {class_name} -> {ytid}")

        cmd = [
            sys.executable,
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
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )

            downloaded[class_name] += 1

            print("✅", downloaded)

        except:
            pass

print("\nDESCARGA TERMINADA")
print(downloaded)
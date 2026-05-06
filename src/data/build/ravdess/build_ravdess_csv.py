import os
import pandas as pd
import wave

# ==========================
# RUTAS
# ==========================

# carpeta donde está el script
BUILD_DIR = os.path.dirname(os.path.abspath(__file__))

# subir hasta proyecto4geeks
PROJECT_ROOT = os.path.abspath(
    os.path.join(BUILD_DIR, "..", "..", "..", "..")
)

# 📁 audios RAVDESS
RAVDESS_PATH = os.path.join(BUILD_DIR, "sonidos")

SINONIMOS_PATH = os.path.abspath(
    os.path.join(BUILD_DIR, "..", "sinonimosV2.csv")
)

# 📁 destino FINAL
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(RAW_DATA_DIR, "ravdess_dataset.csv")

# ==========================
# MAPEO EMOCIONES RAVDESS
# ==========================

emotion_map = {
    "05": "RAVDESS_angry",
    "06": "RAVDESS_fearful",
    "07": "RAVDESS_disgust",
    "08": "RAVDESS_surprised",
}

# ==========================
# CARGAR DICCIONARIO
# ==========================

sinonimos = pd.read_csv(SINONIMOS_PATH)

synonym_to_class = {}
synonym_to_emergency = {}
synonym_to_env = {}

for _, row in sinonimos.iterrows():
    synonym_to_class[row["synonym"]] = row["canonical"]
    synonym_to_emergency[row["synonym"]] = row["emergency"]
    synonym_to_env[row["synonym"]] = row["environment"]

# ==========================
# FUNCION INFO AUDIO
# ==========================

def audio_info(path):
    with wave.open(path, "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        channels = w.getnchannels()
        duration = frames / float(rate)

    size_kb = os.path.getsize(path) / 1024

    return duration, rate, channels, size_kb


# ==========================
# RECORRER AUDIOS
# ==========================

rows = []
id_label = 0

for root, _, files in os.walk(RAVDESS_PATH):
    for file in files:

        if not file.endswith(".wav"):
            continue

        partes = file.split("-")

        if len(partes) < 3:
            continue

        emotion_code = partes[2]

        if emotion_code not in emotion_map:
            continue

        synonym = emotion_map[emotion_code]

        if synonym not in synonym_to_class:
            continue

        canonical = synonym_to_class[synonym]
        emergency = synonym_to_emergency[synonym]
        environment = synonym_to_env[synonym]

        audio_path = os.path.join(root, file)

        duration, sr, channels, size_kb = audio_info(audio_path)

        rows.append({
            "id_label": id_label,
            "audio": file,
            "folder": os.path.basename(root),
            "label": canonical,
            "human_label": canonical,
            "labels": canonical,
            "human_labels": canonical,
            "danger_level": 2 if emergency else 0,
            "split": "train",
            "dataset_source": "RAVDESS",
            "duration": duration,
            "sample_rate": sr,
            "channels": "Mono" if channels == 1 else "Stereo",
            "bit_depth": 16,
            "bit_velocity": "",
            "siz": size_kb,
            "date_modification": os.path.getmtime(audio_path),
            "environment": environment,
            "emergency": emergency
        })

        id_label += 1


# ==========================
# GUARDAR CSV
# ==========================

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ CSV generado: {OUTPUT_CSV}")
print(f"🎧 Audios procesados: {len(df)}")
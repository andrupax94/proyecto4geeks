from pathlib import Path

# ===============================
# ROOT
# ===============================

ROOT_DIR = Path(__file__).resolve().parents[2]

# ===============================
# DATA GENERAL
# ===============================

DATA_DIR = ROOT_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

SRC_DIR = ROOT_DIR / "src" 
BUILD_DIR = SRC_DIR / "data" / "build"

# ===============================
# BUILD DATA
# ===============================

BUILD_DIR = ROOT_DIR / "src" / "data" / "build"

# ===============================
# AUDIOSET
# ===============================

AUDIOSET_BUILD_DIR = BUILD_DIR / "AudioSet"
AUDIOSET_SONIDOS_DIR = AUDIOSET_BUILD_DIR / "sonidos"

SINONIMOS_V3_PATH = BUILD_DIR / "sinonimosV3.csv"
CANONICAL_CLASSES_PATH = BUILD_DIR / "canonical_clases.csv"

AUDIOSET_DATASET_PATH = RAW_DIR / "audioset.csv"

# ===============================
# FEATURES / ESPECTOGRAMAS
# ===============================

ESPECTOGRAMS_DIR = INTERIM_DIR / "processed_dataset"

LABEL_MAPPING = {
    "human_label": INTERIM_DIR / "processed_dataset" / "label_mapping_human_label.pkl",
    "alertable": INTERIM_DIR / "processed_dataset" / "label_mapping_alertable.pkl",
}
PROCESSED_METADATA = ESPECTOGRAMS_DIR / "processed_metadata.csv"
PROCESSED_METADATA_SPLIT_FIX = ESPECTOGRAMS_DIR / "dataset_fixed.csv"

# ===============================
# DATASETS FINALES
# ===============================

DATASET_FINAL = RAW_DIR / "dataset_final.csv"
EDA_BALANCED = RAW_DIR / "eda_balanced.csv"

# ===============================
# MODELOS
# ===============================
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
HISTORY_DIR = MODELS_DIR / "history"
FINAL_MODEL_DIR = MODELS_DIR / "final"

# =========================
# YOUTUBE  DATASETS
# =========================

# 💥 EXPLOSIONS
YOUTUBE_EXPLOSIONS_DIR = DATA_DIR / "youtube_explosions"

YOUTUBE_RAW_DIR = YOUTUBE_EXPLOSIONS_DIR / "raw"
YOUTUBE_PROCESSED_DIR = YOUTUBE_EXPLOSIONS_DIR / "explosion"
YOUTUBE_URLS_FILE = YOUTUBE_EXPLOSIONS_DIR / "youtube_urls.txt"
TEST_FOLDER=ROOT_DIR / "tests"
TEST_AUDIO_FOLDER= TEST_FOLDER/ "audios"

# 🔥 FIRE
YOUTUBE_FIRE_DIR = DATA_DIR / "youtube_fire"
YOUTUBE_FIRE_VIDEO_DIR = YOUTUBE_FIRE_DIR / "raw"
YOUTUBE_FIRE_AUDIO_DIR = YOUTUBE_FIRE_DIR / "fire"
YOUTUBE_URLS_FILE_FIRE= YOUTUBE_FIRE_DIR / "youtube_urls.txt"

# PARAMETROS AUDIO
MAX_VIDEO_DURATION = 1800
SEGMENT_SECONDS = 5
TARGET_SR = 16000
CHANNELS = 1
ENERGY_THRESHOLD = 0.003
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

# ===============================
# DATASETS FINALES
# ===============================

DATASET_FINAL = RAW_DIR / "dataset_final.csv"

# ===============================
# MODELOS
# ===============================

CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
FINAL_MODEL_DIR = ROOT_DIR / "models" / "final"

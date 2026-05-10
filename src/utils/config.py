from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
ESPECTOGRAMS_DIR = INTERIM_DIR / "processed_dataset"
PROCESSED_DIR = DATA_DIR / "processed"

CHECKPOINT_MODEL= ROOT_DIR / "models" / "checkpoints"
CHECKPOINT_MODEL= ROOT_DIR / "models" / "final"


DATASET_FINAL = RAW_DIR / "dataset_final.csv"
LABEL_MAPPING = ESPECTOGRAMS_DIR / "label_mapping.pkl"
PROCESSED_METADATA = ESPECTOGRAMS_DIR / "processed_metadata.csv"

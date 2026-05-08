from pathlib import Path

ROOT_DIR = Path(_file_).resolve().parents[3]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
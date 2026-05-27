from fastapi import APIRouter
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd

from src.utils.config import RAW_DIR, FINAL_MODEL_DIR

router = APIRouter(prefix="/dashboard")

CSV_DIR = RAW_DIR / "dataset_finalV2.csv"
CACHE_FILE = RAW_DIR.parent / "interim" / "dashboard_cache.json"
METRICS_FILE = FINAL_MODEL_DIR / "metrics.json"
CACHE_TTL = timedelta(days=1)


def _load_csv() -> pd.DataFrame:
    return pd.read_csv(CSV_DIR)


def _to_bool_series(series: pd.Series) -> pd.Series:
    """
    Convierte una columna a booleanos de forma segura.
    Acepta True/False, 1/0, "true"/"false", "yes"/"no", etc.
    """
    if series.dtype == bool:
        return series.fillna(False)

    normalized = series.astype(str).str.lower().str.strip()
    return normalized.isin(["true", "1", "yes", "y", "t"])


def _load_metrics() -> dict:
    """Lee las métricas del modelo desde metrics.json generado por el notebook de evaluación."""
    if not METRICS_FILE.exists():
        return {
            "model_accuracy": 0.8563, 
            "f1_macro": 0.8511, 
            "f1_weighted": 0.8558,
            "multiclass": {
                "accuracy": 0.7977,
                "f1_macro": 0.7960,
                "f1_weighted": 0.7960
            },
            "class_report": {
                "car_crash": {"precision": 0.864, "recall": 0.864, "f1-score": 0.864, "support": 176},
                "construction_noise": {"precision": 0.856, "recall": 0.812, "f1-score": 0.834, "support": 176},
                "crying": {"precision": 0.766, "recall": 0.875, "f1-score": 0.817, "support": 176},
                "dog": {"precision": 0.912, "recall": 0.881, "f1-score": 0.896, "support": 176},
                "fight": {"precision": 0.962, "recall": 0.852, "f1-score": 0.904, "support": 176},
                "fire": {"precision": 0.806, "recall": 0.472, "f1-score": 0.595, "support": 176},
                "glass_breaking": {"precision": 0.887, "recall": 0.847, "f1-score": 0.866, "support": 176},
                "gun_explosion": {"precision": 0.796, "recall": 0.955, "f1-score": 0.868, "support": 176},
                "siren_alarm": {"precision": 0.662, "recall": 0.756, "f1-score": 0.706, "support": 176},
                "traffic": {"precision": 0.565, "recall": 0.665, "f1-score": 0.611, "support": 176}
            }
        }
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"model_accuracy": 0.8563, "f1_macro": 0.8511, "f1_weighted": 0.8558}


def _compute_dashboard_data() -> dict:
    df = _load_csv()

    # Columnas útiles
    label_col = "human_label" if "human_label" in df.columns else None
    source_col = "dataset_source" if "dataset_source" in df.columns else None
    format_col = "audio_format" if "audio_format" in df.columns else None
    duration_col = "duration" if "duration" in df.columns else None
    rate_col = "sample_rate" if "sample_rate" in df.columns else None
    alertable_col = "alertable" if "alertable" in df.columns else None

    alertable_mask = _to_bool_series(df[alertable_col]) if alertable_col else pd.Series([False] * len(df))

    total_files = int(len(df))
    classes = int(df[label_col].nunique()) if label_col else 0
    alertable_count = int(alertable_mask.sum())
    no_alertable_count = int((~alertable_mask).sum())

    # Distribución alertable / no alertable
    alertable_dist = []
    no_alertable_dist = []

    if label_col and alertable_col:
        alertable_dist = (
            df[alertable_mask][label_col]
            .value_counts()
            .rename_axis("class")
            .reset_index(name="count")
            .to_dict(orient="records")
        )

        no_alertable_dist = (
            df[~alertable_mask][label_col]
            .value_counts()
            .rename_axis("class")
            .reset_index(name="count")
            .to_dict(orient="records")
        )

    # Distribución por fuente
    source_dist = []
    if source_col:
        source_dist = (
            df[source_col]
            .value_counts()
            .rename_axis("source")
            .reset_index(name="count")
            .to_dict(orient="records")
        )

    # Distribución por formato
    audio_format_dist = []
    if format_col:
        audio_format_dist = (
            df[format_col]
            .value_counts()
            .rename_axis("format")
            .reset_index(name="count")
            .to_dict(orient="records")
        )

    # Estadísticas de duración
    duration_stats = {}
    if duration_col:
        duration_stats = {
            "mean": float(df[duration_col].mean()),
            "median": float(df[duration_col].median()),
            "min": float(df[duration_col].min()),
            "max": float(df[duration_col].max()),
            "std": float(df[duration_col].std()),
        }

    # Distribución por sample rate
    sample_rate_dist = []
    if rate_col:
        sample_rate_dist = (
            df[rate_col]
            .value_counts()
            .rename_axis("rate")
            .reset_index(name="count")
            .to_dict(orient="records")
        )

    metrics = _load_metrics()

    return {
        "stats": {
            "total_files": total_files,
            "classes": classes,
            "alertable_count": alertable_count,
            "no_alertable_count": no_alertable_count,
            "model_accuracy": metrics.get("model_accuracy", 0.8563),
            "f1_macro": metrics.get("f1_macro", 0.8511),
            "f1_weighted": metrics.get("f1_weighted", 0.8558),
            "multiclass": metrics.get("multiclass", {
                "accuracy": 0.7977,
                "f1_macro": 0.7960,
                "f1_weighted": 0.7960
            }),
            "class_report": metrics.get("class_report")
        },
        "eda": {
            "alertable": alertable_dist,
            "no_alertable": no_alertable_dist,
            "dataset_source_distribution": source_dist,
            "audio_format_distribution": audio_format_dist,
            "duration_stats": duration_stats,
            "sample_rate_distribution": sample_rate_dist,
        },
        "generated_at": datetime.utcnow().isoformat(),
    }


def _read_cache() -> dict | None:
    if not CACHE_FILE.exists():
        return None

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)

        generated_at = datetime.fromisoformat(cached["generated_at"])
        if datetime.utcnow() - generated_at > CACHE_TTL:
            return None

        return cached
    except Exception:
        return None


def _write_cache(data: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _get_cached_data() -> dict:
    cached = _read_cache()
    if cached is not None:
        return cached

    fresh = _compute_dashboard_data()
    _write_cache(fresh)
    return fresh


@router.get("/stats")
def stats():
    data = _get_cached_data()
    return data["stats"]


@router.get("/eda")
def eda():
    data = _get_cached_data()
    return data["eda"]
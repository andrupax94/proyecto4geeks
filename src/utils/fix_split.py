from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import PROCESSED_METADATA, ESPECTOGRAMS_DIR

# =========================================================
# CONFIG
# =========================================================
@dataclass
class CFG:
    input_csv: str = "dataset.csv"
    output_csv: str = "dataset_fixed.csv"

    audio_col: str = "audio"
    label_col: str | None = None

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    seed: int = 42
    fix: bool = True


# =========================================================
# UTILIDADES
# =========================================================
def infer_label_column(df: pd.DataFrame) -> str:
    if "human_label" in df.columns and df["human_label"].notna().any():
        return "human_label"
    if "label" in df.columns:
        return "label"
    raise ValueError("No encuentro 'human_label' ni 'label'.")


def build_group_id(audio_name: str) -> str:
    stem = Path(str(audio_name)).stem
    parts = stem.split("-")
    return "-".join(parts[:-1]) if len(parts) >= 4 else stem


def build_group_table(df: pd.DataFrame, audio_col: str, label_col: str) -> pd.DataFrame:
    df = df.copy()
    df["group_id"] = df[audio_col].apply(build_group_id)

    groups = []
    for gid, g in df.groupby("group_id"):
        label = g[label_col].fillna("__nan__").mode()
        label = label.iloc[0] if len(label) else "__nan__"

        groups.append({
            "group_id": gid,
            "label": label,
            "n_rows": len(g)
        })

    return pd.DataFrame(groups)


# =========================================================
# REPORT
# =========================================================
def split_report(df: pd.DataFrame, label_col: str) -> None:
    print("\n=== DISTRIBUCIÓN SPLITS ===")
    print(df["split"].value_counts())

    print("\n=== DISTRIBUCIÓN GLOBAL ===")
    print(df[label_col].value_counts())

    print("\n=== CHECK RÁPIDO ===")
    for split, g in df.groupby("split"):
        print(f"\n--- {split} ---")
        print(g[label_col].value_counts(normalize=True).head(10))


# =========================================================
# SPLIT CORRECTO (SIN LEAKAGE)
# =========================================================
def stratified_group_split(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    label_col = cfg.label_col or infer_label_column(df)

    work = df.copy()
    work["group_id"] = work[cfg.audio_col].apply(build_group_id)

    group_df = build_group_table(work, cfg.audio_col, label_col)

    train_ratio = cfg.train_ratio
    val_ratio = cfg.val_ratio
    test_ratio = cfg.test_ratio

    temp_ratio = val_ratio + test_ratio

    train_groups, temp_groups = train_test_split(
        group_df,
        test_size=temp_ratio,
        random_state=cfg.seed,
        stratify=group_df["label"]
    )

    val_ratio_adj = val_ratio / temp_ratio

    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=(1 - val_ratio_adj),
        random_state=cfg.seed,
        stratify=temp_groups["label"]
    )

    train_ids = set(train_groups["group_id"])
    val_ids = set(val_groups["group_id"])
    test_ids = set(test_groups["group_id"])

    def assign(gid):
        if gid in train_ids:
            return "train"
        if gid in val_ids:
            return "val"
        if gid in test_ids:
            return "test"
        return "unknown"

    work["split"] = work["group_id"].apply(assign)

    if (work["split"] == "unknown").any():
        raise RuntimeError("Hay filas sin split asignado.")

    return work.drop(columns=["group_id"])


# =========================================================
# MAIN
# =========================================================
def main():
    cfg = CFG(
        input_csv=PROCESSED_METADATA,
        output_csv=ESPECTOGRAMS_DIR / "dataset_fixed.csv",
        fix=True
    )

    df = pd.read_csv(cfg.input_csv)

    label_col = cfg.label_col or infer_label_column(df)

    print("\n==============================")
    print("📊 DATASET ORIGINAL")
    print("==============================")

    split_report(df, label_col)

    if not cfg.fix:
        print("\n❌ No se aplicó fix (cfg.fix=False)")
        return

    print("\n==============================")
    print("🔧 REGENERANDO SPLIT")
    print("==============================")

    fixed = stratified_group_split(df, cfg)

    print("\n==============================")
    print("📊 DATASET CORREGIDO")
    print("==============================")

    split_report(fixed, label_col)

    fixed.to_csv(cfg.output_csv, index=False)
    print(f"\n💾 Guardado en: {cfg.output_csv}")


if __name__ == "__main__":
    main()
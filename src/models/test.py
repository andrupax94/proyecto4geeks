import pandas as pd
from src.utils.config import RAW_DIR, BUILD_DIR
# =========================
# CONFIG
# =========================
DATASET_CSV = RAW_DIR / "dataset_final.csv"
CANONICAL_CSV = BUILD_DIR / "canonical_alertable_environment.csv"
OUTPUT_CSV = RAW_DIR / "dataset_updated.csv"

# =========================
# CARGAR CSVs
# =========================
df = pd.read_csv(DATASET_CSV)
canonical_df = pd.read_csv(CANONICAL_CSV)

# =========================
# NORMALIZAR
# =========================
df["human_label"] = df["human_label"].astype(str).str.strip().str.lower()
canonical_df["canonical"] = canonical_df["canonical"].astype(str).str.strip().str.lower()

# =========================
# MAPAS
# =========================
alertable_map = dict(
    zip(
        canonical_df["canonical"],
        canonical_df["alertable"]
    )
)

environment_map = dict(
    zip(
        canonical_df["canonical"],
        canonical_df["environment"]
    )
)

# =========================
# CREAR ALERTABLE
# =========================
df["alertable"] = df["human_label"].map(alertable_map)

# Si no encuentra valor => False
df["alertable"] = df["alertable"].fillna(False)

# =========================
# ACTUALIZAR ENVIRONMENT
# =========================
df["env"] = df["human_label"].map(environment_map).fillna(df["env"])

# =========================
# LIMPIAR BOOLEANOS
# =========================
df["alertable"] = df["alertable"].astype(str).str.lower().map({
    "true": True,
    "false": False
})

# =========================
# GUARDAR
# =========================
df.to_csv(OUTPUT_CSV, index=False)

print(f"CSV actualizado guardado en: {OUTPUT_CSV}")
print(df[["human_label", "alertable", "env"]].head())
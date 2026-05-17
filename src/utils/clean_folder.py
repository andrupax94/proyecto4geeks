from pathlib import Path
from src.utils.config import *
def clean_directory(path: str, keep_ext=(".wav", ".csv",".tsv", ".py", ".ipynb"), recursive=True, dry_run=False):
    base_path = Path(path)

    if not base_path.exists():
        raise ValueError(f"La ruta no existe: {path}")

    files = base_path.rglob("*") if recursive else base_path.glob("*")

    deleted = 0

    for file in files:
        if file.is_file():
            if file.suffix.lower() not in keep_ext:
                if dry_run:
                    print(f"[DRY-RUN] Eliminaría: {file}")
                else:
                    file.unlink()
                    print(f"Eliminado: {file}")
                deleted += 1

    print(f"\nTotal archivos eliminados: {deleted}")


# =========================
# AUTO-EJECUCIÓN
# =========================
if __name__ == "__main__":
  
    path = BUILD_DIR / "driver_safety"
    clean_directory(
        path=path,
        recursive=True,
        dry_run=False
    )
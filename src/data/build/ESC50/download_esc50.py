from pathlib import Path
import shutil
import subprocess
import sys
import pandas as pd
import os
import stat
import time

# ==========================
# Configuración
# ==========================
BASE_PATH = Path(__file__).resolve().parent

ESC50_REPO = "https://github.com/karolpiczak/ESC-50.git"

ESC50_PATH = BASE_PATH / "ESC-50"
META_PATH = ESC50_PATH / "meta" / "esc50.csv"
AUDIO_PATH = ESC50_PATH / "audio"

# salida final
SONIDOS_PATH = BASE_PATH.parents[3] / "data" / "raw" / "ESC50"
SONIDOS_PATH.mkdir(parents=True, exist_ok=True)

# copia local del CSV
CSV_DESTINO = BASE_PATH / "esc50.csv"

# ==========================
# Clases necesarias
# ==========================
CLASES = [
    "siren",
    "glass_breaking",
    "fireworks",
    "engine",
    "car_horn",
    "wind",
    "footsteps",
    "laughing",
    "breathing",
]

# ==========================
# Utilidades
# ==========================
def ejecutar_comando(cmd):
    resultado = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if resultado.returncode != 0:
        raise RuntimeError(
            f"Error ejecutando comando:\n{' '.join(cmd)}\n\n"
            f"STDOUT:\n{resultado.stdout}\n\n"
            f"STDERR:\n{resultado.stderr}"
        )

    return resultado


def _borrar_readonly(func, path, exc_info):
    """
    Handler para borrar archivos/dirs con permisos de solo lectura en Windows.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def eliminar_dataset():
    if ESC50_PATH.exists():
        print("🗑 Eliminando carpeta ESC-50...")
        shutil.rmtree(ESC50_PATH, onerror=_borrar_readonly)

        # pequeña comprobación
        if ESC50_PATH.exists():
            raise RuntimeError(f"No se pudo eliminar: {ESC50_PATH}")


def clonar_dataset():
    print("⬇ Descargando ESC-50...")

    # por seguridad, borrar antes de clonar
    if ESC50_PATH.exists():
        eliminar_dataset()

    ejecutar_comando([
        "git",
        "clone",
        "--depth", "1",
        ESC50_REPO,
        str(ESC50_PATH)
    ])

    # borrar git para evitar locks en Windows
    git_path = ESC50_PATH / ".git"
    if git_path.exists():
        shutil.rmtree(git_path, onerror=_borrar_readonly)
        print("🧹 Carpeta .git eliminada.")


# ==========================
# Procesamiento
# ==========================
def crear_dataset_filtrado():
    meta = pd.read_csv(META_PATH)

    # filtrar categorías
    meta = meta[meta["category"].isin(CLASES)].copy()

    if meta.empty:
        raise ValueError("No se encontraron categorías válidas.")

    # guardar CSV filtrado al lado del script
    meta.to_csv(CSV_DESTINO, index=False)
    print(f"📝 CSV copiado en:\n{CSV_DESTINO}")

    copiados = 0
    faltantes = []

    # copiar audios en una sola carpeta
    for _, row in meta.iterrows():
        archivo = row["filename"]
        origen = AUDIO_PATH / archivo
        destino = SONIDOS_PATH / archivo

        if not origen.exists():
            faltantes.append(str(origen))
            continue

        shutil.copy2(origen, destino)
        copiados += 1

    print("\n✅ Dataset filtrado creado.")
    print(f"📁 Carpeta salida: {SONIDOS_PATH}")
    print(f"📦 Archivos copiados: {copiados}")

    if faltantes:
        print(f"\n⚠ Archivos faltantes: {len(faltantes)}")

    # eliminar repo descargado
    eliminar_dataset()
    print("🧹 Repositorio ESC-50 eliminado.")
def dataset_valido():
    """
    Verifica estructura mínima del dataset.
    """
    if not ESC50_PATH.exists():
        return False

    if not META_PATH.exists():
        return False

    if not AUDIO_PATH.exists():
        return False

    wavs = list(AUDIO_PATH.glob("*.wav"))
    if len(wavs) == 0:
        return False

    return True
def asegurar_dataset():
    if dataset_valido():
        print("✅ ESC-50 ya existe y parece válido.")
        return

    print("⚠ Dataset inexistente o corrupto.")
    eliminar_dataset()
    clonar_dataset()

    if not dataset_valido():
        raise RuntimeError(
            "El dataset sigue siendo inválido después de reclonarlo."
        )

    print("✅ ESC-50 descargado correctamente.")

# ==========================
# Main
# ==========================
def main():
    asegurar_dataset()
    crear_dataset_filtrado()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR:\n{e}")
        sys.exit(1)
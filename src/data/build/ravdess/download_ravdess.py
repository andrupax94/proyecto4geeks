import os
import shutil

# ==========================
# Carpeta donde está el script
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 📂 Dataset RAVDESS (tu ruta actual)
SOURCE = os.path.join(BASE_DIR, "Audio_Speech_Actors_01-24")

# 📂 Carpeta destino
DEST = os.path.join(BASE_DIR, "sonidos")

# ==========================
# Emociones útiles
# ==========================
emociones = {
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Crear carpetas destino
for emocion in emociones.values():
    os.makedirs(os.path.join(DEST, emocion), exist_ok=True)

print("🔎 Buscando audios RAVDESS...")

contador = 0

# ==========================
# Recorrer actores
# ==========================
for root, _, files in os.walk(SOURCE):

    for file in files:

        if file.lower().endswith(".wav"):

            partes = file.split("-")

            if len(partes) >= 3:

                codigo = partes[2]

                if codigo in emociones:

                    origen = os.path.join(root, file)
                    destino = os.path.join(
                        DEST,
                        emociones[codigo],
                        file
                    )

                    shutil.copy2(origen, destino)
                    contador += 1

print(f"✅ Audios copiados: {contador}")
import os
import shutil

# ==========================
# RUTA BASE (donde está el script)
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# carpeta destino
DEST = os.path.join(BASE_DIR, "sonidos")

emociones = {
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# crear carpetas destino
for emocion in emociones.values():
    os.makedirs(os.path.join(DEST, emocion), exist_ok=True)

print("Buscando audios...")

contador = 0

# ==========================
# BUSCAR EN TODAS LAS SUBCARPETAS
# ==========================
for root, dirs, files in os.walk(BASE_DIR):

    # evitar copiar desde sonidos nuevamente
    if "sonidos" in root:
        continue

    for file in files:

        if file.lower().endswith(".wav"):

            partes = file.split("-")

            if len(partes) >= 3:

                codigo = partes[2]

                if codigo in emociones:

                    emocion = emociones[codigo]

                    origen = os.path.join(root, file)
                    destino = os.path.join(DEST, emocion, file)

                    shutil.copy2(origen, destino)
                    contador += 1

print(f"✅ Audios copiados: {contador}")
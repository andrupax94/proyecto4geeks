from src.utils.config import (
    YOUTUBE_FIRE_VIDEO_DIR,
    YOUTUBE_FIRE_AUDIO_DIR,
    SEGMENT_SECONDS,
    TARGET_SR,
    ENERGY_THRESHOLD
)

import librosa
import soundfile as sf
import hashlib
import numpy as np
from tqdm import tqdm

# crear carpeta output
YOUTUBE_FIRE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

hashes = set()


def audio_hash(audio):
    return hashlib.md5(audio.tobytes()).hexdigest()


audio_files = list(YOUTUBE_FIRE_VIDEO_DIR.glob("*.wav"))

print(f"\n🔥 Audios encontrados: {len(audio_files)}")

for audio_path in tqdm(audio_files):

    print(f"\nProcesando: {audio_path.name}")

    try:
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        print("Error leyendo:", audio_path.name, e)
        continue

    segment_length = TARGET_SR * SEGMENT_SECONDS
    total_segments = len(y) // segment_length

    for i in range(total_segments):

        start = i * segment_length
        end = start + segment_length

        segment = y[start:end]

        # eliminar silencios
        energy = np.mean(np.abs(segment))
        if energy < ENERGY_THRESHOLD:
            continue

        # eliminar duplicados
        h = audio_hash(segment)
        if h in hashes:
            continue

        hashes.add(h)

        out_file = (
            YOUTUBE_FIRE_AUDIO_DIR /
            f"{audio_path.stem}_seg{i}.wav"
        )

        sf.write(out_file, segment, TARGET_SR)

print("\n✅ Dataset FIRE creado correctamente")
from src.utils.config import (
    YOUTUBE_CARCRASH_DIR,
    YOUTUBE_CARCRASH_AUDIO_DIR
)

import librosa
import soundfile as sf
import hashlib
import numpy as np

YOUTUBE_CARCRASH_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_SECONDS = 5
TARGET_SR = 16000
ENERGY_THRESHOLD = 0.04

hashes = set()


def audio_hash(audio):
    return hashlib.md5(audio.tobytes()).hexdigest()


for audio_path in YOUTUBE_CARCRASH_DIR.glob("*.mp3"):

    print("Procesando:", audio_path.name)

    y, sr = librosa.load(audio_path, sr=TARGET_SR)

    segment_length = TARGET_SR * SEGMENT_SECONDS
    total_segments = len(y) // segment_length

    for i in range(total_segments):

        start = i * segment_length
        end = start + segment_length

        segment = y[start:end]

        # quitar silencio
        energy = np.mean(np.abs(segment))
        if energy < ENERGY_THRESHOLD:
            continue

        # eliminar duplicados
        h = audio_hash(segment)
        if h in hashes:
            continue

        hashes.add(h)

        out_file = (
            YOUTUBE_CARCRASH_AUDIO_DIR /
            f"{audio_path.stem}_seg{i}.wav"
        )

        sf.write(out_file, segment, TARGET_SR)
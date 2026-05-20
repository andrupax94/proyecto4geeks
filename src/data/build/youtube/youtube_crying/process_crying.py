# pip install librosa soundfile tqdm

from src.utils.config import (
    YOUTUBE_CRYING_RAW_DIR,
    YOUTUBE_CRYING_DIR_PROCESSED,
    CRYING_ENERGY_THRESHOLD,
)

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm


YOUTUBE_CRYING_DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

CLIP_DURATION = 5  # segundos


for file in tqdm(list(YOUTUBE_CRYING_RAW_DIR.glob("*.wav"))):

    y, sr = librosa.load(file, sr=None)

    samples_per_clip = sr * CLIP_DURATION
    total_clips = len(y) // samples_per_clip

    for i in range(total_clips):

        start = i * samples_per_clip
        end = start + samples_per_clip

        clip = y[start:end]

        energy = np.mean(clip**2)

        # filtro por energía
        if energy < CRYING_ENERGY_THRESHOLD:
            continue

        output_name = f"{file.stem}_{i}.wav"

        sf.write(
            YOUTUBE_CRYING_DIR_PROCESSED / output_name,
            clip,
            sr
        )
        
    print("energy:", energy)
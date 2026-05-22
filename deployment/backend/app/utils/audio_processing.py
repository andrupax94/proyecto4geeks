import librosa
import numpy as np

SAMPLE_RATE = 22050

def load_audio(path):

    audio, sr = librosa.load(
        path,
        sr=SAMPLE_RATE
    )

    return audio


def extract_features(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=40
    )

    return mfcc
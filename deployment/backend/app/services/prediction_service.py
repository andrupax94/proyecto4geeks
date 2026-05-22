import tempfile

from app.utils.audio_processing import (
    load_audio,
    extract_features
)

from app.models.model_loader import model


async def predict_audio(file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp:

        content = await file.read()

        temp.write(content)

        temp_path = temp.name

    audio = load_audio(temp_path)

    features = extract_features(audio)

    # TODO:
    # tensor
    # prediction
    # decoder

    return {
        "prediction": "sirena",
        "confidence": 0.95
    }
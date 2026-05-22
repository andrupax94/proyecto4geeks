from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File

from app.services.prediction_service import (
    predict_audio
)

router = APIRouter()

@router.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    prediction = await predict_audio(file)

    return prediction
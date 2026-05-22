from fastapi import APIRouter, UploadFile, File
from app.services.prediction_service import predict_audio

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    prediction = await predict_audio(file)
    return prediction

from fastapi import APIRouter, UploadFile, File, Query
from app.services.prediction_service import predict_audio

router = APIRouter()

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    version_bin: int = Query(4, description="Versión del modelo binario (Etapa 1)"),
    version_specific: int = Query(4, description="Versión del modelo específico - alertable/no_alertable (Etapa 2)")
):
    """
    Realiza una predicción sobre un archivo de audio en dos etapas.
    Permite especificar versiones independientes para el modelo binario y el específico.
    """
    prediction = await predict_audio(file, version_bin=version_bin, version_specific=version_specific)
    return prediction

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class TestSoundRequest(BaseModel):
    filename: str | None = None
    threshold: float | None = None

@router.post('/predict')
def predict(payload: TestSoundRequest):
    return {
        'status': 'ok',
        'message': 'Prediction placeholder',
        'received': payload.model_dump(),
        'prediction': 'unknown',
    }

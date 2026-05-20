from fastapi import APIRouter

router = APIRouter()

@router.get('/summary')
def summary():
    return {
        'status': 'ok',
        'message': 'Model summary ready',
        'metrics': {'accuracy': 0.91, 'f1': 0.88},
    }

@router.get('/training')
def training():
    return {
        'status': 'ok',
        'message': 'Training history placeholder',
        'epochs': 25,
    }

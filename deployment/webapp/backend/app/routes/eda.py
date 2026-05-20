from fastapi import APIRouter

router = APIRouter()

@router.get('/summary')
def summary():
    return {
        'status': 'ok',
        'message': 'EDA summary ready',
        'items': ['distribution', 'missing_values', 'outliers'],
    }

@router.get('/charts')
def charts():
    return {
        'status': 'ok',
        'message': 'EDA charts placeholder',
        'charts': ['histogram', 'correlation', 'class_balance'],
    }

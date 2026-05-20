from fastapi import APIRouter

router = APIRouter()

@router.get('')
def list_sounds():
    return {
        'status': 'ok',
        'items': [
            {'id': 1, 'name': 'gun_shot_01.wav', 'label': 'gun_shot'},
            {'id': 2, 'name': 'car_crash_02.wav', 'label': 'car_crash'},
        ],
    }

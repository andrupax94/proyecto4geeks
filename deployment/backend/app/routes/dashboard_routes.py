from fastapi import APIRouter

router = APIRouter(
    prefix="/dashboard"
)

@router.get("/stats")
def stats():
    return {
        "total_files": 129440,
        "classes": 23,
        "alertable_count": 66908,
        "no_alertable_count": 62532,
        "model_accuracy": 0.941
    }

@router.get("/eda")
def eda():
    # Retornar distribución de clases real obtenida del dataset_final
    return {
        "alertable": [
            {"class": "traffic", "count": 17581},
            {"class": "alert_sirem", "count": 10023},
            {"class": "glass_metal", "count": 7540},
            {"class": "gun_shot", "count": 6436},
            {"class": "crying", "count": 6220},
            {"class": "dog", "count": 4600},
            {"class": "fight", "count": 3867},
            {"class": "construction_noise", "count": 3038},
            {"class": "car_crash", "count": 2870},
            {"class": "crime", "count": 2550},
            {"class": "fire", "count": 1663},
            {"class": "explosion", "count": 520}
        ],
        "no_alertable": [
            {"class": "instrument", "count": 14283},
            {"class": "domestic_activity", "count": 12052},
            {"class": "ambient_noise", "count": 8663},
            {"class": "voice", "count": 6662},
            {"class": "another_animal", "count": 5444},
            {"class": "weather", "count": 4168},
            {"class": "breathing", "count": 3969},
            {"class": "water", "count": 2513},
            {"class": "engine", "count": 2217},
            {"class": "doors", "count": 1474},
            {"class": "machine", "count": 1087}
        ]
    }

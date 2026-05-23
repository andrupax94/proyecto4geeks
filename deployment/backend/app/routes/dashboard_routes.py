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
            {"class": "glass_metal", "count": 754},
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
        ],
        "dataset_source_distribution": [
            {"source": "zenodo", "count": 51197},
            {"source": "driver_safety", "count": 47726},
            {"source": "VOICe", "count": 9843},
            {"source": "Enhanced_audio_of_accident", "count": 9039},
            {"source": "UrbanSound8k", "count": 8732},
            {"source": "emergencysound", "count": 965},
            {"source": "Guns_DS", "count": 851},
            {"source": "youtube", "count": 596},
            {"source": "ESC50", "count": 360},
            {"source": "audioset", "count": 131}
        ],
        "audio_format_distribution": [
            {"format": "wav", "count": 126686},
            {"format": "wavex", "count": 2753}
        ],
        "duration_stats": {
            "mean": 4.85776651550151,
            "median": 2.975,
            "min": 0.025,
            "max": 550.249,
            "std": 7.022001802693392
        },
        "sample_rate_distribution": [
            {"rate": 44100.0, "count": 76147},
            {"rate": 16000.0, "count": 48367},
            {"rate": 48000.0, "count": 4099},
            {"rate": 96000.0, "count": 620},
            {"rate": 24000.0, "count": 82},
            {"rate": 22050.0, "count": 45},
            {"rate": 11025.0, "count": 39},
            {"rate": 192000.0, "count": 17},
            {"rate": 8000.0, "count": 12},
            {"rate": 11024.0, "count": 7},
            {"rate": 32000.0, "count": 4}
        ]
    }


from fastapi import APIRouter

router = APIRouter(
    prefix="/dashboard"
)

@router.get("/stats")
def stats():

    return {
        "total_files": 1200,
        "classes": 5,
        "model_accuracy": 0.94
    }


@router.get("/eda")
def eda():

    return {
        "distribution": [
            100,
            200,
            300
        ]
    }
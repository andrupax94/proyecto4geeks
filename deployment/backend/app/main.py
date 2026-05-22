from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.prediction_routes import router as prediction_router
from app.routes.dashboard_routes import router as dashboard_router
# from app.routes.audio_routes import router as audio_router
# from app.routes.model_routes import router as model_router

app = FastAPI(title="MIVIA API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(prediction_router)
app.include_router(dashboard_router)
# app.include_router(audio_router)
# app.include_router(model_router)

@app.get("/")
def home():
    return {
        "message": "MIVIA Backend Running"
    }
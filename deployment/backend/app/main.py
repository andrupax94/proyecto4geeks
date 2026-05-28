from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from app.routes.prediction_routes import router as prediction_router
    from app.routes.dashboard_routes import router as dashboard_router
    from app.routes.github_commits import router as github_router
    from app.routes.model_routes import router as model_router
except ImportError:
    try:
        from .routes.prediction_routes import router as prediction_router
        from .routes.dashboard_routes import router as dashboard_router
        from .routes.github_commits import router as github_router
        from .routes.model_routes import router as model_router
    except ImportError:
        from deployment.backend.app.routes.prediction_routes import router as prediction_router
        from deployment.backend.app.routes.dashboard_routes import router as dashboard_router
        from deployment.backend.app.routes.github_commits import router as github_router
        from deployment.backend.app.routes.model_routes import router as model_router

app = FastAPI(title="MIVIA API")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://andreseduardo.ddns.net:80",
    "http://andreseduardo.ddns.net",
    "https://andreseduardo.ddns.net:443",
    "https://andreseduardo.ddns.net",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(github_router)
app.include_router(prediction_router)
app.include_router(dashboard_router)
app.include_router(model_router)

@app.get("/")
def home():
    return {"message": "MIVIA API is active"}

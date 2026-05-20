from fastapi import FastAPI

app = FastAPI(title="MIVIA API")

@app.get("/")
def home():
    return {
        "message": "MIVIA Backend Running"
    }
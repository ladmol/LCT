from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import extract

app = FastAPI(
    title="Vehicle Re-ID API",
    version="1.0.0",
    description="Создание визуального цифрового признака автомобиля без номеров.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(extract.router, prefix="/api")


@app.get("/api/health", tags=["service"])
def health() -> dict[str, str | int | bool]:
    return {
        "status": "ok",
        "model_version": extract.MODEL_VERSION,
        "embedding_dim": 2816,
        "weights_available": extract.weights_available(),
    }

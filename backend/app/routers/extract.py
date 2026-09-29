from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError

MODEL_VERSION = "vehicle-reid-ensemble-256-v1"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_DETECTIONS = 32
REPO_ROOT = Path(__file__).resolve().parents[3]
ML_ROOT = REPO_ROOT / "ml"
FIRST_WEIGHT = ML_ROOT / "weights" / "convnext_tiny_256_fp16.pt"
SECOND_WEIGHT = ML_ROOT / "weights" / "resnet50_256_fp16.pt"

router = APIRouter(tags=["embedding"])


class DetectionRequest(BaseModel):
    detection_id: str = Field(min_length=1, max_length=128)
    bbox_xywh: tuple[int, int, int, int]


class ExtractMeta(BaseModel):
    event_id: str = Field(min_length=1, max_length=128)
    camera_id: str = Field(min_length=1, max_length=128)
    captured_at: str | None = None
    detections: list[DetectionRequest] = Field(min_length=1, max_length=MAX_DETECTIONS)


class DetectionEmbedding(BaseModel):
    detection_id: str
    embedding: list[float]
    embedding_norm: float


class ExtractResponse(BaseModel):
    event_id: str
    camera_id: str
    model_version: str
    embedding_dim: int
    inference_ms: float
    detections: list[DetectionEmbedding]


def weights_available() -> bool:
    return all(
        path.is_file() and path.stat().st_size > 1_000_000
        for path in (FIRST_WEIGHT, SECOND_WEIGHT)
    )


@lru_cache(maxsize=1)
def get_embedder():
    if not weights_available():
        raise RuntimeError("Deployment weights are missing. Run git lfs pull.")
    ml_path = str(ML_ROOT)
    if ml_path not in sys.path:
        sys.path.insert(0, ml_path)
    from reid.infer import VehicleEnsembleEmbedder  # pyrefly: ignore [missing-import]

    return VehicleEnsembleEmbedder(FIRST_WEIGHT, SECOND_WEIGHT, tta_flip=True)


def parse_meta(raw: str) -> ExtractMeta:
    try:
        return ExtractMeta.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_META", "message": str(error)},
        ) from error


@router.post("/extract", response_model=ExtractResponse)
async def extract(
    image: Annotated[UploadFile, File(description="JPEG or PNG frame")],
    meta: Annotated[str, Form(description="JSON matching ExtractMeta")],
) -> ExtractResponse:
    request = parse_meta(meta)
    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_IMAGE", "message": "Image is empty"},
        )
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "IMAGE_TOO_LARGE",
                "message": "Maximum image size is 20 MiB",
            },
        )

    bboxes = [item.bbox_xywh for item in request.detections]
    started = time.perf_counter()
    try:
        vectors = get_embedder().embed_many(image_bytes, bboxes)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_BBOX", "message": str(error)},
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail={"code": "MODEL_UNAVAILABLE", "message": str(error)},
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_IMAGE", "message": str(error)},
        ) from error

    elapsed_ms = (time.perf_counter() - started) * 1000
    detections = [
        DetectionEmbedding(
            detection_id=item.detection_id,
            embedding=vector.astype(np.float32).tolist(),
            embedding_norm=float(np.linalg.norm(vector)),
        )
        for item, vector in zip(request.detections, vectors, strict=True)
    ]
    return ExtractResponse(
        event_id=request.event_id,
        camera_id=request.camera_id,
        model_version=MODEL_VERSION,
        embedding_dim=int(vectors.shape[1]),
        inference_ms=round(elapsed_ms, 1),
        detections=detections,
    )

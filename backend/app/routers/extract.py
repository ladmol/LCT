from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

router = APIRouter(tags=["extract"])


class ExtractMeta(BaseModel):
    camera_id: str | None = None
    timestamp: str | None = None
    plate_number: str | None = None
    plate_confidence: float | None = None


class Attributes(BaseModel):
    color: str
    body_type: str
    viewpoint: str


class Plate(BaseModel):
    text: str | None = None
    confidence: float | None = None
    source: str | None = None


class Detection(BaseModel):
    bbox: list[int]
    confidence: float
    attributes: Attributes
    plate: Plate
    matched_existing_cluster: bool


class ExtractResponse(BaseModel):
    event_id: str
    cluster_id: str
    detections: list[Detection]


@router.post("/extract", response_model=ExtractResponse)
async def extract(
    image: Annotated[UploadFile, File()],
    meta: Annotated[ExtractMeta, Form()],
):
    return ExtractResponse(
        event_id="mock-event-id",
        cluster_id="mock-cluster-id",
        detections=[
            Detection(
                bbox=[0, 0, 100, 100],
                confidence=0.93,
                attributes=Attributes(color="black", body_type="sedan", viewpoint="rear"),
                plate=Plate(
                    text=meta.plate_number,
                    confidence=meta.plate_confidence,
                    source="camera" if meta.plate_number is not None else None,
                ),
                matched_existing_cluster=True,
            )
        ],
    )

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

router = APIRouter(tags=["search"])


class SearchQuery(BaseModel):
    plate_number: str | None = None
    fingerprint: list[float] | None = None
    top_k: int | None = None


class Attributes(BaseModel):
    color: str
    body_type: str
    viewpoint: str


class SuggestedPlate(BaseModel):
    text: str
    confidence: float
    source_event_id: str


class Candidate(BaseModel):
    event_id: str
    cluster_id: str
    similarity: float
    rerank_score: float
    camera_id: str
    timestamp: str
    attributes: Attributes
    suggested_plate: SuggestedPlate | None = None


class SearchResponse(BaseModel):
    candidates: list[Candidate]


@router.post("/search", response_model=SearchResponse)
async def search(
    query: Annotated[SearchQuery, Form()],
    image: Annotated[UploadFile | None, File()] = None,
):
    return SearchResponse(
        candidates=[
            Candidate(
                event_id="mock-event-id",
                cluster_id="mock-cluster-id",
                similarity=0.912,
                rerank_score=0.884,
                camera_id="cam-7",
                timestamp="2026-09-01T10:42:11Z",
                attributes=Attributes(color="black", body_type="sedan", viewpoint="front"),
                suggested_plate=SuggestedPlate(
                    text="A123BC77", confidence=0.91, source_event_id="mock-event-id"
                ),
            )
        ]
    )

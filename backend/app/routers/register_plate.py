from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["register_plate"])


class RegisterPlateRequest(BaseModel):
    cluster_id: str
    plate_number: str
    confidence: float = 1.0
    source: str = "manual"


class RegisterPlateResponse(BaseModel):
    cluster_id: str
    plate_number: str


@router.post("/register_plate", response_model=RegisterPlateResponse)
async def register_plate(request: RegisterPlateRequest):
    return RegisterPlateResponse(cluster_id=request.cluster_id, plate_number=request.plate_number)

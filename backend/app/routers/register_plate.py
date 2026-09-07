from fastapi import APIRouter

router = APIRouter(tags=["register_plate"])


@router.post("/register_plate")
async def register_plate():
    return {"cluster_id": "mock-cluster-id", "plate_number": "mock-plate-number"}

from fastapi import APIRouter

router = APIRouter(tags=["search"])


@router.post("/search")
async def search():
    return {"candidates": []}

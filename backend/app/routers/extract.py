from fastapi import APIRouter

router = APIRouter(tags=["extract"])


@router.post("/extract")
async def extract():
    return {"event_id": "mock-event-id", "cluster_id": "mock-cluster-id", "detections": []}

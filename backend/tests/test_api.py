import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def test_health_reports_deployment_weights() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["weights_available"] is True
    assert response.json()["embedding_dim"] == 2816


def test_extract_returns_normalized_embedding() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (96, 64), (70, 110, 150)).save(buffer, format="JPEG")
    meta = {
        "event_id": "test-event",
        "camera_id": "test-camera",
        "detections": [{"detection_id": "vehicle", "bbox_xywh": [0, 0, 96, 64]}],
    }
    response = client.post(
        "/api/extract",
        files={"image": ("vehicle.jpg", buffer.getvalue(), "image/jpeg")},
        data={"meta": json.dumps(meta)},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["embedding_dim"] == 2816
    assert len(body["detections"][0]["embedding"]) == 2816
    assert abs(body["detections"][0]["embedding_norm"] - 1.0) < 1e-4

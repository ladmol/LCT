from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# COCO class ids for the vehicle types we care about (car, bus, truck).
VEHICLE_CLASS_IDS = {2, 5, 7}

WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "yolov10n.pt"


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    crop: np.ndarray


class VehicleDetector:
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, conf: float = 0.25, device: str | None = None):
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device

    def detect(self, image: np.ndarray) -> list[Detection]:
        results = self.model.predict(image, conf=self.conf, device=self.device, verbose=False)[0]
        detections = []
        for box in results.boxes:
            class_id = int(box.cls.item())
            if class_id not in VEHICLE_CLASS_IDS:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf.item()),
                    class_id=class_id,
                    class_name=results.names[class_id],
                    crop=image[y1:y2, x1:x2],
                )
            )
        return detections


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import cv2

    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        from ultralytics.utils import ASSETS

        image_path = ASSETS / "bus.jpg"

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    detector = VehicleDetector()
    detections = detector.detect(image)

    print(f"{len(detections)} vehicle(s) detected in {image_path}")
    for i, det in enumerate(detections):
        print(f"  [{i}] {det.class_name} conf={det.confidence:.2f} bbox={det.bbox}")
        cv2.imwrite(f"crop_{i}.jpg", det.crop)

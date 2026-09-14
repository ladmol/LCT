"""Compare two YOLO checkpoints on the same image.

Dev-convenience tool, not part of the pipeline itself — see ml/detection/ for that.

Inference runs sequentially — one model finishes completely before the next one
starts — so the timing comparison is fair (no shared/parallel compute). Only
once every model has been run do their windows get shown, together.

Usage:
    uv run python tools/compare_models.py path/to/image.jpg
    uv run python tools/compare_models.py path/to/image.jpg yolov10n.pt yolov10s.pt
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from ultralytics import YOLO

from detection.detector import WEIGHTS_DIR
from tools.playground import predict, window_title

LOG_PATH = Path(__file__).resolve().parent.parent / "detection.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def wait_for_close_all(window_names: list[str], poll_ms: int = 50) -> None:
    """Block until every window has been closed (q/Esc, or each window's own [x])."""
    while any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1 for name in window_names):
        if cv2.waitKey(poll_ms) & 0xFF in (ord("q"), 27):
            break
    for name in window_names:
        cv2.destroyWindow(name)


def main(image_path: str, weights_names: list[str]) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(image_path)

    window_names = []
    for weights_name in weights_names:
        model = YOLO(WEIGHTS_DIR / weights_name)  # loaded and run one at a time, never concurrently
        results = predict(model, frame)

        detections = ", ".join(
            f"{results.names[int(box.cls.item())]} {float(box.conf.item()) * 100:.2f}%"
            for box in results.boxes or []
        ) or "no detections"
        speed = results.speed
        logger.info(
            "%s | image=%s | %s | preprocess=%.2fms inference=%.2fms postprocess=%.2fms",
            weights_name,
            image_path,
            detections,
            speed["preprocess"],
            speed["inference"],
            speed["postprocess"],
        )

        window_name = f"{weights_name} | {window_title(results)}"
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, results.plot())
        window_names.append(window_name)
        del model

    wait_for_close_all(window_names)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: compare_models.py path/to/image.jpg [weights1.pt weights2.pt ...]")

    image_path = sys.argv[1]
    weights_names = sys.argv[2:] or ["yolov10n.pt", "yolov10s.pt"]
    main(image_path, weights_names)

"""Batch-benchmark YOLO checkpoints over a folder of images (no windows, log only).

Dev-convenience tool, not part of the pipeline itself — see ml/detection/ for that.

Loads each model once, runs it over every image in the folder sequentially (never
overlapping with the other model), and logs one line per (model, image) into the
same detection.log used by compare_models.py.

Usage:
    uv run python tools/benchmark.py ../data/raw/test_images
    uv run python tools/benchmark.py ../data/raw/test_images yolov10n.pt yolov10s.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from ultralytics import YOLO

from detection.detector import WEIGHTS_DIR
from tools.compare_models import logger
from tools.playground import predict


def benchmark(image_dir: Path, weights_names: list[str]) -> None:
    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )
    if not image_paths:
        raise SystemExit(f"No images found in {image_dir}")

    for weights_name in weights_names:
        model = YOLO(WEIGHTS_DIR / weights_name)  # one model at a time, never concurrently
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                logger.warning("%s | image=%s | failed to read image", weights_name, image_path)
                continue

            results = predict(model, frame)
            detections = ", ".join(
                f"{results.names[int(box.cls.item())]} {float(box.conf.item()) * 100:.2f}%"
                for box in results.boxes or []
            ) or "no detections"
            speed = results.speed
            logger.info(
                "%s | image=%s | %s | preprocess=%.2fms inference=%.2fms postprocess=%.2fms",
                weights_name,
                image_path.name,
                detections,
                speed["preprocess"],
                speed["inference"],
                speed["postprocess"],
            )
        del model


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: benchmark.py path/to/image_dir [weights1.pt weights2.pt ...]")

    image_dir = Path(sys.argv[1])
    weights_names = sys.argv[2:] or ["yolov10n.pt", "yolov10s.pt"]
    benchmark(image_dir, weights_names)

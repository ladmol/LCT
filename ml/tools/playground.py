"""Interactive playground for VehicleDetector — pops up a window with boxes + labels + confidence.

Dev-convenience tool, not part of the pipeline itself — see ml/detection/ for that.

The window title shows the best detection's confidence + COCO class, plus inference time.

Usage:
    uv run python tools/playground.py                  # webcam, live
    uv run python tools/playground.py path/to/image.jpg # single image
    uv run python tools/playground.py path/to/video.mp4 # video file, live
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

from detection.detector import DEFAULT_WEIGHTS, VEHICLE_CLASS_IDS

WINDOW_NAME = "Vehicle detector (q or Esc to quit)"


def predict(model: YOLO, frame) -> Results:
    return model.predict(frame, conf=0.25, classes=list(VEHICLE_CLASS_IDS), verbose=False)[0]


def window_title(results: Results) -> str:
    inference_ms = results.speed["inference"]
    if len(results.boxes) == 0:
        return f"No vehicle | {inference_ms:.0f} ms"

    best = max(results.boxes, key=lambda b: float(b.conf.item()))
    class_name = results.names[int(best.cls.item())]
    confidence = float(best.conf.item()) * 100
    return f"{confidence:.0f}% {class_name} | {inference_ms:.0f} ms"


def show(results: Results, window_name: str = WINDOW_NAME) -> None:
    cv2.imshow(window_name, results.plot())
    cv2.setWindowTitle(window_name, window_title(results))


def wait_for_close(window_name: str = WINDOW_NAME, poll_ms: int = 50) -> None:
    """Block until the user presses q/Esc OR closes the window via its [x] button."""
    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKey(poll_ms) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyWindow(window_name)


def run_on_image(model: YOLO, path: str, window_name: str = WINDOW_NAME) -> None:
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(path)
    cv2.namedWindow(window_name)
    show(predict(model, frame), window_name)
    wait_for_close(window_name)


def run_on_stream(model: YOLO, source: int | str, window_name: str = WINDOW_NAME) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")
    cv2.namedWindow(window_name)
    print("Press 'q' or Esc to quit")
    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        ok, frame = cap.read()
        if not ok:
            break
        show(predict(model, frame), window_name)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    yolo = YOLO(DEFAULT_WEIGHTS)
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg is None:
        run_on_stream(yolo, 0)  # default webcam
    elif arg.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        run_on_image(yolo, arg)
    else:
        run_on_stream(yolo, arg)  # video file path

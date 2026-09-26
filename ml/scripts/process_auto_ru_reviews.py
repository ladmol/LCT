"""Create plate/face-redacted vehicle crops from manually reviewed raw photos.

Review coordinates refer to original pixels. Only rows explicitly approved and
marked as checked for plates and faces can enter approved.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def box_xyxy(box, size):
    if len(box) != 4:
        raise ValueError(f"Expected xywh box, got {box}")
    x, y, w, h = map(int, box)
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > size[0] or y + h > size[1]:
        raise ValueError(f"Box {box} is outside image {size}")
    return x, y, x + w, y + h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/auto_ru"))
    parser.add_argument("--reviews", type=Path, default=Path("data/auto_ru/reviews.jsonl"))
    args = parser.parse_args()
    source = args.data / "manifest.jsonl"
    if not source.is_file() or not args.reviews.is_file():
        raise FileNotFoundError("Both manifest.jsonl and reviews.jsonl are required")
    manifest = {row["image_path"]: row for row in map(json.loads, source.read_text(encoding="utf-8").splitlines())}
    reviews = list(map(json.loads, args.reviews.read_text(encoding="utf-8").splitlines()))
    approved = []
    seen = set()
    for review in reviews:
        if review.get("status") != "approved":
            continue
        if review["image_path"] in seen:
            raise ValueError(f"Duplicate approved review: {review['image_path']}")
        seen.add(review["image_path"])
        if review.get("plate_reviewed") is not True or review.get("face_reviewed") is not True:
            raise ValueError(f"Explicit plate/face review missing: {review.get('image_path')}")
        row = manifest[review["image_path"]]
        restyling = review.get("restyling_label", row["restyling_label"])
        if restyling not in {"unknown", "pre-restyling", "restyling"}:
            raise ValueError(f"Unexpected restyling label: {restyling}")
        raw_path = (args.data / row["image_path"]).resolve()
        if not raw_path.is_relative_to((args.data / "raw").resolve()):
            raise ValueError(f"Image is outside raw quarantine: {raw_path}")
        with Image.open(raw_path) as original:
            image = original.convert("RGB")
        vehicle_box = box_xyxy(review["vehicle_bbox"], image.size)
        draw = ImageDraw.Draw(image)
        for box in review.get("redactions", []):
            draw.rectangle(box_xyxy(box, image.size), fill=(0, 0, 0))
        cropped = image.crop(vehicle_box)
        destination = args.data / "approved" / row["listing_id"] / Path(row["image_path"]).name
        if not destination.resolve().is_relative_to((args.data / "approved").resolve()):
            raise ValueError(f"Approved destination is outside approved/: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(destination, format="JPEG", quality=95)
        approved.append(
            {
                "instance_id": row["listing_id"],
                "image_path": str(destination.relative_to(args.data)).replace("\\", "/"),
                "make": review.get("make", row["make"]).lower(),
                "model": review.get("model", row["model"]).lower(),
                "catalog_name": row["catalog_name"],
                "restyling_label": restyling,
                "source_url": row["source_url"],
                "plate_reviewed": True,
                "face_reviewed": True,
            }
        )
    output = args.data / "approved.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in approved), encoding="utf-8")
    print(f"Approved {len(approved)} photos; manifest={output}")


if __name__ == "__main__":
    main()

"""Render one contact sheet per listing for manual Auto.ru photo review."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/auto_ru"))
    parser.add_argument("--output", type=Path, default=Path("outputs/auto_ru_review"))
    args = parser.parse_args()
    manifest = args.data / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    grouped = defaultdict(list)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        grouped[row["listing_id"]].append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    for listing_id, items in grouped.items():
        width, height = 400, 320
        columns = 2
        rows = (len(items) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * width, rows * height + 44), "#e9e9e9")
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 10), f"{listing_id} | {items[0]['make']} {items[0]['model']}", fill="black")
        for index, row in enumerate(items):
            path = (args.data / row["image_path"]).resolve()
            if not path.is_relative_to((args.data / "raw").resolve()):
                raise ValueError(f"Raw path outside quarantine: {path}")
            with Image.open(path) as source:
                thumb = ImageOps.contain(source.convert("RGB"), (width - 12, height - 36))
            x, y = index % columns * width, index // columns * height + 44
            sheet.paste(thumb, (x + (width - thumb.width) // 2, y + 5))
            draw.text((x + 8, y + height - 26), Path(row["image_path"]).name, fill="black")
        sheet.save(args.output / f"{listing_id}.jpg", quality=90)
    print(f"Created {len(grouped)} review sheets in {args.output}")


if __name__ == "__main__":
    main()

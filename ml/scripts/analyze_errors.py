"""Build a reproducible error report and contact sheet for a Re-ID checkpoint."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from reid.data import VehicleRecord, crop_vehicle, read_records
from reid.infer import embed_records, load_model
from reid.protocol import cross_camera_protocol, split_identities


def rank_rows(
    query_vectors: np.ndarray,
    gallery_vectors: np.ndarray,
    query: list[VehicleRecord],
    gallery: list[VehicleRecord],
) -> list[dict[str, object]]:
    scores = query_vectors @ gallery_vectors.T
    gallery_ids = [record.vehicle_id for record in gallery]
    rows: list[dict[str, object]] = []
    for index, (record, score) in enumerate(zip(query, scores)):
        order = np.argsort(-score, kind="stable")
        top = int(order[0])
        matches = [
            position
            for position, candidate in enumerate(order, start=1)
            if gallery_ids[int(candidate)] == record.vehicle_id
        ]
        rank = matches[0] if matches else None
        true_index = int(order[rank - 1]) if rank is not None else None
        true_score = float(score[true_index]) if true_index is not None else None
        rows.append(
            {
                "query_index": index,
                "query_id": record.image_id,
                "vehicle_id": record.vehicle_id,
                "query_camera": record.camera_id,
                "bbox_width": record.bbox[2],
                "bbox_height": record.bbox[3],
                "bbox_area": record.bbox[2] * record.bbox[3],
                "bbox_aspect": record.bbox[2] / record.bbox[3],
                "known": rank is not None,
                "rank": rank,
                "top1_id": gallery[top].image_id,
                "top1_vehicle_id": gallery[top].vehicle_id,
                "top1_camera": gallery[top].camera_id,
                "top1_score": float(score[top]),
                "true_gallery_id": gallery[true_index].image_id
                if true_index is not None
                else None,
                "true_gallery_camera": gallery[true_index].camera_id
                if true_index is not None
                else None,
                "true_score": true_score,
                "wrong_margin": float(score[top] - true_score)
                if rank is not None and rank > 1
                else 0.0,
            }
        )
    return rows


def _ranking_summary(rows: list[dict[str, object]]) -> dict[str, float | int]:
    known = [row for row in rows if row["known"]]
    ranks = np.array([int(row["rank"]) for row in known])
    return {
        "queries": len(rows),
        "known_queries": len(known),
        "unknown_queries": len(rows) - len(known),
        "rank1": float(np.mean(ranks <= 1)),
        "rank5": float(np.mean(ranks <= 5)),
        "rank10": float(np.mean(ranks <= 10)),
        "miss_top10": int(np.sum(ranks > 10)),
        "median_rank": float(np.median(ranks)),
    }


def _group_summary(
    rows: list[dict[str, object]], field: str, min_size: int = 1
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["known"]:
            groups[str(row[field])].append(row)
    result = {}
    for name, group in sorted(groups.items()):
        if len(group) < min_size:
            continue
        ranks = np.array([int(row["rank"]) for row in group])
        result[name] = {
            "queries": len(group),
            "rank1": float(np.mean(ranks <= 1)),
            "rank5": float(np.mean(ranks <= 5)),
            "median_rank": float(np.median(ranks)),
        }
    return result


def add_bbox_buckets(rows: list[dict[str, object]]) -> list[float]:
    areas = np.array([float(row["bbox_area"]) for row in rows])
    boundaries = [float(value) for value in np.quantile(areas, [0.25, 0.5, 0.75])]
    labels = ("smallest", "small", "large", "largest")
    for row in rows:
        row["bbox_bucket"] = labels[
            int(np.searchsorted(boundaries, float(row["bbox_area"]), side="right"))
        ]
        aspect = float(row["bbox_aspect"])
        row["aspect_bucket"] = (
            "tall" if aspect < 1 else "wide" if aspect > 2 else "normal"
        )
        if row["known"]:
            row["camera_pair"] = (
                f"{row['query_camera']} -> {row['true_gallery_camera']}"
            )
        else:
            row["camera_pair"] = "unknown"
    return boundaries


def _load_crop(
    source: zipfile.ZipFile, record: VehicleRecord, size: tuple[int, int]
) -> Image.Image:
    with (
        source.open(f"images/{record.image_id}.jpg") as raw,
        Image.open(io.BytesIO(raw.read())) as image,
    ):
        crop = crop_vehicle(image, record.bbox)
    crop.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(crop, ((size[0] - crop.width) // 2, (size[1] - crop.height) // 2))
    return canvas


def make_contact_sheet(
    archive: Path,
    rows: list[dict[str, object]],
    query: list[VehicleRecord],
    gallery: list[VehicleRecord],
    output: Path,
    limit: int,
) -> int:
    query_by_image = {record.image_id: record for record in query}
    gallery_by_image = {record.image_id: record for record in gallery}
    mistakes = sorted(
        (row for row in rows if row["known"] and int(row["rank"]) > 1),
        key=lambda row: (float(row["wrong_margin"]), int(row["rank"])),
        reverse=True,
    )[:limit]
    tile_w, tile_h, header = 240, 180, 38
    sheet = Image.new(
        "RGB", (tile_w * 3, (tile_h + header) * len(mistakes)), (238, 238, 238)
    )
    draw = ImageDraw.Draw(sheet)
    with zipfile.ZipFile(archive) as source:
        for row_index, row in enumerate(mistakes):
            y = row_index * (tile_h + header)
            records = (
                query_by_image[str(row["query_id"])],
                gallery_by_image[str(row["top1_id"])],
                gallery_by_image[str(row["true_gallery_id"])],
            )
            titles = (
                f"QUERY cam={row['query_camera']}",
                f"WRONG top1 {float(row['top1_score']):.3f}",
                f"TRUE rank={row['rank']} {float(row['true_score']):.3f}",
            )
            for column, (record, title) in enumerate(zip(records, titles)):
                x = column * tile_w
                sheet.paste(
                    _load_crop(source, record, (tile_w, tile_h)), (x, y + header)
                )
                draw.text((x + 5, y + 5), title, fill=(0, 0, 0))
                draw.text((x + 5, y + 20), record.image_id[:28], fill=(50, 50, 50))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    return len(mistakes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/convnext_tiny_256/best.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/error_analysis_resnet50_256")
    )
    parser.add_argument("--split", choices=("dev", "holdout"), default="holdout")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sheet-size", type=int, default=30)
    args = parser.parse_args()

    model, metadata, device = load_model(args.checkpoint, args.device)
    records = read_records(args.archive, "train.csv")
    splits = split_identities(records, int(metadata.get("seed", 42)))
    seed_offset = 1 if args.split == "dev" else 2
    query, gallery = cross_camera_protocol(
        records, splits[args.split], int(metadata.get("seed", 42)) + seed_offset
    )
    query_vectors = embed_records(
        model,
        args.archive,
        query,
        metadata["image_size"],
        device,
        args.batch_size,
        args.workers,
    )
    gallery_vectors = embed_records(
        model,
        args.archive,
        gallery,
        metadata["image_size"],
        device,
        args.batch_size,
        args.workers,
    )
    rows = rank_rows(query_vectors, gallery_vectors, query, gallery)
    boundaries = add_bbox_buckets(rows)
    args.output.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0])
    with (args.output / "queries.csv").open("w", encoding="utf-8", newline="") as raw:
        writer = csv.DictWriter(raw, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    sheet_rows = make_contact_sheet(
        args.archive,
        rows,
        query,
        gallery,
        args.output / "confident_mistakes.jpg",
        args.sheet_size,
    )
    unknown_scores = [float(row["top1_score"]) for row in rows if not row["known"]]
    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "ranking": _ranking_summary(rows),
        "bbox_area_quartiles": boundaries,
        "by_bbox_size": _group_summary(rows, "bbox_bucket"),
        "by_aspect": _group_summary(rows, "aspect_bucket"),
        "by_query_camera": _group_summary(rows, "query_camera", min_size=5),
        "by_camera_pair": _group_summary(rows, "camera_pair", min_size=5),
        "unknown_top1_score": {
            "median": float(np.median(unknown_scores)),
            "p90": float(np.quantile(unknown_scores, 0.9)),
            "max": float(np.max(unknown_scores)),
        },
        "contact_sheet_rows": sheet_rows,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

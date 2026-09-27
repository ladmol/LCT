"""Compare deterministic inference-time ensembles on the fixed Re-ID protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from eval.metrics import choose_refusal_threshold, refusal_metrics, retrieval_metrics
from reid.data import ContestDataset, VehicleRecord, read_records
from reid.infer import load_model
from reid.protocol import cross_camera_protocol, split_identities
from scripts.analyze_errors import add_bbox_buckets, rank_rows


@torch.inference_mode()
def embed_variants(model, archive, records, image_size, device, batch_size, workers):
    loader = DataLoader(
        ContestDataset(archive, records, image_size),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    output = {
        name: [] for name in ("base", "flip", "wide_multicrop", "flip_wide_multicrop")
    }
    for images, _, indices in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            base, _ = model(images)
            flipped, _ = model(torch.flip(images, dims=(3,)))
            flip_average = F.normalize(base + flipped, dim=1)
            wide_average = base.clone()
            combined_average = flip_average.clone()
            wide_positions = [
                position
                for position, source_index in enumerate(indices.tolist())
                if records[source_index].bbox[2] / records[source_index].bbox[3] > 2
            ]
            if wide_positions:
                selected = images[wide_positions]
                width = selected.shape[3]
                crop_width = round(width * 0.72)
                left = F.interpolate(
                    selected[:, :, :, :crop_width],
                    size=(image_size, image_size),
                    mode="bilinear",
                )
                right = F.interpolate(
                    selected[:, :, :, -crop_width:],
                    size=(image_size, image_size),
                    mode="bilinear",
                )
                left_vector, _ = model(left)
                right_vector, _ = model(right)
                wide_average[wide_positions] = F.normalize(
                    base[wide_positions] + left_vector + right_vector, dim=1
                )
                combined_average[wide_positions] = F.normalize(
                    base[wide_positions]
                    + flipped[wide_positions]
                    + left_vector
                    + right_vector,
                    dim=1,
                )
        for name, vectors in (
            ("base", base),
            ("flip", flip_average),
            ("wide_multicrop", wide_average),
            ("flip_wide_multicrop", combined_average),
        ):
            output[name].append(vectors.float().cpu().numpy())
    return {
        name: np.concatenate(parts).astype(np.float32) for name, parts in output.items()
    }


def ranking_by_aspect(query_vectors, gallery_vectors, query, gallery):
    rows = rank_rows(query_vectors, gallery_vectors, query, gallery)
    add_bbox_buckets(rows)
    result = {}
    for aspect in ("normal", "wide", "tall"):
        ranks = [
            int(row["rank"])
            for row in rows
            if row["known"] and row["aspect_bucket"] == aspect
        ]
        result[aspect] = {
            "queries": len(ranks),
            "rank1": float(np.mean(np.array(ranks) <= 1)) if ranks else 0.0,
            "rank5": float(np.mean(np.array(ranks) <= 5)) if ranks else 0.0,
            "median_rank": float(np.median(ranks)) if ranks else 0.0,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/convnext_tiny_256/best.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/inference_variants.json")
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model, metadata, device = load_model(args.checkpoint, args.device)
    records = read_records(args.archive, "train.csv")
    seed = int(metadata.get("seed", 42))
    split = split_identities(records, seed)
    protocols: dict[str, tuple[list[VehicleRecord], list[VehicleRecord]]] = {
        name: cross_camera_protocol(records, split[name], seed + offset)
        for name, offset in (("dev", 1), ("holdout", 2))
    }
    embedded = {}
    for split_name, (query, gallery) in protocols.items():
        embedded[split_name] = {
            "query": embed_variants(
                model,
                args.archive,
                query,
                metadata["image_size"],
                device,
                args.batch_size,
                args.workers,
            ),
            "gallery": embed_variants(
                model,
                args.archive,
                gallery,
                metadata["image_size"],
                device,
                args.batch_size,
                args.workers,
            ),
        }

    report = {}
    for variant in embedded["dev"]["query"]:
        dev_query, dev_gallery = protocols["dev"]
        holdout_query, holdout_gallery = protocols["holdout"]
        dev_q = embedded["dev"]["query"][variant]
        dev_g = embedded["dev"]["gallery"][variant]
        holdout_q = embedded["holdout"]["query"][variant]
        holdout_g = embedded["holdout"]["gallery"][variant]
        dev_qids = [row.vehicle_id for row in dev_query]
        dev_gids = [row.vehicle_id for row in dev_gallery]
        holdout_qids = [row.vehicle_id for row in holdout_query]
        holdout_gids = [row.vehicle_id for row in holdout_gallery]
        threshold = choose_refusal_threshold(
            dev_q, dev_g, dev_qids, dev_gids, "balanced"
        )
        report[variant] = {
            "dev": retrieval_metrics(dev_q, dev_g, dev_qids, dev_gids),
            "holdout": retrieval_metrics(
                holdout_q, holdout_g, holdout_qids, holdout_gids
            ),
            "holdout_refusal_balanced": refusal_metrics(
                holdout_q, holdout_g, holdout_qids, holdout_gids, threshold["threshold"]
            ),
            "holdout_by_aspect": ranking_by_aspect(
                holdout_q, holdout_g, holdout_query, holdout_gallery
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

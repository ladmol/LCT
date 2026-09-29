"""Evaluate multi-frame vehicle descriptors on an identity-disjoint protocol."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval.decision import choose_decision_rule, decision_metrics, decision_values
from eval.metrics import choose_refusal_threshold, refusal_metrics, retrieval_metrics
from reid.data import VehicleRecord, read_records
from reid.infer import combine_embeddings, embed_records, load_model
from reid.protocol import split_identities


@dataclass(frozen=True)
class Tracklet:
    vehicle_id: str
    camera_id: str
    records: tuple[VehicleRecord, ...]


def build_tracklet_protocol(
    records: list[VehicleRecord], identities: list[str], seed: int
) -> tuple[list[Tracklet], list[Tracklet]]:
    """Use one camera as gallery and other cameras as query for known IDs."""
    rng = random.Random(seed)
    shuffled = identities.copy()
    rng.shuffle(shuffled)
    known = set(shuffled[: round(len(shuffled) * 0.75)])
    by_identity: dict[str, dict[str, list[VehicleRecord]]] = {}
    for record in records:
        if record.vehicle_id in shuffled:
            by_identity.setdefault(record.vehicle_id, {}).setdefault(
                record.camera_id or "unknown", []
            ).append(record)

    query, gallery = [], []
    for vehicle_id in shuffled:
        cameras = by_identity.get(vehicle_id, {})
        if not cameras:
            continue
        camera_ids = sorted(cameras)
        if vehicle_id not in known:
            camera_id = rng.choice(camera_ids)
            query.append(Tracklet(vehicle_id, camera_id, tuple(cameras[camera_id])))
            continue
        if len(camera_ids) < 2:
            continue
        gallery_camera = rng.choice(camera_ids)
        gallery.append(
            Tracklet(vehicle_id, gallery_camera, tuple(cameras[gallery_camera]))
        )
        query.extend(
            Tracklet(vehicle_id, camera_id, tuple(cameras[camera_id]))
            for camera_id in camera_ids
            if camera_id != gallery_camera
        )
    if not query or not gallery:
        raise ValueError("Cannot build a cross-camera tracklet protocol")
    return query, gallery


def pool_tracklet(
    vectors: np.ndarray, records: tuple[VehicleRecord, ...], method: str
) -> np.ndarray:
    if method == "first":
        pooled = vectors[0]
    elif method == "mean":
        pooled = vectors.mean(axis=0)
    elif method == "medoid":
        pooled = vectors[(vectors @ vectors.T).sum(axis=1).argmax()]
    elif method == "quality_mean":
        quality = []
        for record in records:
            _, _, width, height = record.bbox
            area = math.sqrt(width * height)
            aspect_penalty = math.exp(-abs(math.log(width / height)))
            quality.append(area * aspect_penalty)
        weights = np.asarray(quality, dtype=np.float32)
        weights /= weights.sum()
        pooled = (vectors * weights[:, None]).sum(axis=0)
    else:
        raise ValueError(f"Unknown pooling method: {method}")
    norm = np.linalg.norm(pooled)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Tracklet pooling produced an invalid vector")
    return (pooled / norm).astype(np.float32)


def embed_tracklets(
    tracklets: list[Tracklet],
    archive: Path,
    checkpoint: Path,
    second_checkpoint: Path | None,
    first_weight: float,
    tta_flip: bool,
    batch_size: int,
    workers: int,
    device_name: str,
) -> dict[str, np.ndarray]:
    records = [record for tracklet in tracklets for record in tracklet.records]
    model, metadata, device = load_model(checkpoint, device_name)
    vectors = embed_records(
        model,
        archive,
        records,
        metadata["image_size"],
        device,
        batch_size,
        workers,
        tta_flip,
    )
    if second_checkpoint is not None:
        second_model, second_metadata, second_device = load_model(
            second_checkpoint, device_name
        )
        second_vectors = embed_records(
            second_model,
            archive,
            records,
            second_metadata["image_size"],
            second_device,
            batch_size,
            workers,
            tta_flip,
        )
        vectors = combine_embeddings(vectors, second_vectors, first_weight)

    methods = ("first", "mean", "medoid", "quality_mean")
    output = {method: [] for method in methods}
    offset = 0
    for tracklet in tracklets:
        end = offset + len(tracklet.records)
        for method in methods:
            output[method].append(
                pool_tracklet(vectors[offset:end], tracklet.records, method)
            )
        offset = end
    return {
        method: np.stack(pooled).astype(np.float32) for method, pooled in output.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--second-checkpoint", type=Path)
    parser.add_argument("--first-weight", type=float, default=0.5)
    parser.add_argument("--tta-flip", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/tracklet_pooling.json")
    )
    args = parser.parse_args()

    records = read_records(args.archive, "train.csv")
    split = split_identities(records, 42)
    protocols = {
        name: build_tracklet_protocol(records, split[name], 42 + offset)
        for name, offset in (("dev", 1), ("holdout", 2))
    }
    embedded = {}
    for split_name, (query, gallery) in protocols.items():
        embedded[split_name] = {
            "query": embed_tracklets(
                query,
                args.archive,
                args.checkpoint,
                args.second_checkpoint,
                args.first_weight,
                args.tta_flip,
                args.batch_size,
                args.workers,
                args.device,
            ),
            "gallery": embed_tracklets(
                gallery,
                args.archive,
                args.checkpoint,
                args.second_checkpoint,
                args.first_weight,
                args.tta_flip,
                args.batch_size,
                args.workers,
                args.device,
            ),
        }

    report = {}
    dev_query, dev_gallery = protocols["dev"]
    holdout_query, holdout_gallery = protocols["holdout"]
    dev_qids = [tracklet.vehicle_id for tracklet in dev_query]
    dev_gids = [tracklet.vehicle_id for tracklet in dev_gallery]
    holdout_qids = [tracklet.vehicle_id for tracklet in holdout_query]
    holdout_gids = [tracklet.vehicle_id for tracklet in holdout_gallery]
    for method in embedded["dev"]["query"]:
        dev_q = embedded["dev"]["query"][method]
        dev_g = embedded["dev"]["gallery"][method]
        threshold = choose_refusal_threshold(
            dev_q, dev_g, dev_qids, dev_gids, "balanced"
        )
        balanced_rule = choose_decision_rule(
            dev_q, dev_g, dev_qids, dev_gids, objective="balanced"
        )
        strict_rule = choose_decision_rule(
            dev_q,
            dev_g,
            dev_qids,
            dev_gids,
            objective="strict",
            target_precision=0.95,
            target_tnr=0.95,
        )
        conservative_rule = choose_decision_rule(
            dev_q,
            dev_g,
            dev_qids,
            dev_gids,
            objective="strict",
            target_precision=0.90,
            target_tnr=0.90,
        )
        holdout_q = embedded["holdout"]["query"][method]
        holdout_g = embedded["holdout"]["gallery"][method]
        holdout_values = decision_values(
            holdout_q, holdout_g, holdout_qids, holdout_gids
        )
        report[method] = {
            "dev": retrieval_metrics(dev_q, dev_g, dev_qids, dev_gids),
            "holdout": retrieval_metrics(
                holdout_q, holdout_g, holdout_qids, holdout_gids
            ),
            "holdout_refusal_balanced": refusal_metrics(
                holdout_q,
                holdout_g,
                holdout_qids,
                holdout_gids,
                threshold["threshold"],
            ),
            "holdout_margin_balanced": decision_metrics(
                holdout_values,
                float(balanced_rule["similarity_threshold"]),
                float(balanced_rule["margin_threshold"]),
            ),
            "strict_rule_dev": strict_rule,
            "conservative_rule_dev": conservative_rule,
            "holdout_conservative": decision_metrics(
                holdout_values,
                float(conservative_rule["similarity_threshold"]),
                float(conservative_rule["margin_threshold"]),
            ),
            "holdout_strict": decision_metrics(
                holdout_values,
                float(strict_rule["similarity_threshold"]),
                float(strict_rule["margin_threshold"]),
            ),
        }
    report["protocol"] = {
        "dev_query_tracklets": len(dev_query),
        "dev_gallery_tracklets": len(dev_gallery),
        "holdout_query_tracklets": len(holdout_query),
        "holdout_gallery_tracklets": len(holdout_gallery),
        "note": "Each tracklet contains up to three images from one camera.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

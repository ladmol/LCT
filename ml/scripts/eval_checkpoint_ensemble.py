"""Select and evaluate a two-checkpoint feature ensemble on the fixed protocol."""

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
from reid.infer import combine_embeddings, load_model
from reid.protocol import cross_camera_protocol, split_identities


@torch.inference_mode()
def embed_base_and_flip(
    model, archive, records, image_size, device, batch_size, workers
):
    loader = DataLoader(
        ContestDataset(archive, records, image_size),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    base_parts, flip_parts = [], []
    for images, _, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            base, _ = model(images)
            flipped, _ = model(torch.flip(images, dims=(3,)))
            flip = F.normalize(base + flipped, dim=1)
        base_parts.append(base.float().cpu().numpy())
        flip_parts.append(flip.float().cpu().numpy())
    return {
        "base": np.concatenate(base_parts).astype(np.float32),
        "flip": np.concatenate(flip_parts).astype(np.float32),
    }


def identities(records: list[VehicleRecord]) -> list[str | None]:
    return [record.vehicle_id for record in records]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/checkpoint_ensemble.json")
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    first_model, first_meta, device = load_model(args.first, args.device)
    second_model, second_meta, second_device = load_model(args.second, args.device)
    if device != second_device:
        raise ValueError("Both checkpoints must run on the same device")

    records = read_records(args.archive, "train.csv")
    seed = int(first_meta.get("seed", 42))
    if int(second_meta.get("seed", seed)) != seed:
        raise ValueError("Checkpoints use different protocol seeds")
    split = split_identities(records, seed)
    protocols = {
        name: cross_camera_protocol(records, split[name], seed + offset)
        for name, offset in (("dev", 1), ("holdout", 2))
    }
    embedded: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    for split_name, (query, gallery) in protocols.items():
        embedded[split_name] = {}
        for side, rows in (("query", query), ("gallery", gallery)):
            embedded[split_name][side] = {
                "first": embed_base_and_flip(
                    first_model,
                    args.archive,
                    rows,
                    first_meta["image_size"],
                    device,
                    args.batch_size,
                    args.workers,
                ),
                "second": embed_base_and_flip(
                    second_model,
                    args.archive,
                    rows,
                    second_meta["image_size"],
                    device,
                    args.batch_size,
                    args.workers,
                ),
            }

    dev_query, dev_gallery = protocols["dev"]
    dev_qids, dev_gids = identities(dev_query), identities(dev_gallery)
    best_config: tuple[str, str, float, dict[str, float]] | None = None
    for first_mode in ("base", "flip"):
        for second_mode in ("base", "flip"):
            for first_weight in np.linspace(0.1, 0.9, 9):
                dev_q = combine_embeddings(
                    embedded["dev"]["query"]["first"][first_mode],
                    embedded["dev"]["query"]["second"][second_mode],
                    float(first_weight),
                )
                dev_g = combine_embeddings(
                    embedded["dev"]["gallery"]["first"][first_mode],
                    embedded["dev"]["gallery"]["second"][second_mode],
                    float(first_weight),
                )
                metrics = retrieval_metrics(dev_q, dev_g, dev_qids, dev_gids)
                config = (first_mode, second_mode, float(first_weight), metrics)
                if best_config is None or (metrics["mAP"], metrics["rank1"]) > (
                    best_config[3]["mAP"],
                    best_config[3]["rank1"],
                ):
                    best_config = config
    assert best_config is not None
    first_mode, second_mode, first_weight, dev_metrics = best_config
    dev_q = combine_embeddings(
        embedded["dev"]["query"]["first"][first_mode],
        embedded["dev"]["query"]["second"][second_mode],
        first_weight,
    )
    dev_g = combine_embeddings(
        embedded["dev"]["gallery"]["first"][first_mode],
        embedded["dev"]["gallery"]["second"][second_mode],
        first_weight,
    )
    threshold = choose_refusal_threshold(dev_q, dev_g, dev_qids, dev_gids, "balanced")
    holdout_query, holdout_gallery = protocols["holdout"]
    holdout_q = combine_embeddings(
        embedded["holdout"]["query"]["first"][first_mode],
        embedded["holdout"]["query"]["second"][second_mode],
        first_weight,
    )
    holdout_g = combine_embeddings(
        embedded["holdout"]["gallery"]["first"][first_mode],
        embedded["holdout"]["gallery"]["second"][second_mode],
        first_weight,
    )
    holdout_qids, holdout_gids = identities(holdout_query), identities(holdout_gallery)
    report = {
        "first_checkpoint": str(args.first),
        "second_checkpoint": str(args.second),
        "selection": {
            "first_mode": first_mode,
            "second_mode": second_mode,
            "first_weight": first_weight,
            "dev": dev_metrics,
        },
        "embedding_dim": int(holdout_q.shape[1]),
        "holdout": retrieval_metrics(holdout_q, holdout_g, holdout_qids, holdout_gids),
        "holdout_refusal_balanced": refusal_metrics(
            holdout_q, holdout_g, holdout_qids, holdout_gids, threshold["threshold"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

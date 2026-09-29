"""Leave-one-photo-out retrieval on approved local external crops.

This is a data sanity check. It is not a substitute for cross-camera contest
validation because photos from one listing may share a background.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from reid.data import image_transform
from reid.external import read_approved
from reid.infer import load_model


def leave_one_out_metrics(
    features: np.ndarray, identities: list[str]
) -> dict[str, float]:
    if len(features) != len(identities):
        raise ValueError("Feature and identity lengths differ")
    scores = features @ features.T
    np.fill_diagonal(scores, -np.inf)
    aps, top1 = [], []
    for index, row in enumerate(scores):
        order = np.argsort(-row, kind="stable")
        order = order[order != index]
        matches = np.array([identities[j] == identities[index] for j in order])
        if not matches.any():
            continue
        precision = np.cumsum(matches) / (np.arange(len(matches)) + 1)
        aps.append(float(precision[matches].mean()))
        top1.append(float(matches[0]))
    return {
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "rank1": float(np.mean(top1)) if top1 else 0.0,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/auto_ru"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/baseline/best.pt")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    records = read_approved(args.data)
    if not records:
        raise ValueError("No approved external crops")
    model, metadata, device = load_model(args.checkpoint, args.device)
    transform = image_transform(metadata["image_size"])
    vectors = []
    for start in range(0, len(records), args.batch_size):
        crops = []
        for record in records[start : start + args.batch_size]:
            with Image.open(record.path) as image:
                crops.append(transform(image.convert("RGB")))
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            embedding, _ = model(torch.stack(crops).to(device))
        vectors.append(embedding.float().cpu().numpy())
    features = np.concatenate(vectors)
    identities = [record.instance_id for record in records]
    metrics = leave_one_out_metrics(features, identities)
    print(
        f"photos={len(records)} ids={len(set(identities))} mAP={metrics['mAP']:.4f} rank1={metrics['rank1']:.4f}"
    )
    print(
        "Local listing photos share backgrounds; use contest holdout for model selection."
    )


if __name__ == "__main__":
    main()

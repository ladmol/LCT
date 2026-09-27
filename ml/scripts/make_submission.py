"""Export all three contest artifacts from one checkpoint and the original ZIP."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from reid.data import read_records
from reid.infer import combine_embeddings, embed_records, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/convnext_tiny_256/best.pt")
    )
    parser.add_argument(
        "--second-checkpoint",
        type=Path,
        default=None,
        help="Optional second model; its vector is concatenated with the first",
    )
    parser.add_argument("--first-weight", type=float, default=0.5)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/submission_convnext_tiny_256")
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--refusal-threshold",
        type=float,
        default=None,
        help="Cosine threshold for candidates.csv; default comes from the checkpoint",
    )
    parser.add_argument(
        "--tta-flip",
        action="store_true",
        help="Average original and horizontal-flip embeddings",
    )
    args = parser.parse_args()
    model, metadata, device = load_model(args.checkpoint, args.device)
    if args.second_checkpoint is not None and not 0 < args.first_weight < 1:
        raise ValueError("--first-weight must be strictly between zero and one")
    if args.second_checkpoint is not None and args.refusal_threshold is None:
        raise ValueError(
            "An ensemble requires an explicitly calibrated --refusal-threshold"
        )
    threshold = (
        args.refusal_threshold
        if args.refusal_threshold is not None
        else metadata.get("refusal_threshold")
    )
    if threshold is None:
        raise ValueError("Checkpoint has no calibrated refusal threshold")
    if not -1 <= threshold <= 1:
        raise ValueError("Cosine refusal threshold must be between -1 and 1")
    query = read_records(args.archive, "test_query.csv")
    gallery = read_records(args.archive, "test_gallery.csv")
    q = embed_records(
        model,
        args.archive,
        query,
        metadata["image_size"],
        device,
        args.batch_size,
        args.workers,
        args.tta_flip,
    )
    g = embed_records(
        model,
        args.archive,
        gallery,
        metadata["image_size"],
        device,
        args.batch_size,
        args.workers,
        args.tta_flip,
    )
    if args.second_checkpoint is not None:
        second_model, second_metadata, second_device = load_model(
            args.second_checkpoint, args.device
        )
        second_q = embed_records(
            second_model,
            args.archive,
            query,
            second_metadata["image_size"],
            second_device,
            args.batch_size,
            args.workers,
            args.tta_flip,
        )
        second_g = embed_records(
            second_model,
            args.archive,
            gallery,
            second_metadata["image_size"],
            second_device,
            args.batch_size,
            args.workers,
            args.tta_flip,
        )
        q = combine_embeddings(q, second_q, args.first_weight)
        g = combine_embeddings(g, second_g, args.first_weight)
    scores = q @ g.T
    if len(gallery) < 10:
        raise ValueError("Gallery must contain at least ten vehicles")
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "embeddings.npy", np.concatenate((q, g)).astype(np.float32))
    with (args.output / "submission.csv").open(
        "w", encoding="utf-8", newline=""
    ) as raw:
        writer = csv.writer(raw)
        writer.writerow(["query_id"] + [f"gallery_id_{i}" for i in range(1, 11)])
        for record, row in zip(query, scores):
            best = np.argsort(-row, kind="stable")[:10]
            writer.writerow([record.image_id] + [gallery[j].image_id for j in best])
    with (args.output / "candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as raw:
        writer = csv.writer(raw)
        writer.writerow(["query_id", "gallery_id", "confidence"])
        for record, row in zip(query, scores):
            best = np.argsort(-row, kind="stable")[:10]
            accepted = [j for j in best if row[j] >= threshold]
            if not accepted:
                writer.writerow([record.image_id, "", ""])
            else:
                for j in accepted:
                    writer.writerow(
                        [
                            record.image_id,
                            gallery[j].image_id,
                            f"{np.clip((row[j] + 1) / 2, 0, 1):.6f}",
                        ]
                    )
    print(
        f"Saved {len(query)} queries, {len(gallery)} gallery objects to {args.output}"
    )


if __name__ == "__main__":
    main()

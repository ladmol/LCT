"""Compare F1-focused and balanced refusal thresholds on dev and holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import choose_refusal_threshold, refusal_metrics
from reid.data import read_records
from reid.infer import load_model
from reid.protocol import split_identities
from reid.train import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/convnext_tiny_256/best.pt")
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    model, metadata, device = load_model(args.checkpoint, args.device)
    records = read_records(args.archive, "train.csv")
    seed = metadata.get("seed", 42)
    splits = split_identities(records, seed)
    dev = evaluate(
        model,
        args.archive,
        records,
        splits["dev"],
        metadata["image_size"],
        device,
        args.batch_size,
        args.workers,
        seed + 1,
    )
    holdout = evaluate(
        model,
        args.archive,
        records,
        splits["holdout"],
        metadata["image_size"],
        device,
        args.batch_size,
        args.workers,
        seed + 2,
    )
    result = {"dev_ranking": dev[0], "holdout_ranking": holdout[0]}
    for objective in ("f1", "balanced"):
        calibrated = choose_refusal_threshold(dev[1], dev[2], dev[3], dev[4], objective)
        measured = refusal_metrics(
            holdout[1], holdout[2], holdout[3], holdout[4], calibrated["threshold"]
        )
        result[objective] = {"dev": calibrated, "holdout": measured}
    report = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

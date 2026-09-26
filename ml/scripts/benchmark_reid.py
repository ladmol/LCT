"""Measure end-to-end ZIP decoding, cropping, and embedding throughput."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch

from reid.data import read_records
from reid.infer import embed_records, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/resnet50_256/best.pt"))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta-flip", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("samples and batch-size must be positive")
    model, metadata, device = load_model(args.checkpoint, args.device)
    records = read_records(args.archive, "test_query.csv")[: args.samples]
    embed_records(
        model, args.archive, records[: min(len(records), args.batch_size)], metadata["image_size"],
        device, args.batch_size, args.workers, args.tta_flip,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    vectors = embed_records(
        model, args.archive, records, metadata["image_size"], device,
        args.batch_size, args.workers, args.tta_flip,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0
    print(
        f"arch={metadata['arch']} samples={len(vectors)} dim={vectors.shape[1]} "
        f"seconds={elapsed:.3f} fps={len(vectors) / elapsed:.2f} "
        f"peak_cuda_mb={peak:.1f} checkpoint_mb={args.checkpoint.stat().st_size / 2**20:.1f}"
    )


if __name__ == "__main__":
    main()

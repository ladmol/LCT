"""Strip training-only tensors and store a compact offline inference checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    backbone = {
        key: value.half() if value.is_floating_point() else value
        for key, value in checkpoint["model"].items()
        if key.startswith("backbone.")
    }
    deployment = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"model", "num_classes"}
    }
    deployment.update(
        {
            "model": backbone,
            "num_classes": 0,
            "deployment": True,
            "storage_dtype": "float16",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(deployment, args.output)
    print(f"Saved {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()

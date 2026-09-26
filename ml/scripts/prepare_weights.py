"""Save official torchvision ImageNet weights for offline Re-ID training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    convnext_tiny,
    resnet18,
    resnet50,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["resnet18", "resnet50", "convnext_tiny"], default="resnet50")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(f"data/weights/{args.arch}_imagenet.pth")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(output.parent.resolve())
    if args.arch == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif args.arch == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    torch.save(model.state_dict(), output)
    print(f"Saved {args.arch} ImageNet backbone to {output}")


if __name__ == "__main__":
    main()

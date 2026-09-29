"""Small, replaceable embedding backbone for the first reproducible baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_tiny, resnet18, resnet50


class VehicleReID(nn.Module):
    def __init__(self, arch: str = "resnet18", num_classes: int = 0):
        super().__init__()
        if arch == "resnet18":
            self.backbone = resnet18(weights=None)
            self.embedding_dim = 512
        elif arch == "resnet50":
            self.backbone = resnet50(weights=None)
            self.embedding_dim = 2048
        elif arch == "convnext_tiny":
            self.backbone = convnext_tiny(weights=None)
            self.embedding_dim = 768
            self.backbone.classifier[2] = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {arch}")
        if arch.startswith("resnet"):
            self.backbone.fc = nn.Identity()
        self.classifier = (
            nn.Linear(self.embedding_dim, num_classes) if num_classes else None
        )
        self.arch = arch

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.backbone(images)
        embedding = F.normalize(features, p=2, dim=1)
        logits = self.classifier(features) if self.classifier is not None else None
        return embedding, logits


def batch_hard_triplet(
    embedding: torch.Tensor, labels: torch.Tensor, margin: float = 0.3
):
    distances = torch.cdist(embedding.float(), embedding.float(), p=2)
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    hardest_positive = distances.masked_fill(~same | eye, -1).max(dim=1).values
    hardest_negative = distances.masked_fill(same, float("inf")).min(dim=1).values
    return F.relu(hardest_positive - hardest_negative + margin).mean()


def arcface_logits(
    embedding: torch.Tensor,
    classifier_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.2,
    scale: float = 30.0,
) -> torch.Tensor:
    """Apply an angular margin to the target identity while keeping cosine logits."""
    if not 0 <= margin < 1 or scale <= 0:
        raise ValueError("ArcFace margin and scale must be valid positive values")
    with torch.autocast(device_type=embedding.device.type, enabled=False):
        cosine = F.linear(
            F.normalize(embedding.float(), dim=1),
            F.normalize(classifier_weight.float(), dim=1),
        ).clamp(-1 + 1e-6, 1 - 1e-6)
        target_cosine = cosine.gather(1, labels[:, None])
        target_with_margin = torch.cos(torch.acos(target_cosine) + margin)
        logits = cosine.scatter(1, labels[:, None], target_with_margin)
        return logits * scale

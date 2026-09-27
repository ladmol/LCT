"""Inference API shared by evaluation, submission generation, and the backend."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader

from attributes.attribute_heads import AttributeHeads
from reid.data import ContestDataset, VehicleRecord, crop_vehicle, image_transform
from reid.model import VehicleReID


def load_model(checkpoint_path: str | Path, device: str = "auto"):
    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
    )
    target = torch.device(resolved_device if device == "auto" else device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = VehicleReID(checkpoint["arch"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model"])
    model.to(target).eval()
    return model, checkpoint, target


@torch.inference_mode()
def embed_records(
    model: VehicleReID,
    archive: str | Path,
    records: list[VehicleRecord],
    image_size: int,
    device: torch.device,
    batch_size: int = 32,
    workers: int = 0,
    tta_flip: bool = False,
) -> np.ndarray:
    loader = DataLoader(
        ContestDataset(archive, records, image_size),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    vectors = []
    for images, _, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            features, _ = model(images)
            if tta_flip:
                flipped, _ = model(torch.flip(images, dims=(3,)))
                features = F.normalize(features + flipped, dim=1)
        vectors.append(features.float().cpu().numpy())
    return (
        np.concatenate(vectors).astype(np.float32)
        if vectors
        else np.empty((0, model.embedding_dim), np.float32)
    )


def combine_embeddings(
    first: np.ndarray, second: np.ndarray, first_weight: float = 0.5
) -> np.ndarray:
    """Build one unit vector whose cosine is a weighted average of two cosines."""
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("Embeddings must be two-dimensional [batch, features]")
    if first.shape[0] != second.shape[0]:
        raise ValueError("Embedding batches must contain the same number of objects")
    if not np.isfinite(first_weight) or not 0 <= first_weight <= 1:
        raise ValueError("first_weight must be between zero and one")
    return np.concatenate(
        (np.sqrt(first_weight) * first, np.sqrt(1.0 - first_weight) * second), axis=1
    ).astype(np.float32)


class VehicleEmbedder:
    """Minimal integration point: image bytes + xywh box -> unit float32 vector."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        *,
        tta_flip: bool = False,
    ):
        self.model, self.metadata, self.device = load_model(checkpoint_path, device)
        self.transform = image_transform(self.metadata["image_size"])
        self.tta_flip = tta_flip

    @torch.inference_mode()
    def embed(self, image_bytes: bytes, bbox: tuple[int, int, int, int]) -> np.ndarray:
        return self.embed_many(image_bytes, [bbox])[0]

    @torch.inference_mode()
    def embed_many(
        self,
        image_bytes: bytes,
        bboxes: list[tuple[int, int, int, int]],
        batch_size: int = 32,
    ) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not bboxes:
            return np.empty((0, self.model.embedding_dim), dtype=np.float32)
        with Image.open(io.BytesIO(image_bytes)) as image:
            crops = [self.transform(crop_vehicle(image, bbox)) for bbox in bboxes]
        vectors = []
        for start in range(0, len(crops), batch_size):
            tensors = torch.stack(crops[start : start + batch_size]).to(self.device)
            with torch.autocast(
                device_type=self.device.type, enabled=self.device.type == "cuda"
            ):
                features, _ = self.model(tensors)
                if self.tta_flip:
                    flipped, _ = self.model(torch.flip(tensors, dims=(3,)))
                    features = F.normalize(features + flipped, dim=1)
            vectors.append(features.float().cpu().numpy())
        return np.concatenate(vectors).astype(np.float32)


class VehicleEnsembleEmbedder:
    """Two-backbone production embedder returning one cosine-compatible vector."""

    def __init__(
        self,
        first_checkpoint: str | Path,
        second_checkpoint: str | Path,
        device: str = "auto",
        *,
        first_weight: float = 0.5,
        tta_flip: bool = True,
    ):
        if not 0 < first_weight < 1:
            raise ValueError("first_weight must be strictly between zero and one")
        self.first = VehicleEmbedder(first_checkpoint, device, tta_flip=tta_flip)
        self.second = VehicleEmbedder(second_checkpoint, device, tta_flip=tta_flip)
        self.first_weight = first_weight
        self.embedding_dim = (
            self.first.model.embedding_dim + self.second.model.embedding_dim
        )

    def embed(self, image_bytes: bytes, bbox: tuple[int, int, int, int]) -> np.ndarray:
        return self.embed_many(image_bytes, [bbox])[0]

    def embed_many(
        self,
        image_bytes: bytes,
        bboxes: list[tuple[int, int, int, int]],
        batch_size: int = 32,
    ) -> np.ndarray:
        first = self.first.embed_many(image_bytes, bboxes, batch_size)
        second = self.second.embed_many(image_bytes, bboxes, batch_size)
        return combine_embeddings(first, second, self.first_weight)


class VehicleAttributeRecognizer:
    """Experimental attribute predictions from a reviewed external checkpoint."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto"):
        self.model, self.metadata, self.device = load_model(checkpoint_path, device)
        if (
            "attribute_heads" not in self.metadata
            or "attribute_labels" not in self.metadata
        ):
            raise ValueError("Checkpoint does not contain attribute heads")
        labels = self.metadata["attribute_labels"]
        self.labels = labels
        self.heads = AttributeHeads(
            self.model.embedding_dim,
            max(1, len(labels["make"])),
            max(1, len(labels["model"])),
            max(1, len(labels["restyling"])),
        )
        self.heads.load_state_dict(self.metadata["attribute_heads"])
        self.heads.to(self.device).eval()
        self.transform = image_transform(self.metadata["image_size"])

    @torch.inference_mode()
    def predict(
        self, image_bytes: bytes, bbox: tuple[int, int, int, int]
    ) -> dict[str, dict[str, float | str] | None]:
        with Image.open(io.BytesIO(image_bytes)) as image:
            tensor = (
                self.transform(crop_vehicle(image, bbox)).unsqueeze(0).to(self.device)
            )
        with torch.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        ):
            vector, _ = self.model(tensor)
            logits = self.heads(vector)
        result = {}
        for field, mapping in self.labels.items():
            if field == "instance":
                continue
            if len(mapping) < 2:
                result[field] = None
                continue
            probabilities = logits[field].float().softmax(dim=1)[0]
            index = int(probabilities.argmax())
            reverse = {value: key for key, value in mapping.items()}
            result[field] = {
                "label": reverse[index],
                "score": float(probabilities[index]),
            }
        return result

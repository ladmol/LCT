"""Read the contest archive without extracting its 7 GB of source images."""

from __future__ import annotations

import csv
import io
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


@dataclass(frozen=True)
class VehicleRecord:
    image_id: str
    bbox: tuple[int, int, int, int]  # x, y, width, height in the source frame
    vehicle_id: str | None = None
    camera_id: str | None = None


def read_records(archive: str | Path, csv_name: str) -> list[VehicleRecord]:
    with zipfile.ZipFile(archive) as source, source.open(csv_name) as raw:
        rows = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
        expected = {"image_id", "x", "y", "w", "h"}
        if not expected.issubset(rows.fieldnames or []):
            raise ValueError(f"{csv_name}: expected columns {sorted(expected)}")
        records = [
            VehicleRecord(
                image_id=row["image_id"],
                bbox=tuple(int(row[key]) for key in ("x", "y", "w", "h")),
                vehicle_id=row.get("vehicle_id") or None,
                camera_id=row.get("camera_id") or None,
            )
            for row in rows
        ]
    if any(w <= 0 or h <= 0 for record in records for _, _, w, h in [record.bbox]):
        raise ValueError(f"{csv_name}: BBox width and height must be positive")
    return records


def crop_vehicle(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    """Clamp a supplied xywh box and return RGB pixels; never run a detector."""
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError("BBox width and height must be positive")
    left, top = max(0, x), max(0, y)
    right, bottom = min(image.width, x + w), min(image.height, y + h)
    if left >= right or top >= bottom:
        raise ValueError(f"BBox {bbox} is outside image {image.size}")
    return image.convert("RGB").crop((left, top, right, bottom))


def image_transform(image_size: int, *, training: bool = False, augmentation: str = "basic"):
    """Preserve aspect ratio, then use ImageNet normalization for the backbone."""
    if augmentation not in {"basic", "strong"}:
        raise ValueError(f"Unknown augmentation profile: {augmentation}")
    steps = [SquarePad(image_size)]
    if training:
        steps.append(transforms.RandomHorizontalFlip())
        if augmentation == "strong":
            steps.extend(
                [
                    transforms.RandomAffine(
                        degrees=5,
                        translate=(0.04, 0.04),
                        scale=(0.9, 1.1),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        fill=(124, 116, 104),
                    ),
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.04)],
                        p=0.8,
                    ),
                    transforms.RandomGrayscale(p=0.05),
                ]
            )
        else:
            steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    if training and augmentation == "strong":
        steps.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.14), ratio=(0.3, 3.3), value=0))
    return transforms.Compose(steps)


class SquarePad:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> Image.Image:
        fitted = ImageOps.contain(
            image, (self.image_size, self.image_size), method=Image.Resampling.BICUBIC
        )
        canvas = Image.new("RGB", (self.image_size, self.image_size), (124, 116, 104))
        canvas.paste(fitted, ((self.image_size - fitted.width) // 2, (self.image_size - fitted.height) // 2))
        return canvas


class ContestDataset(Dataset):
    def __init__(
        self,
        archive: str | Path,
        records: list[VehicleRecord],
        image_size: int = 256,
        *,
        training: bool = False,
        augmentation: str = "basic",
        label_to_index: dict[str, int] | None = None,
    ):
        self.archive_path = str(archive)
        self.records = records
        self.transform = image_transform(image_size, training=training, augmentation=augmentation)
        self.label_to_index = label_to_index or {}
        self._archive: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archive"] = None  # A DataLoader worker opens its own ZipFile.
        return state

    def __getitem__(self, index: int):
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.archive_path)
        record = self.records[index]
        with self._archive.open(f"images/{record.image_id}.jpg") as raw, Image.open(raw) as image:
            crop = crop_vehicle(image, record.bbox)
            tensor = self.transform(crop)
        label = self.label_to_index.get(record.vehicle_id or "", -1)
        return tensor, label, index


class IdentityBatchSampler(Sampler[list[int]]):
    """Sample P identities x K views, preferring different cameras for positives."""

    def __init__(self, records: list[VehicleRecord], identities: int = 4, views: int = 4, seed: int = 42):
        if identities < 2 or views < 2:
            raise ValueError("Identity batches need at least two identities and two views")
        self.groups: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            if record.vehicle_id is None:
                raise ValueError("Identity sampler needs vehicle_id for every record")
            self.groups.setdefault(record.vehicle_id, []).append(index)
        if len(self.groups) < identities:
            raise ValueError("Too few identities for one batch")
        self.identities = identities
        self.views = views
        self.seed = seed
        self.epoch = 0
        self.records = records

    def __len__(self) -> int:
        return len(self.groups) // self.identities

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        ids = list(self.groups)
        rng.shuffle(ids)
        for start in range(0, len(ids) - self.identities + 1, self.identities):
            batch: list[int] = []
            for identity in ids[start : start + self.identities]:
                indices = self.groups[identity].copy()
                rng.shuffle(indices)
                chosen = []
                used_cameras = set()
                for index in indices:
                    camera = self.records[index].camera_id
                    if camera not in used_cameras:
                        chosen.append(index)
                        used_cameras.add(camera)
                    if len(chosen) == self.views:
                        break
                chosen.extend(i for i in indices if i not in chosen)  # fill with other views
                if len(chosen) < self.views:
                    chosen.extend(rng.choices(indices, k=self.views - len(chosen)))
                batch.extend(chosen[: self.views])
            yield batch

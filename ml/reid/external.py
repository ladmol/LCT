"""Only approved, redacted Auto.ru crops can enter external pretraining."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, Sampler

from reid.data import image_transform


@dataclass(frozen=True)
class ExternalRecord:
    path: Path
    instance_id: str
    make: str
    model: str
    restyling: str


def read_approved(data_dir: Path) -> list[ExternalRecord]:
    manifest = data_dir / "approved.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Approved manifest is missing: {manifest}")
    records = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (
            row.get("plate_reviewed") is not True
            or row.get("face_reviewed") is not True
        ):
            raise ValueError(f"Unreviewed external image: {row.get('image_path')}")
        path = (data_dir / row["image_path"]).resolve()
        if (
            not path.is_relative_to((data_dir / "approved").resolve())
            or not path.is_file()
        ):
            raise ValueError(f"Approved image is missing or outside approved/: {path}")
        records.append(
            ExternalRecord(
                path=path,
                instance_id=row["instance_id"],
                make=row["make"].lower(),
                model=f"{row['make']}/{row['model']}".lower(),
                restyling=row.get("restyling_label", "unknown"),
            )
        )
    return records


class ApprovedAutoDataset(Dataset):
    def __init__(
        self,
        records: list[ExternalRecord],
        image_size: int,
        labels: dict[str, dict[str, int]],
    ):
        self.records = records
        self.transform = image_transform(image_size, training=True)
        self.labels = labels

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.path) as image:
            pixels = image.convert("RGB")
            tensor = self.transform(pixels)
        return (
            tensor,
            self.labels["instance"][record.instance_id],
            self.labels["make"].get(record.make, -1),
            self.labels["model"].get(record.model, -1),
            self.labels["restyling"].get(record.restyling, -1),
        )


class ModelAwareBatchSampler(Sampler[list[int]]):
    """Prefer two different cars of the same model in every identity batch."""

    def __init__(
        self,
        records: list[ExternalRecord],
        identities: int = 8,
        views: int = 2,
        seed: int = 42,
    ):
        self.records = records
        self.identities = identities
        self.views = views
        self.seed = seed
        self.epoch = 0
        self.by_id: dict[str, list[int]] = {}
        self.by_model: dict[str, set[str]] = {}
        for index, record in enumerate(records):
            self.by_id.setdefault(record.instance_id, []).append(index)
            self.by_model.setdefault(record.model, set()).add(record.instance_id)
        if identities < 2 or views < 2 or len(self.by_id) < identities:
            raise ValueError(
                "Need at least P distinct instances and two photos per instance"
            )
        if min(Counter(record.instance_id for record in records).values()) < views:
            raise ValueError(f"Each approved listing needs at least {views} photos")

    def __len__(self):
        return max(1, len(self.records) // (self.identities * self.views))

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        models = [name for name, ids in self.by_model.items() if len(ids) >= 2]
        all_ids = list(self.by_id)
        for _ in range(len(self)):
            if models:
                selected = rng.sample(sorted(self.by_model[rng.choice(models)]), 2)
            else:
                selected = rng.sample(all_ids, 2)
            selected.extend(
                rng.sample(
                    [identity for identity in all_ids if identity not in selected],
                    self.identities - 2,
                )
            )
            batch = []
            for identity in selected:
                batch.extend(rng.sample(self.by_id[identity], self.views))
            yield batch

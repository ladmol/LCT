"""Identity-disjoint development split and cross-camera query/gallery protocol."""

from __future__ import annotations

import random

from reid.data import VehicleRecord


def split_identities(records: list[VehicleRecord], seed: int = 42) -> dict[str, list[str]]:
    identities = sorted({record.vehicle_id for record in records if record.vehicle_id is not None})
    random.Random(seed).shuffle(identities)
    train_end = round(len(identities) * 0.70)
    dev_end = round(len(identities) * 0.85)
    return {
        "train": identities[:train_end],
        "dev": identities[train_end:dev_end],
        "holdout": identities[dev_end:],
    }


def cross_camera_protocol(
    records: list[VehicleRecord], identities: list[str], seed: int = 42
) -> tuple[list[VehicleRecord], list[VehicleRecord]]:
    """Put 75% of identities in gallery; remaining identities simulate unknown queries."""
    rng = random.Random(seed)
    ids = identities.copy()
    rng.shuffle(ids)
    known_count = round(len(ids) * 0.75)
    known = set(ids[:known_count])
    by_id: dict[str, list[VehicleRecord]] = {}
    for record in records:
        if record.vehicle_id in ids:
            by_id.setdefault(record.vehicle_id, []).append(record)
    query, gallery = [], []
    for identity in ids:
        views = by_id.get(identity, [])
        if not views:
            continue
        if identity not in known:
            query.append(rng.choice(views))
            continue
        by_camera: dict[str, list[VehicleRecord]] = {}
        for record in views:
            by_camera.setdefault(record.camera_id or "unknown", []).append(record)
        cameras = sorted(by_camera)
        if len(cameras) < 2:
            continue
        gallery_camera = rng.choice(cameras)
        gallery.append(rng.choice(by_camera[gallery_camera]))
        query.extend(record for camera in cameras if camera != gallery_camera for record in by_camera[camera])
    if not query or not gallery:
        raise ValueError("Cannot build a cross-camera query/gallery protocol")
    return query, gallery

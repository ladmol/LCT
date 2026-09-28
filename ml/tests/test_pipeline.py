"""Check the boundaries most likely to invalidate a Re-ID experiment."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from eval.decision import choose_decision_rule, decision_metrics, decision_values
from eval.metrics import choose_refusal_threshold, refusal_metrics, retrieval_metrics
from reid.autonomy import decide_match, pool_tracklet_embeddings
from reid.data import SquarePad, VehicleRecord, crop_vehicle
from reid.external import ModelAwareBatchSampler, read_approved
from reid.infer import VehicleAttributeRecognizer, combine_embeddings
from reid.model import arcface_logits
from reid.pretrain_external import main as pretrain_external
from reid.protocol import cross_camera_protocol, split_identities
from scripts.eval_external_pilot import leave_one_out_metrics
from scripts.process_auto_ru_reviews import main as process_reviews


def test_bbox_and_square_padding_keep_the_complete_vehicle():
    image = Image.new("RGB", (100, 20), "red")
    crop = crop_vehicle(image, (10, 0, 80, 20))
    padded = SquarePad(100)(crop)
    assert padded.size == (100, 100)
    assert padded.getpixel((50, 50)) == (255, 0, 0)
    assert padded.getpixel((50, 0)) == (124, 116, 104)
    with pytest.raises(ValueError):
        crop_vehicle(image, (200, 0, 10, 10))


def test_holdout_has_new_ids_and_query_uses_other_camera():
    records = [
        VehicleRecord(f"{identity}_{camera}", (0, 0, 10, 10), identity, camera)
        for identity in map(str, range(20))
        for camera in ("a", "b")
    ]
    split = split_identities(records)
    groups = [set(split[key]) for key in ("train", "dev", "holdout")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    query, gallery = cross_camera_protocol(records, split["dev"] + split["holdout"])
    for item in query:
        for candidate in gallery:
            if candidate.vehicle_id == item.vehicle_id:
                assert candidate.camera_id != item.camera_id


def test_ranking_and_refusal_are_separate_metrics():
    query = np.eye(2, dtype=np.float32)
    gallery = np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
    query_ids = ["known", "unknown"]
    gallery_ids = ["known", "other"]
    ranking = retrieval_metrics(query, gallery, query_ids, gallery_ids)
    assert ranking["mAP"] == ranking["rank1"] == 1.0
    assert ranking["known_queries"] == 1
    chosen = choose_refusal_threshold(query, gallery, query_ids, gallery_ids)
    assert (
        refusal_metrics(query, gallery, query_ids, gallery_ids, chosen["threshold"])[
            "f1"
        ]
        == 1.0
    )
    assert chosen["tnr"] == 1.0


def test_combined_embedding_is_unit_length_and_averages_cosines():
    first = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    second = np.array([[0.6, 0.8], [0.8, 0.6]], dtype=np.float32)
    combined = combine_embeddings(first, second, 0.5)
    assert combined.shape == (2, 4)
    assert np.allclose(np.linalg.norm(combined, axis=1), 1.0)
    expected = 0.5 * (first @ first.T) + 0.5 * (second @ second.T)
    assert np.allclose(combined @ combined.T, expected)
    with pytest.raises(ValueError, match="same number"):
        combine_embeddings(first, second[:1])
    with pytest.raises(ValueError, match="two-dimensional"):
        combine_embeddings(first[0], second[0])
    with pytest.raises(ValueError, match="between zero and one"):
        combine_embeddings(first, second, float("nan"))


def test_tracklet_pooling_and_margin_decision():
    pooled = pool_tracklet_embeddings(
        np.array([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32)
    )
    decision = decide_match(
        pooled,
        np.array([[1.0, 0.0], [0.7, 0.7]], dtype=np.float32),
        similarity_threshold=0.8,
        margin_threshold=0.1,
    )
    assert np.isclose(np.linalg.norm(pooled), 1.0)
    assert decision.gallery_index == 0 and decision.accepted


def test_arcface_reduces_the_target_logit_by_an_angular_margin():
    embedding = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    logits = arcface_logits(embedding, weight, labels, margin=0.2, scale=1.0)
    assert torch.all((0.97 < logits.diag()) & (logits.diag() < 1.0))
    assert torch.allclose(logits.flip(1).diag(), torch.zeros(2), atol=1e-5)


def test_margin_rule_can_reject_an_ambiguous_top1():
    query = np.array([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]], dtype=np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    gallery = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    values = decision_values(query, gallery, ["a", "unknown", "b"], ["a", "b"])
    metrics = decision_metrics(values, similarity_threshold=0.5, margin_threshold=0.2)
    assert metrics["precision"] == metrics["recall"] == metrics["tnr"] == 1.0
    rule = choose_decision_rule(
        query, gallery, ["a", "unknown", "b"], ["a", "b"], objective="balanced"
    )
    assert rule["precision"] == rule["tnr"] == 1.0


def test_external_pilot_excludes_the_query_photo_itself():
    features = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    assert leave_one_out_metrics(features, ["a", "a", "b", "b"]) == {
        "mAP": 1.0,
        "rank1": 1.0,
    }


def test_external_data_requires_review_and_enough_views(tmp_path):
    approved = tmp_path / "approved" / "listing"
    approved.mkdir(parents=True)
    image = approved / "00.jpg"
    Image.new("RGB", (8, 8), "red").save(image)
    row = {
        "image_path": "approved/listing/00.jpg",
        "instance_id": "listing",
        "make": "toyota",
        "model": "camry",
        "restyling_label": "unknown",
        "plate_reviewed": False,
        "face_reviewed": True,
    }
    manifest = tmp_path / "approved.jsonl"
    manifest.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="Unreviewed"):
        read_approved(tmp_path)
    row["plate_reviewed"] = True
    manifest.write_text(json.dumps(row), encoding="utf-8")
    records = read_approved(tmp_path)
    with pytest.raises(ValueError, match="distinct instances"):
        ModelAwareBatchSampler(records, identities=2, views=2)


def test_review_masks_plate_before_writing_approved_crop(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "listing"
    raw.mkdir(parents=True)
    Image.new("RGB", (20, 10), "white").save(raw / "00.jpg")
    source = {
        "image_path": "raw/listing/00.jpg",
        "listing_id": "listing",
        "make": "toyota",
        "model": "camry",
        "catalog_name": "Camry",
        "restyling_label": "unknown",
        "source_url": "https://auto.ru/example",
    }
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(source) + "\n", encoding="utf-8"
    )
    review = {
        "image_path": source["image_path"],
        "status": "approved",
        "vehicle_bbox": [0, 0, 20, 10],
        "redactions": [[2, 2, 5, 5]],
        "plate_reviewed": True,
        "face_reviewed": True,
        "restyling_label": "pre-restyling",
    }
    (tmp_path / "reviews.jsonl").write_text(json.dumps(review) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "review",
            "--data",
            str(tmp_path),
            "--reviews",
            str(tmp_path / "reviews.jsonl"),
        ],
    )
    process_reviews()
    approved = read_approved(tmp_path)
    assert approved[0].restyling == "pre-restyling"
    with Image.open(approved[0].path) as image:
        assert max(image.getpixel((4, 4))) < 10


def test_external_training_and_attribute_inference_smoke(tmp_path, monkeypatch):
    weights = Path("data/weights/resnet18_imagenet.pth")
    if not weights.is_file():
        pytest.skip("Optional local ImageNet weights are missing")
    data = tmp_path / "external"
    manifest = []
    for identity in range(8):
        make, model = ("toyota", "camry") if identity < 4 else ("kia", "rio")
        folder = data / "approved" / str(identity)
        folder.mkdir(parents=True)
        for view in range(2):
            name = f"{view}.jpg"
            Image.new("RGB", (64, 64), (identity * 25, view * 40, 100)).save(
                folder / name
            )
            manifest.append(
                {
                    "image_path": f"approved/{identity}/{name}",
                    "instance_id": str(identity),
                    "make": make,
                    "model": model,
                    "restyling_label": "restyling" if identity % 2 else "pre-restyling",
                    "plate_reviewed": True,
                    "face_reviewed": True,
                }
            )
    (data / "approved.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8"
    )
    output = tmp_path / "model"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pretrain",
            "--data",
            str(data),
            "--output",
            str(output),
            "--init-weights",
            str(weights),
            "--contest-archive",
            str(tmp_path / "missing.zip"),
            "--epochs",
            "1",
            "--image-size",
            "64",
        ],
    )
    pretrain_external()
    recognizer = VehicleAttributeRecognizer(output / "best.pt", device="cpu")
    prediction = recognizer.predict(
        (data / manifest[0]["image_path"]).read_bytes(), (0, 0, 64, 64)
    )
    assert prediction["make"]["label"] in {"toyota", "kia"}
    assert prediction["restyling"]["label"] in {"pre-restyling", "restyling"}

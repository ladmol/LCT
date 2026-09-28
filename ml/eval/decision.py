"""Calibrate autonomous acceptance using similarity and top-1 separation."""

from __future__ import annotations

import numpy as np


def decision_values(
    query: np.ndarray,
    gallery: np.ndarray,
    query_ids: list[str | None],
    gallery_ids: list[str | None],
) -> dict[str, np.ndarray]:
    if len(gallery) < 2:
        raise ValueError("Decision calibration requires at least two gallery objects")
    scores = query @ gallery.T
    top = scores.argmax(axis=1)
    best = scores[np.arange(len(scores)), top]
    second = np.partition(scores, -2, axis=1)[:, -2]
    has_match = np.asarray([identity in gallery_ids for identity in query_ids])
    correct = np.asarray(
        [
            gallery_ids[gallery_index] == query_ids[index]
            for index, gallery_index in enumerate(top)
        ]
    )
    return {
        "best": best,
        "margin": best - second,
        "has_match": has_match,
        "correct": correct,
    }


def decision_metrics(
    values: dict[str, np.ndarray], similarity_threshold: float, margin_threshold: float
) -> dict[str, float]:
    accepted = (values["best"] >= similarity_threshold) & (
        values["margin"] >= margin_threshold
    )
    has_match = values["has_match"].astype(bool)
    correct = values["correct"].astype(bool)
    true_positive = accepted & correct & has_match
    false_positive = accepted & (~correct | ~has_match)
    unknown = ~has_match
    tp = int(true_positive.sum())
    fp = int(false_positive.sum())
    known = int(has_match.sum())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / known if known else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "tnr": float((~accepted & unknown).sum() / unknown.sum())
        if unknown.any()
        else 0.0,
        "accepted": float(accepted.sum()),
        "accept_rate": float(accepted.mean()),
        "similarity_threshold": float(similarity_threshold),
        "margin_threshold": float(margin_threshold),
    }


def choose_decision_rule(
    query: np.ndarray,
    gallery: np.ndarray,
    query_ids: list[str | None],
    gallery_ids: list[str | None],
    *,
    objective: str = "balanced",
    target_precision: float = 0.95,
    target_tnr: float = 0.95,
) -> dict[str, float | bool]:
    if objective not in {"balanced", "strict"}:
        raise ValueError(f"Unknown decision objective: {objective}")
    values = decision_values(query, gallery, query_ids, gallery_ids)
    best = values["best"]
    margin = values["margin"]
    similarity_thresholds = np.unique(np.r_[best.min() - 1e-6, best, best.max() + 1e-6])
    margin_thresholds = np.unique(
        np.r_[margin.min() - 1e-6, margin, margin.max() + 1e-6]
    )
    options = [
        decision_metrics(values, similarity_threshold, margin_threshold)
        for similarity_threshold in similarity_thresholds
        for margin_threshold in margin_thresholds
    ]
    if objective == "balanced":
        chosen = max(
            options,
            key=lambda item: (
                2 * item["f1"] * item["tnr"] / (item["f1"] + item["tnr"] + 1e-12),
                item["precision"],
                item["recall"],
            ),
        )
        return {**chosen, "target_met": True}

    minimum_accepted = max(5, round(len(query) * 0.02))
    eligible = [
        item
        for item in options
        if item["accepted"] >= minimum_accepted
        and item["precision"] >= target_precision
        and item["tnr"] >= target_tnr
    ]
    if eligible:
        chosen = max(
            eligible,
            key=lambda item: (item["recall"], item["accepted"], item["f1"]),
        )
        return {**chosen, "target_met": True}
    nontrivial = [item for item in options if item["accepted"] >= minimum_accepted]
    chosen = max(
        nontrivial,
        key=lambda item: (
            min(item["precision"] / target_precision, item["tnr"] / target_tnr),
            item["recall"],
        ),
    )
    return {**chosen, "target_met": False}

"""Cross-camera retrieval and open-set refusal metrics."""

from __future__ import annotations

import numpy as np


def retrieval_metrics(
    query: np.ndarray,
    gallery: np.ndarray,
    query_ids: list[str | None],
    gallery_ids: list[str | None],
    query_cameras: list[str | None] | None = None,
    gallery_cameras: list[str | None] | None = None,
) -> dict[str, float]:
    if len(query) != len(query_ids) or len(gallery) != len(gallery_ids):
        raise ValueError("Embedding and label lengths differ")
    scores = query @ gallery.T
    ap_values, rank1, rank5 = [], [], []
    for index, score in enumerate(scores):
        if query_ids[index] is None:
            continue
        order = np.argsort(-score, kind="stable")
        if query_cameras is not None and gallery_cameras is not None:
            order = np.array(
                [
                    j
                    for j in order
                    if not (
                        gallery_ids[j] == query_ids[index]
                        and gallery_cameras[j] == query_cameras[index]
                    )
                ],
                dtype=int,
            )
        matches = np.array(
            [gallery_ids[j] == query_ids[index] for j in order], dtype=bool
        )
        if not matches.any():
            continue  # Unknown queries belong to refusal evaluation, not mAP.
        precision = np.cumsum(matches) / (np.arange(len(matches)) + 1)
        ap_values.append(float(precision[matches].mean()))
        rank1.append(float(matches[:1].any()))
        rank5.append(float(matches[:5].any()))
    return {
        "mAP": float(np.mean(ap_values)) if ap_values else 0.0,
        "rank1": float(np.mean(rank1)) if rank1 else 0.0,
        "rank5": float(np.mean(rank5)) if rank5 else 0.0,
        "known_queries": float(len(ap_values)),
    }


def refusal_metrics(
    query: np.ndarray,
    gallery: np.ndarray,
    query_ids: list[str | None],
    gallery_ids: list[str | None],
    threshold: float,
) -> dict[str, float]:
    """F1 for a correct top-1 identification; TNR for no-match queries."""
    scores = query @ gallery.T
    top = scores.argmax(axis=1)
    best = scores[np.arange(len(scores)), top]
    has_match = np.array([identity in gallery_ids for identity in query_ids])
    correct = np.array([gallery_ids[j] == query_ids[i] for i, j in enumerate(top)])
    accepted = best >= threshold
    tp = int((accepted & correct & has_match).sum())
    fp = int((accepted & (~correct | ~has_match)).sum())
    fn = int((has_match & ~(accepted & correct)).sum())
    unknown = ~has_match
    return {
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "tnr": float((~accepted & unknown).sum() / unknown.sum())
        if unknown.any()
        else 0.0,
        "threshold": float(threshold),
        "known_queries": float(has_match.sum()),
        "unknown_queries": float(unknown.sum()),
    }


def choose_refusal_threshold(
    query: np.ndarray,
    gallery: np.ndarray,
    query_ids: list[str | None],
    gallery_ids: list[str | None],
    objective: str = "f1",
) -> dict[str, float]:
    if objective not in {"f1", "balanced"}:
        raise ValueError(f"Unknown refusal objective: {objective}")
    scores = (query @ gallery.T).max(axis=1)
    boundaries = np.unique(np.r_[scores.min() - 1e-6, scores, scores.max() + 1e-6])
    options = [
        refusal_metrics(query, gallery, query_ids, gallery_ids, value)
        for value in boundaries
    ]
    if objective == "f1":
        return max(options, key=lambda item: (item["f1"], item["tnr"]))
    return max(
        options,
        key=lambda item: (
            2 * item["f1"] * item["tnr"] / (item["f1"] + item["tnr"] + 1e-12),
            item["f1"],
        ),
    )

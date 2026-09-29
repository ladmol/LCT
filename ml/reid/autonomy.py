"""Tracklet aggregation and open-set decisions for autonomous matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchDecision:
    gallery_index: int
    similarity: float
    margin: float
    accepted: bool


def pool_tracklet_embeddings(
    vectors: np.ndarray, quality_weights: np.ndarray | None = None
) -> np.ndarray:
    """Create one unit descriptor from several observations of one vehicle."""
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("Tracklet embeddings must have shape [frames, features]")
    if quality_weights is None:
        pooled = vectors.mean(axis=0)
    else:
        weights = np.asarray(quality_weights, dtype=np.float32)
        if (
            weights.shape != (len(vectors),)
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError(
                "Quality weights must be non-negative and match the frames"
            )
        pooled = np.average(vectors, axis=0, weights=weights)
    norm = float(np.linalg.norm(pooled))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Tracklet pooling produced an invalid vector")
    return (pooled / norm).astype(np.float32)


def decide_match(
    query: np.ndarray,
    gallery: np.ndarray,
    *,
    similarity_threshold: float,
    margin_threshold: float,
) -> MatchDecision:
    """Accept top-1 only when it is strong and separated from the runner-up."""
    if query.ndim != 1 or gallery.ndim != 2 or gallery.shape[1] != len(query):
        raise ValueError("Query and gallery embedding dimensions differ")
    if len(gallery) < 2:
        raise ValueError("At least two gallery candidates are required")
    scores = gallery @ query
    top_two = np.argpartition(scores, -2)[-2:]
    ordered = top_two[np.argsort(scores[top_two])[::-1]]
    best, second = int(ordered[0]), int(ordered[1])
    similarity = float(scores[best])
    margin = float(similarity - scores[second])
    return MatchDecision(
        gallery_index=best,
        similarity=similarity,
        margin=margin,
        accepted=similarity >= similarity_threshold and margin >= margin_threshold,
    )

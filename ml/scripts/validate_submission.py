"""Check that export files agree with the test CSV order and each other."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from reid.data import read_records


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def validate(archive: Path, output: Path) -> None:
    query = read_records(archive, "test_query.csv")
    gallery = read_records(archive, "test_gallery.csv")
    qids = [record.image_id for record in query]
    gids = [record.image_id for record in gallery]
    vectors = np.load(output / "embeddings.npy", allow_pickle=False)
    if vectors.dtype != np.float32 or vectors.shape[0] != len(query) + len(gallery) or vectors.ndim != 2:
        raise ValueError("Embeddings must be float32 [query + gallery, dimension]")
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
        raise ValueError("Embeddings must be finite and unit-normalized")
    ranking = rows(output / "submission.csv")
    if len(ranking) != len(query) or [row.get("query_id") for row in ranking] != qids:
        raise ValueError("submission.csv does not follow test_query.csv order")
    scores = vectors[: len(query)] @ vectors[len(query) :].T
    query_index = {image_id: index for index, image_id in enumerate(qids)}
    gallery_index = {image_id: index for index, image_id in enumerate(gids)}
    expected = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    top_by_query = {}
    for index, row in enumerate(ranking):
        actual = [row.get(f"gallery_id_{rank}") for rank in range(1, 11)]
        target = [gids[j] for j in expected[index]]
        if actual != target:
            raise ValueError(f"Top-10 disagrees with embeddings for {qids[index]}")
        top_by_query[qids[index]] = set(target)
    candidates = rows(output / "candidates.csv")
    by_query: dict[str, list[dict[str, str]]] = {query_id: [] for query_id in qids}
    for row in candidates:
        query_id = row.get("query_id")
        if query_id not in by_query:
            raise ValueError(f"Unexpected candidate query: {query_id}")
        by_query[query_id].append(row)
    for query_id, choices in by_query.items():
        if not choices:
            raise ValueError(f"Missing candidate/refusal row for {query_id}")
        if len(choices) == 1 and not choices[0].get("gallery_id"):
            if choices[0].get("confidence"):
                raise ValueError("Refusal row must have empty confidence")
            continue
        for choice in choices:
            gallery_id = choice.get("gallery_id")
            if gallery_id not in top_by_query[query_id]:
                raise ValueError(f"Candidate outside top-10: {query_id}, {gallery_id}")
            confidence = float(choice["confidence"])
            if not 0 <= confidence <= 1:
                raise ValueError(f"Invalid confidence: {confidence}")
            cosine = scores[query_index[query_id], gallery_index[gallery_id]]
            expected_confidence = float(np.clip((cosine + 1) / 2, 0, 1))
            if abs(confidence - expected_confidence) > 1e-5:
                raise ValueError(f"Confidence disagrees with embeddings for {query_id}")
    print(f"Validated {len(query)} queries, {len(gallery)} gallery images, {vectors.shape[1]} dimensions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission_resnet50_256"))
    args = parser.parse_args()
    validate(args.archive, args.output)


if __name__ == "__main__":
    main()

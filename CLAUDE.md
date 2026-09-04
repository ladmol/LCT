# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository currently contains only planning documentation — no application code has been written yet. There are no build, lint, or test commands to run. Once code exists (per the phases in `docs/dev-plan.md`), this file should be updated with the actual commands (likely `pytest`, a linter, and `docker compose up` for local Qdrant/Postgres, given the chosen stack below).

## What this project is

Vehicle Re-Identification (Re-ID) service for the ЛЦТ 2026 hackathon (Falcon Tech task): builds a visual "fingerprint" of a vehicle (body type, color, damage, stickers, wheels, tint, etc.) that does **not** rely on the license plate, so the same vehicle can be matched across cameras/angles/lighting even when the plate is unreadable. When a plate was read for at least one event in a matched cluster, the service proposes it as the likely plate for the other events in that cluster.

## Documentation map

The design lives in three docs that together define the system — they cross-reference each other, so read all three before writing code:

- `docs/architecture.md` — pipeline components, data-flow diagram (Mermaid), matching logic, fingerprint vector format, and the DB schema split (vector DB fields vs. relational metadata fields).
- `docs/tech-stack.md` — chosen technology per component, GPU/CPU requirements for training vs. inference, per-stage latency budget.
- `docs/dev-plan.md` — build plan for a 2-person team (Track A: ML/Data, Track B: Backend/Infra/Frontend) working in parallel, with 4 sync points (m1–m4) where the API/embedding contract is fixed. Don't change the embedding format or API contract shape outside those sync points without checking both `architecture.md` and `tech-stack.md`.

## Architecture at a glance

```
Frame -> YOLOv8 detector -> crop/preprocess (OpenCV)
      -> Re-ID embedder (FastReID: OSNet/ResNet50, optionally TransReID)
         + attribute heads (color/body type/viewpoint)
      -> fused fingerprint vector (fixed-dim, L2-normalized)
      -> Vector DB (Qdrant or FAISS) + relational metadata (Postgres/SQLite)
      -> ANN search -> spatio-temporal/attribute prefilter -> k-reciprocal re-ranking
      -> ranked candidates, with plate propagated from any clustered event that had a readable plate
```

Everything — ML pipeline, backend, and demo UI — is intended to be Python, to avoid cross-language integration overhead during the hackathon (`docs/tech-stack.md` §1). Backend is a single FastAPI service exposing `/extract`, `/search`, `/register_plate` (full contracts in `docs/architecture.md` §3.8). Demo UI is Streamlit/Gradio calling that API directly — no separate frontend build step is planned.

## Key constraints to preserve when implementing

- **No training from scratch.** Detector (YOLOv8) and plate OCR (EasyOCR/PaddleOCR) are used pretrained, as-is. The Re-ID model starts from FastReID/torchreid pretrained weights (VeRi-776/VehicleID) and is fine-tuned — never trained from zero.
- **Vector DB vs. relational DB split.** The vector index stores only `fingerprint_id` + `vector`. Everything descriptive (`camera_id`, `timestamp`, `bbox`, `track_id`, `plate_number`, `plate_confidence`, `color`, `body_type`, `viewpoint`, `cluster_id`) lives in the relational DB, joined by `fingerprint_id`. Don't push metadata fields into the vector store or vice versa (`docs/architecture.md` §5).
- **`cluster_id` -> `plate_number` is the core mechanism** for the "suggest a plate for unreadable events" feature: if any event in a cluster has a recognized plate, it's proposed for the rest of the cluster. Any matching/clustering logic must populate `cluster_id` consistently for this to work.

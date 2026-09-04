# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

The repo has a scaffold (`backend/`, `ml/`, `frontend/` — see below) but no real implementation yet: modules under `backend/app/` and `ml/` are empty placeholder files. Fill them in per the phases in `docs/dev-plan.md`; there's no lint/test config set up yet, so add it (e.g. `pytest`, `ruff`) as part of Segment 1/2 rather than assuming it already exists.

## Commands

- `backend/`, `ml/` — Python via [uv](https://github.com/astral-sh/uv), pinned to **3.13** (`.python-version`; capped at `<3.14` because PaddlePaddle/PaddleOCR has no `cp314` wheels yet — see `docs/tech-stack.md`). Setup: `cd backend && uv sync` / `cd ml && uv sync`. Run a script: `uv run <file>.py`.
- `frontend/` — React + TS via [pnpm](https://pnpm.io/). Setup: `cd frontend && pnpm install`. Dev server: `pnpm dev`. Build: `pnpm build`.
- `docker compose up` (repo root) — starts Qdrant + PostgreSQL for local development, per `docker-compose.yml` / `.env.example`.

## What this project is

Vehicle Re-Identification (Re-ID) service for the ЛЦТ 2026 hackathon (Falcon Tech task): builds a visual "fingerprint" of a vehicle (body type, color, damage, stickers, wheels, tint, etc.) that does **not** rely on the license plate, so the same vehicle can be matched across cameras/angles/lighting even when the plate is unreadable. When a plate was read for at least one event in a matched cluster, the service proposes it as the likely plate for the other events in that cluster.

## Documentation map

The design lives in three docs that together define the system — they cross-reference each other, so read all three before writing code:

- `docs/architecture.md` — pipeline components, data-flow diagram (Mermaid), matching logic, fingerprint vector format, and the DB schema split (vector DB fields vs. relational metadata fields).
- `docs/tech-stack.md` — chosen technology per component, GPU/CPU requirements for training vs. inference, per-stage latency budget.
- `docs/dev-plan.md` — build plan for a 2-person team (Track A: ML/Data, Track B: Backend/Infra/Frontend) working in parallel, with 4 sync points (m1–m4) where the API/embedding contract is fixed. Don't change the embedding format or API contract shape outside those sync points without checking both `architecture.md` and `tech-stack.md`.

## Architecture at a glance

Ingestion is **push, not stream**: cameras sit behind existing ПАК (recognition units) that already detect/crop the vehicle and often already OCR the plate themselves. This service never pulls RTSP/video — it only receives already-cropped-or-croppable frames over HTTP. Shared feature-extraction pipeline used by both endpoints below:

```
Frame -> YOLOv8 detector -> crop/preprocess (OpenCV)
      -> Re-ID embedder (FastReID: OSNet/ResNet50, optionally TransReID)
         + attribute heads (color/body type/viewpoint)
      -> fused fingerprint vector (fixed-dim, L2-normalized)
```

- `POST /extract` — the camera/ПАК push endpoint. Runs the pipeline above, **saves** the event to the Vector DB + relational metadata (Postgres/SQLite), then auto-searches history (ANN -> spatio-temporal/attribute prefilter -> k-reciprocal re-ranking) to assign an existing `cluster_id` or create a new one. This is also how the hackathon demo populates its "history" — `ml/scripts/populate_vector_db.py` loops over a gallery dataset calling this same endpoint, simulating camera events, rather than writing to the DBs directly.
- `POST /search` — the operator-facing, **read-only** endpoint (demo UI). Runs the same pipeline + matching logic, but never writes; returns ranked candidates, with a plate suggestion pulled from the matched cluster if any member of it has one.
- `POST /register_plate` — operator manually attaches/confirms a plate for a `cluster_id`.

Full request/response shapes are in `docs/architecture.md` §3.8 (agreed at sync point m1 — see `docs/dev-plan.md`; only the fingerprint dimension is still open, pinned at m2).

ML pipeline and backend are Python (package management via `uv`, one project each in `backend/` and `ml/` — kept separate so backend installs don't pull in heavy ML deps like torch). Demo UI is a React + TypeScript SPA (Vite, package management via `pnpm`, in `frontend/`) calling the backend API directly.

## Key constraints to preserve when implementing

- **`/extract` writes, `/search` doesn't.** Don't make `/extract` stateless (it must persist + auto-cluster) and don't make `/search` persist anything (it's a read-only lookup for the operator). Don't build live RTSP/video-stream ingestion — events arrive as discrete HTTP pushes from the camera/ПАК, per the diagram in `docs/architecture.md` §2.
- **No training from scratch.** Detector (YOLOv8) and plate OCR (EasyOCR/PaddleOCR) are used pretrained, as-is. The Re-ID model starts from FastReID/torchreid pretrained weights (VeRi-776/VehicleID) and is fine-tuned — never trained from zero.
- **Vector DB vs. relational DB split.** The vector index stores only `fingerprint_id` + `vector`. Everything descriptive (`camera_id`, `timestamp`, `bbox`, `track_id`, `plate_number`, `plate_confidence`, `color`, `body_type`, `viewpoint`, `cluster_id`) lives in the relational DB, joined by `fingerprint_id`. Don't push metadata fields into the vector store or vice versa (`docs/architecture.md` §5).
- **`cluster_id` -> `plate_number` is the core mechanism** for the "suggest a plate for unreadable events" feature: if any event in a cluster has a recognized plate, it's proposed for the rest of the cluster. Any matching/clustering logic must populate `cluster_id` consistently for this to work.

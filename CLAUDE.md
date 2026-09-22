# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Docs-only for now. The repo holds the organizers' spec (`docs/task.md`, `docs/task.pdf`), their dataset annotations (`data/specs/`), and the design (`docs/architecture.md`, `docs/tech-stack.md`, `docs/dev-plan.md`). No code yet — build it per the phases in `docs/dev-plan.md`, using the target layout in `docs/architecture.md` §9. There's no lint/test config yet; add it (`ruff`, `pytest`) in phase 0 rather than assuming it exists.

## What this project is

ЛЦТ 2026, task «Фалькон Тех»: open-set vehicle re-identification **without the license plate**. For each query crop (image + given bbox), rank the gallery by likelihood of being the same vehicle, and refuse when there's no confident match. Test vehicles never appear in train.

Scoring (`docs/task.md` §9): mAP@10 from `submission.csv`, cross-camera only (45%), speed at batch=1 and batched FPS plus weights ≤ 2 GB (20%), engineering/Docker/docs (15%), refusal F1/TNR (10%), defense (10%). UI, Grad-CAM and 10⁶-scale ANN are tie-breakers only (0 points).

## Data (`data/specs/`)

- `train.csv`: `image_id, x, y, w, h, vehicle_id, camera_id` — 9 556 rows, 1 541 IDs, 96 cameras.
- `test_query.csv` (1 110) and `test_gallery.csv` (750): `image_id, x, y, w, h`. `image_id` is the query/gallery identifier.
- One vehicle per frame.
- `images/` (~7 GB JPEG) is gitignored and must be unpacked locally.
- `camera_id` exists only in train. Use it for cross-camera validation and sampling, never as an inference input.

## Submission artifacts (`data/specs/README.md`)

- `submission.csv` (no header): `query_id, gallery_id_1..gallery_id_10`.
- `embeddings.npy`: all query rows (file order), then all gallery rows.
- `candidates.csv` (with header): `query_id, gallery_id, confidence`; refused queries have no rows.

Scoring is done by the organizers' reference script `data/specs/evaluate.py` (protocol summarized in `docs/architecture.md` §4.8):

- mAP@10 from `submission.csv`: same-ID-same-camera junk is stripped from our list before truncating to 10; queries with no match are excluded from mAP.
- Refusal is judged per query, by the top-confidence candidate only.
- `embeddings.npy` metrics are reference-only (plain cosine).

Don't reimplement these metrics. Validation must emit ground truth in the organizers' format (`image_id, vehicle_id, camera_id, split`) and call that script. Don't edit `data/specs/`: it holds the organizers' inputs verbatim.

Organizers run our `predict` container themselves on a **hidden** test split, offline. It must take arbitrary CSV/image paths and bundle all weights.

## Architecture at a glance

See `docs/architecture.md` (what) and `docs/tech-stack.md` (how, with versions). One shared `reid_core` package (preprocess → embed → rank/re-rank → calibrate) is used by:

- training scripts;
- the offline `predict` CLI/container that writes the three artifacts, with no DB and no network;
- the services, via docker compose:
  - `inference` — FastAPI under Granian; nvImageCodec GPU JPEG decode → TensorRT FP16 engine built from ONNX on the target GPU at first start (PyTorch FP16 fallback; not `onnxruntime-gpu`, whose PyPI wheel targets CUDA 13); stateless JPEG + bbox → vector;
  - `api` — FastAPI under Granian, OpenAPI, orchestrates search + online re-ranking + refusal; no torch, doesn't depend on `reid`;
  - `db` — PostgreSQL 18 + pgvector 0.8: `halfvec(512)` + metadata;
  - `frontend` — React 19 + Vite 8 SPA behind nginx.

The embedding is always **512-d, L2-normalized**, via a projection head, whatever the backbone. Which model ships (big CLIP ViT-B/16 vs a small one) is decided by measurement against a 5% relative mAP@10 budget (`docs/tech-stack.md` §2.3), not up front.

## Key constraints to preserve

- **Out of scope per spec:**
  - vehicle detection (bbox is given);
  - plate OCR or any plate-derived feature (disqualification);
  - tracking;
  - spatio-temporal filtering (no camera/time at test);
  - stream ingestion.
- **Don't reconstruct hidden info from pixels** (e.g. pseudo-camera from background) without explicit organizer approval. It's listed as a gray zone in `docs/architecture.md` §10.
- **No training from scratch.** Start from public, ungated, MIT/Apache-licensed pretrained weights via timm (CLIP ViT-B/16, SigLIP 2, DINOv2 ViT-S, ConvNeXt-T, R50-IBN) and fine-tune. Every external weight and dataset, with version and licence, goes into `README.md`.
- **No DINOv3 and no request-access datasets** (VeRi-776, VERI-Wild, VehicleID, CityFlow, or checkpoints trained on them) until the organizers approve; the baseline must not depend on them.
- **Functional preprocessing parity.** Crop/pad/resize/normalize code is shared from `reid_core` everywhere. JPEG decoders legitimately differ (crop cache in training, nvJPEG in inference), so validation metrics must come only from the real `predict` → `evaluate.py` path, and a test keeps `/v1/embed` and `predict` embeddings at cosine ≥ 0.999.
- **Speed claims are measured.** `benchmark` must time exactly the shipped config (precision, TTA flag) in both modes: crop → vector and JPEG + bbox → vector.
- **Refusal threshold τ** is chosen on the open-set validation split with distractor queries and shipped as an artifact next to the weights. It is never fitted on test.
- **Don't change the embedding format or API contract** outside the sync points (m1–m4) in `docs/dev-plan.md`.

## Tooling

- **Python 3.13** everywhere, one [uv](https://github.com/astral-sh/uv) workspace (`reid`, `services/*`) with a single `uv.lock`. Not 3.14: `faiss-gpu-cu12` needs `<3.14`.
- Frontend via [pnpm](https://pnpm.io/).
- Everything runs in Docker for the submission (`docker compose up`). CUDA is a build arg (`CUDA_FLAVOR=cu126|cu130`). Don't use `torch-tensorrt`: go ONNX → TensorRT directly.
- ruff (lint + format), pytest; `ty` is dev-only while it's beta.

## Working conventions

- **Never install tools or dependencies without asking first.** Don't run `uv add`, `uv add --dev`, `pnpm add`, `pip install`, or any other package-manager install/upgrade/remove command yourself. Propose the exact command and wait for explicit approval. Read-only or lockfile-sync commands (`uv sync`, `uv run`, `pnpm install` against an existing lockfile, `pnpm dev`, `pnpm build`) are fine.

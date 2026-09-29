# Repository guide

## Project

Vehicle Re-Identification for the ЛЦТ 2026 Falcon Tech task. The current
production feature is an L2-normalized 2816-dimensional ensemble embedding.
License plates are excluded from inputs, storage, training features and APIs.

## Commands

```bash
cd ml && uv sync --locked && pwsh scripts/check.ps1
cd backend && uv sync --locked && uv run pytest -q
cd frontend && pnpm install --frozen-lockfile && pnpm build
docker compose config
```

Run the local demo with `docker compose up --build`, or start backend and
frontend separately as described in their README files.

## Contracts

- `ML_BACKEND_CONTRACT.md` defines the request, response and model version.
- The API receives one full image and one or more `bbox_xywh` entries.
- BBox order and `detection_id` values must be preserved in the response.
- Compare vectors only when `model_version` and `embedding_dim` match.
- A new model requires a new version and re-indexing of the gallery.
- Models load from local checkpoints and inference must not require internet.

## Data

Raw datasets, downloaded images, review sheets and credentials stay outside
Git. External images may enter training only after manual review and masking of
license plates and faces. Deployment weights are tracked through Git LFS.

## Structure

- `ml/reid`: preprocessing, models, training and inference;
- `ml/eval`: metrics and refusal calibration;
- `ml/scripts`: reproducible experiments and submission tools;
- `backend`: FastAPI adapter around the real ML ensemble;
- `frontend`: React demo UI;
- `docs`: current architecture, stack and follow-up plan.

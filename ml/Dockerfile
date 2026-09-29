FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY attributes ./attributes
COPY eval ./eval
COPY reid ./reid
COPY scripts ./scripts
COPY weights ./weights

ENTRYPOINT ["python", "-m", "scripts.make_submission"]
CMD ["--archive", "data/dataset.zip", "--checkpoint", "weights/convnext_tiny_256_fp16.pt", "--second-checkpoint", "weights/resnet50_256_fp16.pt", "--first-weight", "0.5", "--tta-flip", "--refusal-threshold", "0.5051871538", "--output", "outputs/submission", "--batch-size", "16"]

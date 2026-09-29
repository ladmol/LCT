FROM ghcr.io/astral-sh/uv:0.9.17 AS uv
FROM python:3.13-slim

COPY --from=uv /uv /uvx /bin/
WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --locked --no-dev
COPY backend/app ./app
COPY ml/reid /app/ml/reid
COPY ml/attributes /app/ml/attributes
COPY ml/weights /app/ml/weights

ENV PATH="/app/backend/.venv/bin:$PATH"
EXPOSE 8000
CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]

# Vehicle Re-ID API

```bash
uv sync --locked
uv run fastapi dev app/main.py
```

- `GET /api/health` — готовность API и наличие весов;
- `POST /api/extract` — изображение и JSON `meta` в multipart/form-data;
- `GET /docs` — интерактивная схема API.

Модель загружается один раз при первом запросе. Полный контракт находится в
`../ML_BACKEND_CONTRACT.md`.

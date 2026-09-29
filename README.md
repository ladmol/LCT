# Vehicle Re-Identification — ЛЦТ 2026

Система создаёт нормированный визуальный признак автомобиля и позволяет
сопоставлять одну машину на разных камерах и ракурсах. Номерные знаки не
распознаются, не сохраняются и не используются.

## Что готово

- ансамбль ConvNeXt Tiny + ResNet50 с flip TTA;
- признак `float32` размерности 2816 и L2-нормой 1;
- реальные deployment-веса в Git LFS;
- FastAPI `POST /api/extract` для полного кадра и массива BBox;
- React demo для drag-and-drop загрузки двух фотографий и их сравнения;
- воспроизводимый submission и его валидатор.

Результат ансамбля на локальном holdout: mAP `0.5699`, Rank-1 `0.4446`,
Rank-5 `0.7207`. Для tracklet из трёх кадров: mAP `0.6444`, Rank-1 `0.5308`.

## Быстрый запуск

После клонирования обязательно загрузите веса:

```bash
git lfs install
git lfs pull
docker compose up --build
```

Интерфейс: `http://localhost:5173`, Swagger API: `http://localhost:8000/docs`.

Локальная разработка без Docker:

```bash
cd backend
uv sync --locked
uv run fastapi dev app/main.py

cd ../frontend
pnpm install --frozen-lockfile
pnpm dev
```

## Структура

- `ml/` — обучение, инференс, метрики, веса и submission;
- `backend/` — HTTP-адаптер реальной ML-модели;
- `frontend/` — демонстрационный интерфейс;
- `ML_BACKEND_CONTRACT.md` — точный формат интеграции;
- `docs/` — архитектура и инструкция команде.

Датасеты, сырые изображения и внутренние инструменты загрузки данных в Git не
попадают.

# Технологический стек

| Слой | Технологии | Назначение |
|---|---|---|
| ML | Python, PyTorch, torchvision | обучение и инференс Re-ID |
| API | FastAPI, Pydantic | контракт изображения и BBox |
| UI | React, TypeScript, Vite | демонстрационный интерфейс |
| Vector DB | Qdrant | cosine top-k для embeddings |
| Metadata | PostgreSQL | события, камеры и время |
| Delivery | Docker Compose, Git LFS | воспроизводимый запуск и веса |

Основной профиль рассчитан на RTX 3060. CPU поддерживается для проверки, но
работает медленнее. Checkpoint содержит только backbone и загружается локально;
при запуске модели интернет не нужен.

# Demo UI

Интерфейс загружает JPEG/PNG, передаёт весь кадр как один BBox в
`POST /api/extract` и показывает версию модели, размерность, норму и часть
полученного цифрового признака.

```bash
pnpm install --frozen-lockfile
pnpm dev
```

Vite проксирует `/api` на `http://127.0.0.1:8000`.

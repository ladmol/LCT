# Технологический стек и технические решения

Документ фиксирует, **на чём и как** строится решение: модели, обучение, инференс, сервисы, инфраструктура, версии. **Что** строится и зачем — в [`architecture.md`](architecture.md), **когда** — в [`dev-plan.md`](dev-plan.md).

Версии проверены 2026-09-22 (PyPI, npm, Docker Hub, документация NVIDIA). Перед фиксацией lock-файлов версии не понижаем без причины: приоритет — актуальные стабильные релизы.

**Обозначения:**

- **[изм.]** — цифра измерена нами на данных датасета.
- **[лит.]** — цифра из статьи или model zoo (ссылки в §11). Результаты на VeRi/VERI-Wild с нашим тестом напрямую не сравнимы.
- **[оценка]** — порядок величины. Гипотеза, которую проверяем замером в фазе 0.

## 0. Сводка решений

| Слой | Выбор | Главная причина |
|---|---|---|
| Язык | **Python 3.13** везде (ML, inference, api) | `faiss-gpu-cu12` требует `<3.14`, у torchvision исключена 3.14.1; 3.13 поддерживают все нужные пакеты |
| Пакеты | **uv 0.12**, один workspace, один `uv.lock` | Воспроизводимость; `api` ставится без torch через `uv sync --package` |
| Обучение | **PyTorch 2.14**, **timm 1.0.29**, bf16 AMP, `torch.compile` | Все backbone-кандидаты с открытыми весами доступны через timm |
| Модель | Две модели одним кодом: **большая** (CLIP ViT-B/16) и **малая** (ViT-S / ConvNeXt-T / R50-IBN). Малую везём, если она укладывается в бюджет точности (§2.3) | Скорость — 20% оценки, точность — 45%: решаем замером, а не заранее |
| Эмбеддинг | **D = 512**, L2-норма, хранение float16 | Проекционная голова: смена backbone не меняет контракт и схему БД |
| Инференс | **ONNX → TensorRT 11.3 FP16**. Engine собирается на целевом GPU при первом старте. Fallback — **PyTorch FP16** (веса в safetensors) | Максимум скорости без привязки к нашей видеокарте; fallback работает в обоих вариантах CUDA без дополнительных пакетов |
| Декодирование | **nvImageCodec 0.9** (nvJPEG на GPU). Fallback — **PyTurboJPEG 2.5** с DCT-scaling ½ | При батче 1 декодирование кадра 1920×1080 сопоставимо по времени с моделью |
| Поиск (офлайн) | torch-матрица сходства на GPU + k-reciprocal re-ranking | Тест маленький (1 110 × 750) — точный поиск за миллисекунды |
| Отказ | Логистическая регрессия (**scikit-learn 1.9**) над признаками top-1; τ по F1 из `evaluate.py` | Прямо оптимизирует то, что считают организаторы |
| API | **FastAPI 0.141** + **Pydantic 2.13** под **Granian 2.8** | OpenAPI из коробки; Granian — быстрый ASGI-сервер на Rust |
| БД | **PostgreSQL 18** + **pgvector 0.8.6** (`halfvec(512)`, HNSW) | Векторы и метаданные в одной СУБД, названа в ТЗ; PostgreSQL есть в реестре отечественного ПО (Postgres Pro) |
| Доступ к БД | **SQLAlchemy 2.0** (async) + **psycopg 3.3** + **pgvector-python 0.5**, миграции **Alembic 1.20** | Стандартный стек, миграции воспроизводимы |
| Frontend | **React 19.3**, **Vite 8.3**, TypeScript, **TanStack Query 5**, **Tailwind CSS 4.3**, nginx | Актуальные мажорные версии; типизированный клиент из OpenAPI |
| Контейнеры | Docker + compose, NVIDIA Container Toolkit, `python:3.13-slim` + CUDA-библиотеки из wheel-пакетов | Лёгкие образы; версия CUDA — build-аргумент (§7) |
| Качество | **ruff 0.16** (lint + format), **pytest**, CI в GitHub Actions; `ty` — только в dev, пока beta | Инженерное качество — 15% оценки |

## 1. Факты, которые определяют выбор

- **Кадры:** все 1920×1080, baseline JPEG, субдискретизация 4:2:0, **без restart markers**, в среднем 0.66 МБ [изм., выборка 286 файлов].
- **Bbox:** медиана ≈ 670×490, минимум ≈ 150 px [изм.]. Кроп всегда крупнее входа сети.
- **Объём:** train — 9 556 кроп, 1 541 ID, 96 камер. Test (публичная часть) — 1 110 query × 750 gallery.
- **Железо организаторов неизвестно.** Закладываемся на один NVIDIA GPU с SM ≥ 7.5 (от T4 и новее — минимум TensorRT 11). Драйвер тоже неизвестен (§7).
- **Скорость меряют двумя числами:** время признака на 1 ТС при батче 1 и FPS в пакетном режиме без роста памяти.

## 2. Модели

### 2.1 Кандидаты backbone

Только публичные веса **без гейтинга** и с разрешительной лицензией. Все загружаются через timm.

| Роль | Модель (timm) | Параметры | Вход | Лицензия весов | Зачем |
|---|---|---|---|---|---|
| Большая | CLIP ViT-B/16 (`vit_base_patch16_clip_224.openai`) | 86M | 256×256 | MIT | CLIP-инициализация — сильнейший публичный старт для ReID: CLIP-ReID ViT-B/16 на VeRi-776 ≈ 83.3 mAP [лит.] |
| Большая, альтернатива | SigLIP 2 ViT-B/16 (`vit_base_patch16_siglip_256.v2_webli`) | 86M | 256×256 | Apache-2.0 | Более новая vision-language модель, нативно 256 px; сравниваем с CLIP на валидации |
| Малая, ViT | DINOv2 ViT-S/14 + registers (`vit_small_patch14_reg4_dinov2.lvd142m`) | 22M | 224×224 (256 токенов) | Apache-2.0 | Примерно в 4 раза меньше FLOPs, чем у ViT-B; сильная self-supervised инициализация |
| Малая, CNN | ConvNeXt-T (`convnext_tiny.fb_in22k`) | 28M | 256×256 | Apache-2.0 / MIT | CNN-вариант того же класса по FLOPs |
| Малая, CNN (рецепт FastReID) | ResNet50-IBN + GeM + BNNeck | 25M | 256×256 | MIT | Проверенный рецепт: SBS R50-IBN на VeRi ≈ 81.9 mAP [лит.]; очень эффективен в TensorRT |

**Исключено по лицензии — DINOv3.** Свежая статья (июль 2026, §11) показала, что DINOv3-ConvNeXt-B — сильнейший backbone для vehicle ReID (88.19 mAP на VERI-Wild Small без re-ranking [лит.]). Но веса выдаются по запросу, а лицензия DINOv3 запрещает использование лицам, попадающим под экспортные ограничения. Для хакатона Правительства Москвы это неприемлемый риск. Используем только при письменном согласии организаторов (вопрос в [`architecture.md`](architecture.md) §10). На защите это хороший аргумент: «сильнейший вариант знаем и сознательно не взяли».

Также не берём: MobileCLIP (лицензия Apple только для исследований), ConvNeXt V2 FCMAE (CC-BY-NC).

### 2.2 Архитектура головы и эмбеддинга

```
backbone → pooling (CLS-токен для ViT / GeM для CNN)
         → BNNeck (BatchNorm1d; для ID-лосса — после BN, для triplet — до BN)
         → Linear → D = 512 → L2-норма → float32 (в БД — float16)
```

- **D = 512** фиксируется на m1 и не зависит от backbone: большая и малая модели выдают одинаковый вектор, схема pgvector не меняется.
- Атрибутные головы (цвет, тип кузова) — только вспомогательный лосс при обучении. В инференсе их нет.

### 2.3 Бюджет точности и правило выбора модели

**Бюджет:** малая модель допустима, если её mAP@10 на валидации (через `evaluate.py`) **не более чем на 5% относительно** ниже лучшей большой. Например, 0.900 → не ниже 0.855. Трактовку «~5%» подтверждает пользователь.

Процедура (фаза 1, решение на m2):

1. Один и тот же код и рецепт, меняется только конфиг backbone. Обучаем большую модель и 2–3 малые из §2.1.
2. Для каждой модели считаем mAP@10, F1/TNR через `predict` → `evaluate.py`, а также latency/FPS через `benchmark` на нашем GPU.
3. Если лучшая малая в бюджете — везём её.
4. Если нет — дистилляция (§2.5) большой в малую, затем повтор шага 3.
5. Если и после этого не укладывается — вопрос пользователю: везти большую или принять потерю.

Почему это вероятно сработает [оценка]: малая модель быстрее в 3–4 раза по пакетному FPS. При батче 1 выигрыш меньше, потому что заметную долю времени занимает декодирование JPEG (§4.1). Типичный разрыв ViT-S и ViT-B в ReID — несколько пунктов mAP, а дистилляция сокращает его ещё.

### 2.4 Обучение

| Параметр | Значение (стартовое, тюнится на валидации) |
|---|---|
| Данные | Кэш кропов: bbox + 10% отступа, короткая сторона 320 px, без потерь (PNG). Эпоха идёт без декодирования 1080p |
| Семплер | PK: P = 16 ID × K = 4 кадра; внутри ID по возможности разные `camera_id` |
| Лоссы | CE с label smoothing 0.1 (после BNNeck) + triplet batch-hard (soft margin) до BNNeck; альтернатива triplet — circle loss |
| Оптимизатор | AdamW, weight decay 0.05, layer-wise LR decay 0.65–0.75 для ViT, warmup 5 эпох + cosine |
| LR | Для CLIP/SigLIP backbone порядка 5e-6…1e-5 (CLIP «ломается» от больших LR), для головы ×10; для CNN — 3.5e-4 |
| Эпохи | 60–120 (данных мало; ранняя остановка по mAP@10 на валидации) |
| Точность вычислений | bf16 autocast, `torch.compile`, SDPA / flash attention |
| Аугментации | resize + pad + random crop, горизонтальный flip, random erasing (p = 0.5), лёгкая яркость/контраст, blur, понижение разрешения, JPEG-артефакты, grayscale (p = 0.05, ночь). **Без hue-jitter** — цвет является признаком |
| Маскирование номера | Случайное затирание нижней центральной области кропа (где обычно номер) — аргумент против дисквалификации |
| Внешние датасеты | VeRi-776, VERI-Wild, VehicleID, CityFlowV2 — **только после ответа организаторов**, что доступ «по запросу» считается открытым. Базовая линия от них не зависит |
| Трекинг экспериментов | Локальные логи (TensorBoard-формат) + JSON-отчёты `evaluate.py` в `reports/`; сплиты и seed — в git |

Конфигурация — YAML-файлы, которые валидируются Pydantic-моделями. Один конфиг полностью описывает эксперимент.

### 2.5 Дистилляция (только если не проходит бюджет из §2.3)

- **Учитель** — обученная большая модель, заморожена, bf16, прямой проход на лету (аугментации совпадают).
- **Ученик** — малая модель с той же головой D = 512.
- **Лосс ученика:**
  - ReID-лоссы из §2.4;
  - косинусный лосс между эмбеддингами ученика и учителя;
  - relational KD: KL-дивергенция между матрицами сходства батча у ученика и учителя.
- Одинаковое D позволяет сравнивать векторы напрямую, без адаптеров.

## 3. Поиск, re-ranking, отказ

### 3.1 Офлайн (`predict`)

1. Эмбеддинги query и gallery. Flip-TTA включается, только если на валидации даёт ≥ 0.5 п. mAP@10. Если включён, то включён **и в бенчмарке**.
2. Косинусная матрица на GPU (fp32).
3. k-reciprocal re-ranking на GPU (своя реализация на torch). Стартовые k1 = 20, k2 = 6, λ = 0.3 [лит.] тюнятся на валидации: галерея маленькая, и у запроса обычно 1–3 позитива, поэтому оптимальные k1/k2, скорее всего, меньше.
4. Опционально DBA / α-QE — оставляем, только если дают прирост на валидации.
5. `submission.csv`: top-10. Если организаторы разрешат длинные строки — top-K (§4.8 архитектуры).

### 3.2 Онлайн (`api`)

- pgvector HNSW → top-100 кандидатов → k-reciprocal **внутри этого подмножества**: запрос + 100 кандидатов, матрица 101×101 — доли миллисекунды.
- Полный re-ranking по всей галерее онлайн не делаем: он требует соседей всех объектов и не масштабируется на 10⁶.

### 3.3 Уверенность и порог

- **Признаки запроса:**
  - сходство top-1 (после re-ranking);
  - отрыв top-1 от top-2;
  - взаимность: ранг запроса в списке соседей top-1;
  - средняя близость top-5 (насколько плотное окружение).
- **Модель:** `LogisticRegression` (scikit-learn 1.9). Цель — «верхний кандидат верен», ровно TP из `evaluate.py`.
- **Порог τ** — максимум F1 по определению `evaluate.py` на валидации с дистракторами. Устойчивость проверяется по фолдам и бутстрепу при разных долях дистракторов.
- **Два калибратора** — для офлайн- и онлайн-ранжирования, потому что распределения скоров у них разные. Хранятся как JSON (коэффициенты + τ) рядом с весами, без pickle.

## 4. Инференс и скорость

### 4.1 Куда уходит время

Порядки величин [оценка]; реальные числа — первый замер фазы 0 на нашем GPU.

| Этап (один кроп) | CPU | GPU |
|---|---|---|
| Декодирование 1920×1080 JPEG | 1 ядро libjpeg-turbo: ~8–12 мс; с DCT-scaling ½: ~3–4 мс | nvJPEG: ~1–2 мс; аппаратный декодер на A100/H100/L4-классе: тысячи кадров/с в пакете |
| Кроп + resize + нормализация | ~1 мс | < 0.2 мс |
| Сеть, малая (ViT-S / R50), FP16 TensorRT | — | ~0.5–1 мс |
| Сеть, большая (ViT-B/16 @256), FP16 TensorRT | — | ~2–3 мс |

Выводы:

- **При батче 1** декодирование кадра сопоставимо с сетью, поэтому декодировать нужно на GPU.
- **В пакете** с большой моделью упираемся в сеть, с малой — в декодирование.
- **ROI-декодирование почти не ускоряет:** в файлах нет restart markers, поэтому поток Хаффмана всё равно разбирается целиком, экономятся только IDCT и цветовое преобразование. Надёжные рычаги — GPU-декодер и DCT-scaling ½ на CPU.

### 4.2 Конвейер

```
JPEG bytes + bbox
  → decode   nvImageCodec (nvJPEG; ROI = bbox+pad)      | fallback: PyTurboJPEG, DCT-scale ½ (CPU)
  → crop + resize (bilinear, antialias) + normalize — torch-операции на GPU
  → TensorRT engine (FP16; батч 1 — через CUDA Graph)   | fallback: PyTorch FP16 (safetensors)
  → L2-norm → float32[512]
```

- **Экспорт:** `torch.onnx.export(..., dynamo=True)` → ONNX с динамической осью батча. Проверяем, что выходы ONNX совпадают с PyTorch (cosine > 0.9999) — через CPU-сборку `onnxruntime` в тестах.
- **TensorRT engine собирается на целевом GPU при первом старте** контейнера из ONNX: engine привязан к архитектуре GPU и версии TensorRT, поэтому собрать заранее на нашей карте нельзя.
  - Профили оптимизации: batch 1 и 1–128.
  - Кэш лежит в volume, ключ — (GPU, версия TRT, хэш ONNX).
  - Сборка для ViT-B занимает минуты [оценка]. Healthcheck сервиса ждёт её окончания.
- **Не используем `torch-tensorrt`:** версия 2.13 отстаёт от torch 2.14. Прямой путь ONNX → TensorRT надёжнее.
- **Fallback — PyTorch, а не ONNX Runtime.** `onnxruntime-gpu` 1.30 с PyPI собран под CUDA 13 (extra `cuda` тянет `nvidia-cuda-runtime~=13.0`), поэтому в образе `cu126` он без отдельного фида Microsoft не работает. Torch уже есть в образе в обоих вариантах, так что fallback на нём не требует лишних зависимостей. Если сборка engine упала, модель запускается в FP16 из safetensors.
- **FP8 / INT8** (NVIDIA ModelOpt 0.46, PTQ) — последний рычаг скорости. Включаем, только если mAP@10 падает не больше 0.5 п. и GPU организаторов поддерживает формат (FP8 — Ada/Hopper и новее).
- **Батчинг:**
  - в `inference` — asyncio micro-batcher: собирает до 64 запросов или ждёт до 2 мс;
  - в `predict` — ограниченный конвейер (читатели файлов → GPU-декод батчем → препроцессинг → TRT → запись) с bounded-очередями, поэтому память не растёт (требование ТЗ §7).

### 4.3 Бенчмарк (`benchmark`)

Мерит ровно ту конфигурацию, которая дала mAP в `submission.csv`: тот же engine, точность вычислений и флаг TTA. Два режима, чтобы покрыть обе трактовки «времени формирования признака»:

1. **Кроп → вектор** (уже декодированный и обрезанный тензор на GPU).
2. **JPEG + bbox → вектор** (весь путь, как в сервисе).

Для каждого режима:

- **батч 1** — p50/p95/p99 после прогрева;
- **пакетный режим** — устойчивый FPS за N минут при батче 32/64/128 с мониторингом RSS и VRAM (доказывает отсутствие деградации памяти).

Результат пишется в JSON и попадает в README.

## 5. Сервисы

| Сервис | Стек | Ключевые детали |
|---|---|---|
| `inference` | FastAPI + Granian (1 процесс на GPU), TensorRT, nvImageCodec, torch (как CUDA-рантайм и для препроцессинга) | `POST /v1/embed` (JPEG bytes + bbox) → вектор; micro-batcher; `GET /v1/model`, `/health`. Для `/v1/explain` лениво загружается eager-модель PyTorch — не в горячем пути |
| `api` | FastAPI + Granian (несколько воркеров), httpx (async, keep-alive) → `inference`, SQLAlchemy 2.0 async + psycopg 3.3 + pgvector-python, Alembic, pydantic-settings | **Без torch.** Валидация входа, поиск, онлайн re-ranking (numpy), калибровка, отказ. Кадры хранятся в общем volume, адресация по sha256 |
| `db` | Образ `pgvector/pgvector:0.8.6-pg18-trixie` | `embedding halfvec(512)`; HNSW `halfvec_cosine_ops` (m = 16, ef_construction = 128); `hnsw.ef_search` тюнится; iterative index scans для запросов с фильтрами |
| `frontend` | React 19 + Vite 8 + TS, TanStack Query, Tailwind 4; типизированный клиент генерируется из OpenAPI-схемы `api`; nginx раздаёт статику и проксирует `/api` (один origin, без CORS) | Загрузка кадра, рисование bbox на canvas, выдача с `confidence`, явный отказ, экспорт CSV, тепловая карта |
| `predict` | Образ `inference`, другая точка входа | Без БД и без сети |
| `benchmark` | Образ `inference`, другая точка входа | §4.3 |

**Масштабирование (тай-брейкер):** отдельный скрипт сравнивает pgvector HNSW и **FAISS 1.14 GPU** (CAGRA / IVF-PQ) на 10⁶ векторах: реальные эмбеддинги плюс синтетика. Метрики — recall@10, задержка, память.

## 6. Структура кода (uv workspace)

```
pyproject.toml            # [tool.uv.workspace] members = ["reid", "services/*"]
uv.lock                   # один lock на всё
reid/                     # пакет reid_core: data, models, losses, retrieval, calibration, export, runtime (TRT + torch-fallback)
  configs/                # YAML-конфиги экспериментов
  scripts/                # train, eval, calibrate, export, predict, benchmark
services/inference/       # зависит от reid (extras [runtime]: tensorrt, nvimgcodec)
services/api/             # НЕ зависит от reid: только numpy + веб-стек
frontend/                 # pnpm
weights/                  # итоговые ONNX + калибраторы (JSON) + SHA256SUMS
docker/                   # Dockerfile'ы, entrypoint сборки engine
```

- Extras пакета `reid`: `[train]` (timm, torchvision, sklearn, onnxruntime CPU — для проверки экспорта), `[runtime]` (tensorrt, nvimgcodec, pyturbojpeg).
- В каждом Dockerfile: `uv sync --frozen --no-dev --package <member> --extra …`.

**Паритет препроцессинга.** Обучение читает кэш кропов (libjpeg-turbo → PNG), а инференс декодирует nvJPEG. Пиксели немного отличаются, это ожидаемо. Требование другое — **функциональный паритет**:

1. Валидационные метрики считаются только через настоящий путь `predict` → `evaluate.py`.
2. Тест сравнивает `/v1/embed` и `predict` на одних и тех же кропах: cosine ≥ 0.999.

Функции crop/resize/normalize — одни и те же из `reid_core` везде.

## 7. Контейнеры, CUDA, веса

- **База:** `python:3.13-slim-trixie` + uv. CUDA-библиотеки приходят из wheel-пакетов (torch, `tensorrt-cu12`/`-cu13`, `nvidia-nvimgcodec-cu12`/`-cu13`). Драйвер пробрасывает NVIDIA Container Toolkit; в compose — резервирование GPU для `inference`, `predict`, `benchmark`.
- **Версия CUDA — build-аргумент `CUDA_FLAVOR`:**
  - `cu126`: torch 2.14+cu126, TensorRT cu12. Работает на драйверах R560+ (R525+ через minor-version compatibility), но **не поддерживает Blackwell** (RTX 50xx, B200).
  - `cu130`: torch 2.14+cu130, TensorRT cu13. Нужен драйвер R580+, поддерживает Blackwell.

  Выбираем, когда организаторы сообщат GPU и драйвер; до этого разрабатываем на `cu126`.

  Оба набора проверены резолвом без установки (`uv pip compile`, linux x86_64, Python 3.13), 2026-09-22:
  - `cu126`: torch 2.14.0+cu126, `tensorrt-cu12` 11.3.0.99, `nvidia-nvimgcodec-cu12` 0.9.0.20, cuDNN 9.10;
  - `cu130`: torch 2.14.0+cu130, `tensorrt-cu13` 11.3.0.99, `nvidia-nvimgcodec-cu13` 0.9.0.20, cuDNN 9.24.

  Совместимость в рантайме проверяется в фазе 0 при фиксации `uv.lock`.
- **Веса:**
  - в репозитории лежат ONNX (FP16) и safetensors (для fallback) плюс калибраторы; всё коммитится **прямо в git, без Git LFS**;
  - файлы > 100 МБ (лимит GitHub) шардируются: ONNX — через external data, safetensors — на несколько частей. Малая модель обычно помещается в один файл, большая (~170 МБ в FP16) — в два;
  - Git LFS не используем: бесплатная квота трафика GitHub LFS заканчивается за несколько клонов, а если жюри не скачает веса, решение «не воспроизводится» — это дисквалификация;
  - при старте проверяется `SHA256SUMS`;
  - `HF_HUB_OFFLINE=1`, никаких загрузок в рантайме.
- **Порядок старта:** healthchecks и `depends_on: condition: service_healthy` (сначала `db`, затем сборка engine в `inference`, затем `api`).

## 8. Качество кода и CI

- **ruff 0.16** — lint и format, одна конфигурация в корневом `pyproject.toml`.
- **Типизация** — `ty` в dev-режиме (пока beta), в CI не блокирует.
- **pytest:**
  - unit-тесты на CPU: форматы артефактов, re-ranking, калибровка, валидация API;
  - GPU-тесты под маркером: паритет, TRT против ONNX.
- **GitHub Actions:** lint, тесты на CPU, сборка фронтенда, `docker build` всех образов (без GPU).
- **Frontend:** `tsc --noEmit`, линтер, `vite build`.

## 9. Рассмотрено и отклонено

| Вариант | Почему нет |
|---|---|
| NVIDIA Dynamo-Triton 2.72 (бывший Triton Inference Server) | Даёт dynamic batching и perf_analyzer из коробки, но это тяжёлый образ NGC, привязанный к свежей CUDA (риск по драйверу у организаторов), и отдельный язык конфигурации. Наш inference — ~200 строк Python на тех же TensorRT и nvJPEG. Остаётся запасным вариантом для масштабирования |
| `torch-tensorrt` | Отстаёт от torch (2.13 против 2.14) |
| DINOv3 | Лицензия и гейтинг (§2.1) |
| CLIP-ReID целиком (двухэтапное обучение с текстовыми промптами) | Сложнее, прирост около 1–2 mAP [лит.]; SIE/OLP требуют `camera_id`, которого нет на тесте. Можно добавить в фазе 3 при запасе времени |
| Qdrant / Milvus | Лишний сервис; pgvector закрывает и метаданные, и векторы |
| Объектное хранилище (MinIO и т. п.) | Хватает docker volume |
| Python 3.14 | Нет `faiss-gpu-cu12`, у torchvision исключена 3.14.1 |

## 10. Версии (проверено 2026-09-22)

| Пакет | Версия | | Пакет | Версия |
|---|---|---|---|---|
| Python | 3.13 | | onnxruntime (CPU, только тесты) | 1.30.0 |
| uv | 0.12.17 | | nvidia-modelopt | 0.46.1 |
| torch / torchvision | 2.14.0 / 0.29.0 | | nvidia-nvimgcodec-cu12 | 0.9.0 |
| timm | 1.0.29 | | pyturbojpeg | 2.5.0 |
| open-clip-torch (если понадобится текстовая ветка) | 3.3.0 | | faiss-gpu-cu12 | 1.14.1 |
| tensorrt-cu12 / -cu13 | 11.3.0 | | scikit-learn | 1.9.1 |
| fastapi | 0.141.1 | | pydantic | 2.13.5 |
| granian | 2.8.3 | | sqlalchemy / alembic | 2.0.54 / 1.20.0 |
| psycopg | 3.3.6 | | pgvector (python) | 0.5.0 |
| PostgreSQL + pgvector | 18 + 0.8.6 | | ruff | 0.16.8 |
| react | 19.3.0 | | vite | 8.3.0 |
| @tanstack/react-query | 5.103.2 | | tailwindcss | 4.3.3 |

Точные версии фиксируются в `uv.lock` и `pnpm-lock.yaml`; эта таблица — ориентир на старте.

## 11. Источники

- DINOv3-ConvNeXt для vehicle ReID, протокол cross-camera, sparse re-ranking: [arXiv 2607.22068](https://arxiv.org/abs/2607.22068)
- Лицензия DINOv3: [facebookresearch/dinov3 LICENSE](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- CLIP-ReID (AAAI 2023), код под MIT: [github.com/Syliz517/CLIP-ReID](https://github.com/Syliz517/CLIP-ReID), [arXiv 2211.13977](https://arxiv.org/abs/2211.13977)
- FastReID model zoo (SBS / BoT R50-IBN на VeRi / VERI-Wild): [MODEL_ZOO.md](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md)
- CLIP-SENet (VeRi 92.9 mAP — несопоставимо без проверки протокола): [arXiv 2502.16815](https://arxiv.org/abs/2502.16815)
- TensorRT 11.3, поддержка GPU SM ≥ 7.5: [support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- nvImageCodec, ROI-декодирование: [docs](https://docs.nvidia.com/cuda/nvimagecodec/samples/code_stream.html)
- Dynamo-Triton: [developer.nvidia.com/dynamo-triton](https://developer.nvidia.com/dynamo-triton)
- pgvector (релизы, halfvec, HNSW, iterative scans): [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)

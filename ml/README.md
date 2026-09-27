# ML: поиск одного автомобиля на кадрах разных камер

Этот каталог содержит воспроизводимый Vehicle Re-ID конвейер: обучение,
локальную оценку, построение цифрового признака и экспорт конкурсных файлов.
Сервис получает полный JPEG и готовые BBox автомобилей в формате
`(x, y, width, height)`. Детектор и OCR находятся за границей этого модуля.

## Рекомендуемая конфигурация

Основной режим по качеству — ансамбль двух моделей:

- `outputs/convnext_tiny_256/best.pt`, ConvNeXt Tiny, 256×256;
- `outputs/resnet50_256_finetune/best.pt`, ResNet50, 256×256;
- оригинальный и отражённый по горизонтали кроп усредняются отдельно в каждой
  модели;
- два L2-нормированных признака объединяются с равными весами в один вектор
  `float32` размерности 2816;
- стартовый порог принятия по cosine similarity: `0.5146225095`.

На фиксированном holdout ансамбль получил mAP `0.5630`, Rank-1 `0.4398`,
Rank-5 `0.7127`, F1 принятого top-1 `0.4336` и TNR неизвестных машин `0.7414`.
На RTX 3060 полный путь от JPEG и BBox до вектора работает примерно со
скоростью 27,5 объекта/с при batch 16, пиковая CUDA-память — 444 МБ.

Если важнее скорость, можно использовать только ConvNeXt без отражения:
вектор 768, mAP `0.5255`, Rank-1 `0.4077`, около 63 объектов/с и 275 МБ CUDA.
Это отдельная версия модели, поэтому смешивать её векторы с ансамблем нельзя.

Полная таблица опытов и отрицательные результаты находятся в
[`EXPERIMENTS.md`](EXPERIMENTS.md). Контрольные суммы локальных checkpoint и
готового экспорта зафиксированы в [`MODEL_ARTIFACTS.md`](MODEL_ARTIFACTS.md).

## Как создаётся цифровой признак

```text
полный JPEG + BBox xywh
  → обрезка автомобиля с проверкой координат
  → сохранение пропорций и дополнение до квадрата
  → resize 256×256 и ImageNet-нормировка
  ├→ ConvNeXt Tiny для оригинала и отражения → средний L2-вектор 768
  └→ ResNet50 для оригинала и отражения      → средний L2-вектор 2048
  → [sqrt(0.5) · ConvNeXt, sqrt(0.5) · ResNet50]
  → один L2-нормированный float32-вектор 2816
```

Множители `sqrt(0.5)` нужны, чтобы скалярное произведение итоговых векторов
равнялось среднему cosine similarity двух моделей. Норма результата остаётся
около единицы.

Во время обучения сеть видит `vehicle_id`. Classification loss учит различать
обучающие экземпляры, а batch-hard triplet loss сближает разные камеры одного
автомобиля и раздвигает разные автомобили. Классификатор после обучения не
используется. Каждая координата признака не означает отдельное понятие вроде
«царапина» или «фара»: модель сама кодирует полезные сочетания формы, деталей,
цвета и локальных особенностей. Новые автомобили можно искать без добавления
их в список классов.

## Протокол проверки

В `data/dataset.zip` находятся 9 556 обучающих строк для 1 541 автомобиля,
1 110 test query и 750 test gallery. Архив читается напрямую, распаковка не
нужна. При `seed=42` идентичности делятся на train/dev/holdout в отношении
70/15/15 без пересечений. Для известной машины gallery берётся с одной камеры,
а query — с других. Четверть holdout-идентичностей удаляется из gallery и
проверяет способность отказать.

Вес ансамбля, TTA и порог выбирались на dev. Holdout использовался только для
итогового отчёта. Это локальная оценка; скрытый тест может иметь другое
распределение камер и автомобилей.

- mAP, Rank-1 и Rank-5 оценивают ранжирование известных машин;
- F1 учитывает правильность принятого top-1;
- TNR показывает долю правильных отказов для машин без совпадения;
- `confidence = (cosine + 1) / 2` в CSV является score, а не вероятностью.

## Структура

| Путь | Назначение |
| --- | --- |
| `reid/data.py` | Чтение ZIP, BBox, преобразования и батчи разных камер |
| `reid/model.py` | ResNet18, ResNet50, ConvNeXt Tiny и triplet loss |
| `reid/train.py` | Обучение, выбор эпохи и локальная оценка |
| `reid/infer.py` | API одной модели и ансамбля для backend |
| `reid/protocol.py` | Разделение идентичностей и cross-camera протокол |
| `eval/metrics.py` | mAP, Rank-1/5, F1, TNR и подбор порога |
| `scripts/make_submission.py` | Экспорт top-10, embeddings и candidates |
| `scripts/validate_submission.py` | Проверка форматов и соответствия файлов |
| `scripts/eval_inference_variants.py` | Сравнение TTA и wide-crop |
| `scripts/eval_checkpoint_ensemble.py` | Выбор веса ансамбля только на dev |
| `scripts/benchmark_reid.py` | Скорость полного пути JPEG → вектор |
| `scripts/analyze_errors.py` | Срезы ошибок и визуальный лист промахов |
| `reid/pretrain_external.py` | Эксперимент с внешними фото и атрибутами |
| `data/` | Локальные данные и веса, исключены из Git |
| `outputs/` | Checkpoint, отчёты и экспорты, исключены из Git |

Локальный сборщик внешних изображений также исключён из Git. В репозиторий не
попадают ни исходные фотографии, ни его журналы. Для обучения допускаются
только вручную проверенные изображения с закрытыми номерами и лицами.

## Установка

Поддерживается Python 3.11–3.13. Команды выполняются из каталога `ml`.

```powershell
uv sync --locked --cache-dir .uv-cache --no-managed-python
pwsh -File scripts/check.ps1
```

Скрипт последовательно запускает Ruff, проверку форматирования, базовую проверку
типов Re-ID-кода, тесты и компиляцию Python-модулей. На Linux те же проверки
доступны через `make check`.

Официальные начальные веса torchvision загружаются один раз:

```powershell
uv run python -m scripts.prepare_weights --arch convnext_tiny
uv run python -m scripts.prepare_weights --arch resnet50
```

После этого обучение и инференс в сеть не обращаются.

## Обучение

ConvNeXt Tiny:

```powershell
uv run python -m reid.train --archive data/dataset.zip `
  --output outputs/convnext_tiny_256 --arch convnext_tiny `
  --init-weights data/weights/convnext_tiny_imagenet.pth `
  --epochs 12 --image-size 256 --eval-batch-size 16
```

ResNet50 и дополнительное дообучение:

```powershell
uv run python -m reid.train --archive data/dataset.zip `
  --output outputs/resnet50_256 --arch resnet50 `
  --init-weights data/weights/resnet50_imagenet.pth `
  --epochs 12 --image-size 256 --eval-batch-size 16

uv run python -m reid.train --archive data/dataset.zip `
  --output outputs/resnet50_256_finetune --arch resnet50 `
  --resume outputs/resnet50_256/best.pt --epochs 6 `
  --image-size 256 --eval-batch-size 16 --lr 0.00003
```

При `--resume` загружаются веса лучшей эпохи, но оптимизатор создаётся заново.
Число `--epochs` означает количество дополнительных эпох.

## Экспорт лучшего ансамбля

```powershell
uv run python -m scripts.make_submission `
  --archive data/dataset.zip `
  --checkpoint outputs/convnext_tiny_256/best.pt `
  --second-checkpoint outputs/resnet50_256_finetune/best.pt `
  --first-weight 0.5 --tta-flip `
  --refusal-threshold 0.5146225095 `
  --output outputs/submission_ensemble_balanced --batch-size 16

uv run python -m scripts.validate_submission `
  --archive data/dataset.zip `
  --output outputs/submission_ensemble_balanced
```

Экспорт уже создан и проверен локально: 1 110 query, 750 gallery, 2816
координат. Папка `outputs/submission_ensemble_balanced/` содержит:

- `submission.csv` — десять ID gallery для каждого query;
- `embeddings.npy` — сначала query, затем gallery в порядке исходных CSV;
- `candidates.csv` — кандидаты выше порога или пустая строка отказа.

## API для backend

```python
from reid.infer import VehicleEnsembleEmbedder

embedder = VehicleEnsembleEmbedder(
    "outputs/convnext_tiny_256/best.pt",
    "outputs/resnet50_256_finetune/best.pt",
    first_weight=0.5,
    tta_flip=True,
)

vector = embedder.embed(image_bytes, (x, y, width, height))
# numpy.float32, shape (2816,), L2-норма около 1

vectors = embedder.embed_many(image_bytes, bboxes_xywh)
# numpy.float32, shape (N, 2816), порядок совпадает с bboxes_xywh
```

Checkpoint загружаются один раз при старте процесса. Для нескольких BBox одного
кадра используется `embed_many`. Backend хранит `model_version` рядом с каждым
вектором и сравнивает только одинаковые версии. При смене модели всю gallery
нужно пересчитать. Полный контракт находится в
[`../ML_BACKEND_CONTRACT.md`](../ML_BACKEND_CONTRACT.md).

## Внешние изображения и атрибуты

Текущий пилот содержит 52 исходных изображения из 13 объявлений и 26
проверенных кропов восьми автомобилей. Предобучение ResNet18 на таком объёме
не улучшило holdout: mAP `0.308` против `0.309` у базовой модели. Этого мало
для вывода о полезности марок, моделей и рестайлинга.

Для следующего опыта нужен существенно больший, проверенный набор. Метка одного
объявления может использоваться как предварительный `instance_id`; марка,
модель, поколение и рестайлинг хранятся отдельными полями. До ручной проверки
и маскирования изображение не попадает в обучение. Атрибутные головы остаются
экспериментальными и не входят в обязательный backend-ответ.

## Известные ограничения

- Широкие BBox остаются сложным срезом: у ConvNeXt Rank-1 `0.135`, медианный
  ранг 10. Специальный multi-crop не помог.
- Сильные случайные аугментации ухудшили mAP до `0.382`; они не используются.
- Ансамбль точнее, но примерно в 2,3 раза медленнее быстрого ConvNeXt.
- Порог нужно перекалибровать на реальных целевых камерах до промышленного
  включения автоматического принятия совпадений.

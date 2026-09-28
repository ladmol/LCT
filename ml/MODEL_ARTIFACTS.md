# Артефакты финальной модели

Финальные веса и конкурсные файлы входят в репозиторий. Инференс не скачивает
данные из интернета.

## Deployment-веса

| Файл | Размер, байт | SHA-256 |
| --- | ---: | --- |
| `weights/convnext_tiny_256_fp16.pt` | 55 700 157 | `326019db893d52235f0e079acb5ac95e0e9f5ad960ad08996f1b9079df982312` |
| `weights/resnet50_256_fp16.pt` | 47 223 595 | `fbf7d1a07c5661b1edb1c27fecf114a3349f239062e8eae97fc441f720904cc5` |

Суммарный размер - 102 923 752 байта, около 98 МиБ при лимите хакатона 2 ГБ.
В checkpoint удалены неиспользуемые классификаторы, а backbone хранится в FP16.
При загрузке PyTorch приводит параметры к рабочему dtype. На фиксированном
holdout mAP равен `0,56987`, Rank-1 `0,44462`, Rank-5 `0,72071`.

Проверка:

```powershell
Get-FileHash weights/*.pt -Algorithm SHA256
```

## Конкурсные файлы

| Файл | Размер, байт | SHA-256 |
| --- | ---: | --- |
| `submission/submission.csv` | 404 181 | `73b2cc597278c00c9cf25e48759020e47978ace557f9bb166aa080ee58d6b575` |
| `submission/embeddings.npy` | 20 951 168 | `aa21b9cf42a1e827b5ca94d7a940507ae708e1a335b75c0fbe8408a28db57d5d` |
| `submission/candidates.csv` | 351 812 | `c73703ea1dda3f40ff1cb07a02be298c816974883d0019e4a8c84c75c8b5572b` |

Валидатор подтвердил 1 110 query, 750 gallery, размерность 2816, исходный
порядок строк, единичную норму embeddings и соответствие top-10 матрице cosine.

## Воспроизведение

Положить `dataset.zip` в `data/` и запустить из каталога `ml`:

```powershell
docker compose run --rm inference
```

Или без Docker:

```powershell
uv run python -m scripts.make_submission `
  --archive data/dataset.zip `
  --checkpoint weights/convnext_tiny_256_fp16.pt `
  --second-checkpoint weights/resnet50_256_fp16.pt `
  --first-weight 0.5 --tta-flip `
  --refusal-threshold 0.5051871538 `
  --output outputs/submission --batch-size 16

uv run python -m scripts.validate_submission `
  --archive data/dataset.zip --output outputs/submission
```

# Артефакты финальной модели

Checkpoint и конкурсные экспорты не хранятся в Git. Этот файл фиксирует точные
локальные артефакты, с которыми получены результаты в `EXPERIMENTS.md`.

## Checkpoint ансамбля

| Назначение | Локальный путь | Размер, байт | SHA-256 |
| --- | --- | ---: | --- |
| ConvNeXt Tiny | `outputs/convnext_tiny_256/best.pt` | 114655263 | `d974102dd967301485460403cf63095f0fef9ff8e793c8707b5ace4c4d8dca6d` |
| ResNet50 | `outputs/resnet50_256_finetune/best.pt` | 103175597 | `7e9ab688d8facdba924362d0ca84432718387e1384ba5f3c8a8e8022898e7d91` |

Оба файла нужны для версии `vehicle-reid-ensemble-256-v1`. Их следует
передавать backend как отдельные бинарные артефакты через внутреннее хранилище
или GitHub Release, сохраняя имена и проверяя SHA-256 после скачивания.

Проверка в PowerShell:

```powershell
Get-FileHash outputs/convnext_tiny_256/best.pt -Algorithm SHA256
Get-FileHash outputs/resnet50_256_finetune/best.pt -Algorithm SHA256
```

## Финальный экспорт

Каталог: `outputs/submission_ensemble_balanced/`.

| Файл | Размер, байт | SHA-256 |
| --- | ---: | --- |
| `embeddings.npy` | 20951168 | `529685f96d50ade0173ceac98a2cd047db5785c5133ecc59fdfe66ba6560e00d` |
| `submission.csv` | 404181 | `e11d22403f39f2822549ac02b59a092cd24a8501d26d42bc62f554e2271e0c2c` |
| `candidates.csv` | 361924 | `492a7e9caeb18e307ba96ffc579aa8238e5e1102b60668fd175dcf3b0cd641c5` |

Валидатор подтвердил 1110 query, 750 gallery и размерность признака 2816.

## Воспроизведение экспорта

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

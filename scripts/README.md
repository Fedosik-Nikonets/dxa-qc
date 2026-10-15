# scripts/ — разведка, подбор порогов, метрики и проверки

[← README](../README.md) · [Карта документации](../docs/README.md)

Скрипты пронумерованы в порядке спринта и воспроизводят все цифры из
[`reports/`](../reports/README.md). Полный цикл обучения с нуля и пересчёт
на новых данных — [docs/training.md](../docs/training.md). Почти всем
скриптам нужен набор организатора в `Data/` и окружение разработки
из `requirements.txt`.

```bash
python scripts/15_checkpoint_metrics.py      # пример: метрики всех правил на пяти фолдах
```

## Данные и фундамент

| скрипт | что делает | результат |
|---|---|---|
| [`01_inventory.py`](01_inventory.py) | инвентаризация набора DICOM: метаданные, размеры, масштаб, дубликаты | `reports/inventory_train.csv` |
| [`02_build_dataset.py`](02_build_dataset.py) | датасет уровня «изображение» из меток уровня «исследование × область» | `reports/dataset_images.csv` |
| [`03_make_split.py`](03_make_split.py) | фиксированное разбиение на фолды; единица — исследование, чтобы не было утечки | `reports/split.csv` |
| [`04_segment_qc.py`](04_segment_qc.py) | маска кости на всех уникальных кадрах и листы для проверки глазами | `reports/segmentation_qc.csv` |
| [`05_router_check.py`](05_router_check.py) | сверка роутера области с просмотренным эталоном на 252 кадрах | вывод в консоль |

## Правила позвоночника и бедра

| скрипт | что делает | результат |
|---|---|---|
| [`06_spine_axis.py`](06_spine_axis.py) | разведка правила наклона оси: способы оценки угла | `reports/spine_axis.csv` |
| [`07_spine_rules.py`](07_spine_rules.py) | метрики правил позвоночника с бутстреп-ДИ | `reports/spine_rules.csv` |
| [`08_train_artifact.py`](08_train_artifact.py) | обучение и проверка детектора артефактов (top-hat + логистическая регрессия) | `models/artifact_tophat_v1.npz`, `reports/artifact_*` |
| [`09_femur_rules.py`](09_femur_rules.py) | правило охвата области интереса бедра | `reports/femur_rules.csv` |

## Ротация бедра

| скрипт | что делает | результат |
|---|---|---|
| [`10_femur_rotation.py`](10_femur_rotation.py) | гипотезы ротации по маске и обоснование отказа от них | `reports/femur_rotation*.csv` |
| [`11_femur_landmarks.py`](11_femur_landmarks.py) | выбор 80 снимков бедра и черновые точки ориентиров для проверки | `annotations/femur_points_v1.csv` |
| [`11_rotation_landmarks.py`](11_rotation_landmarks.py) | гипотезы ротации по ручным и автоматическим якорям | `reports/rotation_day6/` |
| [`12_annotate_femur.py`](12_annotate_femur.py) | офлайн-страница для ручной разметки четырёх точек бедра врачом | `annotations/annotator.html` (в публичную копию не входит), `femur_points_human.csv` |
| [`12_refine_rotation.py`](12_refine_rotation.py) | варианты прослеживания медиального края, вложенный выбор | `reports/rotation_refinement/` |
| [`13_femur_rotation_landmarks.py`](13_femur_rotation_landmarks.py) | разделяет ли выступ малого вертела ротацию — по точкам, поставленным человеком | вывод в консоль |
| [`14_rotation_human_profile.py`](14_rotation_human_profile.py) | четыре признака профиля края, два хвоста, вложенная проверка | `reports/rotation_human_profile/` |
| [`17_fit_rotation_rule.py`](17_fit_rotation_rule.py) | заморозка правила ротации для конвейера | `models/rotation_profile_v1.json` |
| [`20_trochanter_localizer.py`](20_trochanter_localizer.py) | OOF-локализатор малого вертела по 58 точкам врача | `reports/rotation_localizer/` |
| [`21_rotation_v2_nested.py`](21_rotation_v2_nested.py) | двойной кросс-фиттинг локализатора и вложенная проверка ротации v2 | `reports/rotation_localizer/` |
| [`22_fit_rotation_v2.py`](22_fit_rotation_v2.py) | заморозка v2 — только если пройдены пороги go (не пройдены) | — |

## Метрики, сравнение вариантов, ТЗ

| скрипт | что делает | результат |
|---|---|---|
| [`15_checkpoint_metrics.py`](15_checkpoint_metrics.py) | все правила на пяти фолдах: разброс, честные пороги, агрегаты | `reports/checkpoint_day7/`, `checkpoint_day9/` |
| [`16_multilabel_head.py`](16_multilabel_head.py) | обучаемая голова поверх геометрии против правил | `reports/multilabel_head/` |
| [`19_rule_errors.py`](19_rule_errors.py) | разбор ошибок геометрических правил и ротации | `reports/errors/` |
| [`23_tz_criteria.py`](23_tz_criteria.py) | три критерия ТЗ, которые не стали правилами, — проверка на данных | `reports/tz_criteria/` |
| [`24_model_comparison.py`](24_model_comparison.py) | все варианты модели на одной мере — слайд ТЗ 4 | `reports/model_comparison/` |
| [`26_quality_score.py`](26_quality_score.py) | подгонка балла `quality_score` и честный ROC-AUC | `models/quality_score_v1.json`, `reports/quality_score/` |

## Проверки перед сдачей

| скрипт | что делает |
|---|---|
| [`18_stress_test.py`](18_stress_test.py) | стресс-корпус «всего плохого» через `batch.run`: каталогом и zip, ни одного исключения → `reports/stress/` |
| [`25_freeze_compare.py`](25_freeze_compare.py) | сверка результатов `run.sh` с эталоном: колонки ТЗ 2.5, решения, overlay |
| [`freeze_check.sh`](freeze_check.sh) | «как на чистой машине»: клон → pytest → сборка linux/amd64 → прогон в контейнере без сети → сверка |
| [`refresh_reports.sh`](refresh_reports.sh) | пересчёт отчётов, зависящих от сквозного прогона |
| [`verify_day6_pipeline.py`](verify_day6_pipeline.py) | совпадение признаков конвейера с таблицами, на которых подогнаны пороги |

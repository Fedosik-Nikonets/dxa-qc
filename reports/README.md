# reports/ — отчёты спринта и таблицы, из которых взяты все цифры

[← README](../README.md) · [Карта документации](../docs/README.md)

Каждый день спринта закрыт отчётом `dayN_*.md`, каждая цифра в README и
документации берётся из таблицы или `summary.json` в этой папке и
пересчитывается скриптом из [`scripts/`](../scripts/README.md). Для
пересчёта нужен набор организатора в `Data/` — в публичную копию он не входит.

## Отчёты по дням спринта

| дата | отчёт | о чём |
|---|---|---|
| 15.09 | [day1_findings.md](day1_findings.md) | разведка данных: состав набора, дубликаты, метки, масштаб |
| 16.09 | [day2_findings.md](day2_findings.md) | фундамент: загрузчик, сегментация, фиксация выборки |
| 17.09 | [day3_findings.md](day3_findings.md) | сквозной скелет: каталог на входе, таблица ТЗ 2.5 на выходе |
| 17–18.09 | [day4_artifact.md](day4_artifact.md) | детектор артефактов позвоночника: top-hat и логистический порог |
| 18.09 | [day4_findings.md](day4_findings.md) | геометрия позвоночника в конвейере: укладка и наклон оси |
| 17–18.09 | [day17_18_partner.md](day17_18_partner.md) | задания напарника за 17–18 сентября |
| 19.09 | [day5_findings.md](day5_findings.md) | геометрия бедра: охват области интереса |
| 19.09 | [day5_rotation.md](day5_rotation.md) | ротация бедра: почему правила по маске нет |
| 19–20.09 | [day6_rotation_draft_landmarks.md](day6_rotation_draft_landmarks.md) | исторический отчёт: черновые точки до человеческой разметки |
| 19–20.09 | [day6_rotation_human_profile.md](day6_rotation_human_profile.md) | контур, два порога и проверка по точкам врача |
| 20.09 | [day6_rotation_landmarks.md](day6_rotation_landmarks.md) | что показала ручная разметка четырёх точек бедра |
| 20.09 | [day6_rotation_refinement.md](day6_rotation_refinement.md) | уточнение поиска малого вертела |
| 21.09 | [day7_checkpoint.md](day7_checkpoint.md) | контрольная точка: все правила на пяти фолдах, честные пороги |
| 22.09 | [day8_multilabel.md](day8_multilabel.md) | обучаемая голова поверх геометрии — не принята |
| 23.09 | [day9_rotation_overlay.md](day9_rotation_overlay.md) | ротация в конвейере и overlay-серия |
| 23.09 | [day11_errors_artifact.md](day11_errors_artifact.md) | разбор ошибок детектора артефактов: рёбра |
| 24.09 | [day12_errors_rules.md](day12_errors_rules.md) | разбор ошибок остальных правил |
| 24–25.09 | [day10_service_container.md](day10_service_container.md) | сервис, контейнер и стресс-тест |
| 24–25.09 | [day12_rotation_localizer.md](day12_rotation_localizer.md) | точный локализатор вертела и ротация v2 — no-go |
| 24.09 | [model_comparison/](model_comparison/README.md) | слайд ТЗ 4 «Сравнение вариантов модели» |
| 26.09 | [f1_sprint/](f1_sprint/README.md) | спринт F1: восемь вариантов с критерием, записанным до запуска — no-go |

## Передача дел и допущения

| файл | что внутри |
|---|---|
| [handoff_day3.md](handoff_day3.md), [handoff_day4.md](handoff_day4.md), [handoff_day5.md](handoff_day5.md) | что нужно знать напарнику после дней 3–5 |
| [handoff_day6_partner.md](handoff_day6_partner.md) | итог задач напарника за 19–20 сентября |
| [handoff_day12.md](handoff_day12.md), [handoff_day12_partner.md](handoff_day12_partner.md) | план до заморозки 27 сентября и вердикт по ротации v2 |
| [questions_to_organizer.md](questions_to_organizer.md) | вопросы организатору и принятые без ответа допущения |

## Таблицы верхнего уровня

| файл | что внутри | скрипт |
|---|---|---|
| `inventory_train.csv`, `inventory_test.csv` | метаданные, размеры, масштаб, дубликаты каждого файла | `01` |
| `dataset_images.csv` | датасет уровня «изображение» с областью и метками | `02` |
| `split.csv` | фиксированное разбиение на пять фолдов по исследованиям | `03` |
| `segmentation_qc.csv` | проверка маски кости на всех уникальных кадрах | `04` |
| `spine_axis.csv`, `spine_rules.csv` | признаки и решения правил позвоночника | `06`, `07` |
| `out_of_scope_spine.csv` | находки вне перечня ТЗ 2.3 (сколиоз и др.) | `07` |
| `artifact_features.csv`, `artifact_oof.csv`, `artifact_cv.json` | детектор артефактов: признак, out-of-fold, проверка | `08` |
| `femur_rules.csv` | геометрия бедра и правило охвата | `09` |
| `femur_rotation.csv`, `femur_rotation_cv.csv` | гипотезы ротации по маске | `10` |
| `batch_diagnostics_train.csv` | сквозной прогон по обучающему набору: все измерения | `refresh_reports.sh` |
| `result_train.xlsx` | таблица ТЗ 2.5 на обучающем наборе | `refresh_reports.sh` |

## Папки с результатами экспериментов

| папка | что внутри | скрипт |
|---|---|---|
| `checkpoint_day7/`, `checkpoint_day9/` | метрики контрольных точек: по фолдам, вложенные пороги | `15` |
| `rotation_day6/` | ротация по ручным точкам и автоматическим якорям | `11_rotation_landmarks` |
| `rotation_refinement/` | варианты прослеживания медиального края | `12_refine_rotation` |
| `rotation_human_profile/` | четыре признака профиля, вложенная проверка — основа правила ротации | `14` |
| `rotation_rule/` | заморозка правила ротации: выбор признака и решения | `17` |
| `multilabel_head/` | обучаемая голова: OOF, дельты против правил | `16` |
| `stress/` | стресс-корпус из 77 файлов: таблица и сводка | `18` |
| `errors/` | разбор ошибок правил и детектора артефактов | `19` |
| `rotation_localizer/` | локализатор малого вертела v2 и вложенная проверка | `20`–`22` |
| `tz_criteria/` | три критерия ТЗ, которые не стали правилами | `23` |
| `model_comparison/` | все варианты на одной мере, слайд ТЗ 4 | `24` |
| `quality_score/` | балл `quality_score`: крутизна и ROC-AUC вложенно | `26` |
| `f1_sprint/` | предрегистрация и итог спринта 26 сентября | — |

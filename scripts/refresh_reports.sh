#!/usr/bin/env bash
# Пересчёт отчётов, которые зависят от сквозного прогона, в закреплённом
# окружении разработки (.venv по requirements.txt). Запускать перед
# заморозкой кода, затем посмотреть `git diff --stat` и закоммитить.
#
#   ./scripts/refresh_reports.sh
#   PYTHON=/путь/к/python ./scripts/refresh_reports.sh
#
# 1. Сквозной прогон обучающего набора во временный каталог и сверка решений
#    со старой reports/batch_diagnostics_train.csv (scripts/25_freeze_compare.py).
#    Если хоть одно решение разошлось — остановка: эталон молча не подменяется.
# 2. Новый прогон становится эталоном: reports/result_train.xlsx и
#    reports/batch_diagnostics_train.csv (новые колонки, время обработки).
# 3. Метрики (15), балл quality_score (26), сверка конвейера и ридеров (verify_day6_pipeline),
#    стресс-тест (18). Слайд сравнения моделей (24) от сквозного прогона не
#    зависит и здесь не пересчитывается.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
R="Data/НД_для_обучения/Исследования"
TMP="$(mktemp -d /tmp/dxa-refresh.XXXXXX)"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

echo ">> 1. сквозной прогон → $TMP"
"$PY" -m dxa_qc.batch --input "$R" --output "$TMP/result.xlsx" --diagnostics "$TMP/diagnostics.csv" --quiet
"$PY" scripts/25_freeze_compare.py --out "$TMP" --expect-rows 499 --no-overlays \
  || { echo "!! решения разошлись с эталоном — отчёты не обновлены, разбор: $TMP" >&2; exit 1; }

echo ">> 2. новый эталон"
cp "$TMP/result.xlsx" reports/result_train.xlsx
cp "$TMP/diagnostics.csv" reports/batch_diagnostics_train.csv

echo ">> 3. метрики, сверка конвейера, стресс-тест"
"$PY" scripts/15_checkpoint_metrics.py > "$TMP/15.log"
"$PY" - <<'PYEOF'
import json
s = json.load(open("reports/checkpoint_day9/summary.json", encoding="utf-8"))
chk = s["pipeline_check"]
print("   15: сверено изображений", chk["matched_images"], "| расхождений геометрии", chk["geometry_mismatches"])
if chk["geometry_mismatches"]:
    raise SystemExit("!! решения конвейера разошлись с таблицами правил")
PYEOF
"$PY" scripts/26_quality_score.py > "$TMP/26.log"
echo "   26: балл quality_score — models/quality_score_v1.json, reports/quality_score/"
"$PY" scripts/verify_day6_pipeline.py > "$TMP/verify.log"
"$PY" - <<'PYEOF'
import json
v = json.load(open("reports/rotation_human_profile/pipeline_verification.json", encoding="utf-8"))
print("   verify_day6_pipeline: passed =", v.get("passed"))
if not v.get("passed"):
    raise SystemExit("!! verify_day6_pipeline не прошёл")
PYEOF
"$PY" scripts/18_stress_test.py --out reports/stress > "$TMP/18.log"
"$PY" - <<'PYEOF'
import json
s = json.load(open("reports/stress/summary.json", encoding="utf-8"))
print("   18: стресс-тест passed =", s.get("passed"))
if not s.get("passed"):
    raise SystemExit("!! стресс-тест не прошёл")
PYEOF

echo ">> изменения в reports/ (время обработки меняется при каждом прогоне):"
git status --short reports
echo ">> журналы: $TMP"

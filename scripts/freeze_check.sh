#!/usr/bin/env bash
# Проверка перед заморозкой кода (27 сентября): «как на чистой машине».
# macOS и Linux, нужны Docker и окружение разработки (.venv по requirements.txt).
#
#   ./scripts/freeze_check.sh
#       клон HEAD → pytest → сборка под linux/amd64 с файлом для сдачи →
#       прогоны через контейнер без сети → сверка с эталоном
#   PLATFORM=linux/arm64 ./scripts/freeze_check.sh
#       быстрее на Mac M2 (без эмуляции), но файл образа для сдачи не пишется
#   SKIP_TESTS=1 / SKIP_BUILD=1   — пропустить шаг (сборка — если образ уже есть)
#   PYTHON=/путь/к/python         — по умолчанию .venv/bin/python репозитория
#   NO_FETCH=1                    — не обновлять origin (без сети)
#
# Шаги:
#   1. git: рабочее дерево чистое, HEAD совпадает с origin/main
#   2. клон HEAD во временный каталог — только закоммиченные файлы
#   3. pytest в клоне
#   4. ./build.sh в клоне; для linux/amd64 — ещё и dxa-qc_<версия>.tar.gz
#      в корень репозитория (в git не попадает)
#   5. run.sh на Data/НД_для_обучения.zip: 500 строк, решения совпадают с
#      reports/batch_diagnostics_train.csv (scripts/25_freeze_compare.py)
#   6. run.sh на Data/Для теста.zip
#   7. стресс-корпус (scripts/18_stress_test.py --make-only) через контейнер
# Итог — таблица шагов и код возврата: 0, если всё OK.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
PLATFORM="${PLATFORM:-linux/amd64}"
WORK="${FREEZE_DIR:-$(mktemp -d /tmp/dxa-freeze.XXXXXX)}"
CLONE="$WORK/clone"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$REPO/src/dxa_qc/__init__.py")"
IMAGE="dxa-qc:$VERSION"
SUMMARY=""
FAILED=0

step() { printf '\n==== %s\n' "$1"; }
ok()   { SUMMARY="$SUMMARY
  OK    $1"; }
bad()  { SUMMARY="$SUMMARY
  FAIL  $1"; FAILED=1; }
warn() { SUMMARY="$SUMMARY
  WARN  $1"; }

command -v docker >/dev/null || { echo "нужен Docker" >&2; exit 1; }
if ! docker info >/dev/null 2>&1; then
  echo "Docker не запущен: запустите Docker Desktop (open -a Docker)," >&2
  echo "дождитесь, пока docker info ответит, и повторите." >&2
  exit 1
fi
echo ">> каталог проверки: $WORK"
echo ">> python: $PY"
echo ">> платформа образа: $PLATFORM, образ: $IMAGE"

step "1. git"
if [ -n "$(git -C "$REPO" status --porcelain)" ]; then
  warn "есть незакоммиченные изменения — в клон и в образ они не попадут"
else
  ok "рабочее дерево чистое"
fi
[ "${NO_FETCH:-0}" = 1 ] || git -C "$REPO" fetch -q origin 2>/dev/null || true
head_sha="$(git -C "$REPO" rev-parse HEAD)"
origin_sha="$(git -C "$REPO" rev-parse origin/main 2>/dev/null || echo нет)"
if [ "$head_sha" = "$origin_sha" ]; then
  ok "HEAD ${head_sha:0:7} = origin/main"
else
  warn "HEAD ${head_sha:0:7} не совпадает с origin/main ${origin_sha:0:7} — не забыть git push"
fi

step "2. клон HEAD"
rm -rf "$CLONE"
git clone -q --no-hardlinks "$REPO" "$CLONE"
ok "клон $(git -C "$CLONE" rev-parse --short HEAD) в $CLONE"

step "3. pytest"
if [ "${SKIP_TESTS:-0}" = 1 ]; then
  warn "pytest пропущен (SKIP_TESTS=1)"
elif (cd "$CLONE" && "$PY" -m pytest -q -p no:cacheprovider); then
  ok "pytest"
else
  bad "pytest"
fi

step "4. сборка образа"
if [ "${SKIP_BUILD:-0}" = 1 ]; then
  warn "сборка пропущена (SKIP_BUILD=1), используется имеющийся $IMAGE"
else
  save=""
  [ "$PLATFORM" = "linux/amd64" ] && save="--save"
  if (cd "$CLONE" && PLATFORM="$PLATFORM" ./build.sh $save); then
    ok "сборка $PLATFORM и запуск health без сети"
    if [ -n "$save" ]; then
      mv -f "$CLONE"/dxa-qc_*.tar.gz "$REPO/"
      f="$(ls "$REPO"/dxa-qc_"$VERSION".tar.gz)"
      sha="$(shasum -a 256 "$f" 2>/dev/null || sha256sum "$f")"
      ok "файл для сдачи: $(basename "$f"), $(du -h "$f" | cut -f1), sha256 ${sha%% *}"
    fi
  else
    bad "сборка образа"
  fi
fi

run_case() {  # $1 — вход, $2 — каталог результатов
  rm -rf "$2"
  IMAGE="$IMAGE" "$CLONE/run.sh" "$1" "$2"
}

step "5. обучающий набор через контейнер"
if run_case "$REPO/Data/НД_для_обучения.zip" "$WORK/out_train" \
   && "$PY" "$CLONE/scripts/25_freeze_compare.py" --out "$WORK/out_train" --expect-rows 500 \
        --reference "$CLONE/reports/batch_diagnostics_train.csv"; then
  ok "НД_для_обучения.zip: 500 строк, решения = эталон"
else
  bad "НД_для_обучения.zip (см. вывод шага 5)"
fi

step "6. «Для теста.zip» через контейнер"
if run_case "$REPO/Data/Для теста.zip" "$WORK/out_test" \
   && "$PY" "$CLONE/scripts/25_freeze_compare.py" --out "$WORK/out_test" --expect-rows 3 --no-reference; then
  ok "Для теста.zip: 3 строки, overlay на каждый файл"
else
  bad "Для теста.zip (см. вывод шага 6)"
fi

step "7. стресс-корпус через контейнер"
rm -rf "$WORK/stress"
if (cd "$CLONE" && "$PY" scripts/18_stress_test.py --make-only "$WORK/stress" >/dev/null); then
  n="$(find "$WORK/stress/corpus" -type f ! -name '.*' | wc -l | tr -d ' ')"
  if run_case "$WORK/stress/corpus" "$WORK/out_stress" \
     && "$PY" "$CLONE/scripts/25_freeze_compare.py" --out "$WORK/out_stress" --expect-rows "$n" --no-reference; then
    ok "стресс-корпус: $n файлов, ни одного вылета"
  else
    bad "стресс-корпус (см. вывод шага 7)"
  fi
else
  bad "стресс-корпус не построен"
fi

printf '\n==== ИТОГ (%s, %s)%s\n' "$IMAGE" "$PLATFORM" "$SUMMARY"
echo "результаты прогонов: $WORK"
exit "$FAILED"

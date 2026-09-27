#!/usr/bin/env bash
# Запуск контейнера (ТЗ 3.2). Linux и macOS, нужен только Docker и собранный
# образ (./build.sh).
#
# Пакетная обработка — контейнер без сети, вход только на чтение:
#   ./run.sh <каталог с исследованиями или zip-архив> <каталог для результатов>
# В каталоге результатов появятся:
#   result.xlsx       — таблица ТЗ 2.5
#   overlays.zip      — дополнительная серия (ТЗ 2.6, 2.7)
#   diagnostics.csv   — служебные колонки для разбора ошибок
#
# Сервис (ТЗ 3.2, 2.6): веб-интерфейс на http://localhost:<порт>/, API — /docs:
#   ./run.sh --serve [порт]
set -euo pipefail

IMAGE="${IMAGE:-dxa-qc:latest}"

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 2; }
abspath() { (cd "$(dirname "$1")" && printf '%s/%s\n' "$(pwd)" "$(basename "$1")"); }

command -v docker >/dev/null || { echo "нужен Docker" >&2; exit 1; }

if [ "${1:-}" = "--serve" ]; then
  port="${2:-8000}"
  echo ">> веб-интерфейс: http://localhost:$port/   API: http://localhost:$port/docs"
  # Сеть здесь нужна только для публикации порта; сам сервис никуда не ходит.
  exec docker run --rm -p "127.0.0.1:$port:8000" "$IMAGE" serve
fi

[ "$#" -eq 2 ] || usage
in="$(abspath "$1")"
out="$(abspath "$2")"
[ -e "$in" ] || { echo "нет входа: $in" >&2; exit 2; }
mkdir -p "$out"

if [ -f "$in" ]; then
  mount=(-v "$in:/data/input.zip:ro"); target=/data/input.zip
else
  mount=(-v "$in:/data/input:ro"); target=/data/input
fi

# --user: результаты принадлежат запустившему, а не uid из образа.
docker run --rm --network none --user "$(id -u):$(id -g)" \
  "${mount[@]}" -v "$out:/data/output" "$IMAGE" \
  batch --input "$target" \
        --output /data/output/result.xlsx \
        --diagnostics /data/output/diagnostics.csv \
        --overlays /data/output/overlays.zip

echo ">> результаты: $out"

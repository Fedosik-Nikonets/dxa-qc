#!/bin/sh
# Точка входа образа.
#   dxa-qc serve                     — HTTP API на $PORT (по умолчанию 8000)
#   dxa-qc batch --input ... --output ... [--overlays ...]   — пакетный прогон
#   dxa-qc health                    — версия и загруженные модели, JSON
set -eu

cmd="${1:-serve}"
[ "$#" -gt 0 ] && shift

case "$cmd" in
  serve)
    exec python -m dxa_qc.service --host 0.0.0.0 --port "${PORT:-8000}" "$@" ;;
  batch)
    exec python -m dxa_qc.batch "$@" ;;
  health)
    exec python -c "import json, dxa_qc.service as s; print(json.dumps(s.health_info(), ensure_ascii=False, indent=2))" ;;
  *)
    exec "$cmd" "$@" ;;
esac

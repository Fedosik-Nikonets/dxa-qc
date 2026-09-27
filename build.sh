#!/usr/bin/env bash
# Сборка образа (ТЗ 3.2). Linux и macOS, нужен только Docker.
#
#   ./build.sh                          — образ под платформу этой машины
#   PLATFORM=linux/amd64 ./build.sh     — под x86_64 (машина проверки организатора)
#   ./build.sh --save                   — ещё и файл dxa-qc_<версия>.tar.gz для передачи
#
# После сборки образ проверяется: запуск без сети, обе модели на месте.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-dxa-qc}"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' src/dxa_qc/__init__.py)"
PLATFORM="${PLATFORM:-}"
SAVE=0
for a in "$@"; do
  case "$a" in
    --save) SAVE=1 ;;
    *) echo "неизвестный ключ: $a" >&2; exit 2 ;;
  esac
done

command -v docker >/dev/null || { echo "нужен Docker: https://docs.docker.com/get-docker/" >&2; exit 1; }

args=(build -t "$IMAGE:$VERSION" -t "$IMAGE:latest")
[ -n "$PLATFORM" ] && args+=(--platform "$PLATFORM")
echo ">> docker ${args[*]} ."
docker "${args[@]}" .

echo ">> проверка: запуск без сети"
docker run --rm --network none ${PLATFORM:+--platform "$PLATFORM"} "$IMAGE:$VERSION" health

echo ">> образ: $IMAGE:$VERSION ($(docker image inspect --format '{{.Architecture}}, {{.Size}} байт' "$IMAGE:$VERSION"))"
echo ">> базовый образ: $(sed -n 's/^ARG PYTHON_IMAGE=//p' Dockerfile)"

if [ "$SAVE" = 1 ]; then
  out="${IMAGE}_${VERSION}.tar.gz"
  docker save "$IMAGE:$VERSION" | gzip > "$out"
  echo ">> сохранено: $out  (загрузка на другой машине: docker load < $out)"
fi

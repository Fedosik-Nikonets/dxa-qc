# Образ сервиса оценки качества DXA. CPU-only, без обращений в сеть на
# работе (ТЗ 3.2): правила и веса лежат в образе, запуск — с --network none.
#
# Сборка и запуск — ./build.sh и ./run.sh (см. README, раздел «Контейнер»).
#
# Базовый образ закреплён версией и digest (ТЗ 3.2): пересборка через месяц
# возьмёт байт-в-байт тот же слой. Digest — мультиплатформенный индекс, то
# есть один и тот же для arm64 (Mac) и amd64 (машина проверки).
ARG PYTHON_IMAGE=python:3.12.7-slim-bookworm@sha256:60d9996b6a8a3689d36db740b49f4327be3be09a21122bd02fb8895abb38b50d
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    PORT=8000

WORKDIR /app

# Зависимости отдельным слоем: правка кода не пересобирает их.
# Только колёса: если под платформу нет готового колеса, сборка падает сразу,
# а не компилирует что-то молча. pip check — набор полон и согласован.
COPY docker/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install --no-deps --only-binary=:all: -r /tmp/requirements-runtime.txt \
 && pip check \
 && rm /tmp/requirements-runtime.txt

COPY src /app/src
COPY models /app/models
COPY docker/entrypoint.sh /usr/local/bin/dxa-qc

# Права выставляются явно: COPY переносит их из рабочей копии, а на Mac файлы
# проекта бывают 600 — пользователь dxa их не прочитал бы, и пакет
# «не нашёлся» бы только при запуске (так и было при первой сборке).
RUN chmod 755 /usr/local/bin/dxa-qc \
 && chmod -R a+rX /app \
 && useradd --create-home --uid 1000 dxa

USER dxa

# Дымовая проверка — от того же пользователя, от которого сервис работает:
# модули импортируются, обе модели находятся, приложение FastAPI создаётся,
# тяжёлых библиотек разработки в образе нет.
RUN python -c "import sys, dxa_qc.batch, dxa_qc.service as s; h = s.health_info(); \
assert all(m['loaded'] for m in h['models'].values()), h; \
assert not {'torch', 'pandas', 'sklearn'} & set(sys.modules), 'лишние модули'; \
import fastapi, uvicorn, multipart; s.create_app(); \
print('ok', h['version'])"

EXPOSE 8000
ENTRYPOINT ["dxa-qc"]
CMD ["serve"]

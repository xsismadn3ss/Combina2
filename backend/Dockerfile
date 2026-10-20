# syntax=docker/dockerfile:1
# Backend de produccion: Python 3.12 (TensorFlow 2.21 estable; no hay wheel
# cp314). Incluye TF para que las redes E/G/C se carguen de verdad en runtime.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PORT=8000
WORKDIR /app

# libgomp1: runtime OpenMP que TensorFlow necesita en Debian slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# `app/` antes de instalar: el discovery de setuptools es `app*`.
COPY pyproject.toml README.md ./
COPY app ./app
RUN uv pip install --system --no-cache ".[ml]"

# Solo artefactos de inferencia (nunca training.db).
COPY models ./models

EXPOSE 8000
# Railway inyecta PORT; el shell expande el default fuera de Railway.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

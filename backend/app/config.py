"""Configuración por entorno (sin secretos en código).

Carga ``.env`` de la raíz del repo sin sobreescribir variables ya definidas
(compose, Railway, CI o la propia shell). Las funciones leen el entorno en
cada llamada para que los tests puedan aislar URLs con ``monkeypatch``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"

# `.env` local (gitignored). En Docker/producción el archivo no existe y las
# variables vienen inyectadas; load_dotenv no falla en ese caso. override=False
# respeta SIEMPRE lo que ya esté definido en el proceso.
load_dotenv(REPO_ROOT / ".env", override=False)

DEFAULT_IDENTIFY_MIN_CONFIDENCE = 0.70

# Solo para DX local (compose). En producción se define DATABASE_URL.
DEFAULT_RUNTIME_DATABASE_URL = (
    "postgresql+psycopg://combina2:combina2@localhost:5432/combina2"
)
DEFAULT_TRAINING_DATABASE_URL = (
    "postgresql+psycopg://combina2:combina2@localhost:5432/combina2_training"
)
DEFAULT_DB_CONNECT_TIMEOUT = 5


@dataclass(frozen=True)
class APICOnfig:
    prefix: str = "/api/v1"


def identify_min_confidence() -> float:
    """Umbral mínimo de confianza calibrada para ``/identify``.

    Configurable por ``IDENTIFY_MIN_CONFIDENCE`` (default 0.70). Valores fuera
    de ``[0, 1]`` o no numéricos se ignoran (se usa el default). El umbral es
    de producto, no de evaluación: sirve para pedir otra foto cuando la
    clasificación es dudosa. Justificación risk-coverage en ``docs/api.md`` y
    ``backend/eval/thresholds.json``.
    """
    raw = os.environ.get("IDENTIFY_MIN_CONFIDENCE")
    if raw is None:
        return DEFAULT_IDENTIFY_MIN_CONFIDENCE
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_IDENTIFY_MIN_CONFIDENCE
    if not 0.0 <= value <= 1.0:
        return DEFAULT_IDENTIFY_MIN_CONFIDENCE
    return value


def normalize_database_url(url: str) -> str:
    """Normaliza la URL de Postgres al driver psycopg 3.

    Railway entrega ``postgresql://...``, Heroku usaba ``postgres://...`` y la
    configuración local ya trae ``postgresql+psycopg://``. Solo se reescribe el
    esquema (se preservan usuario, contraseña, host, puerto, base y query).
    Cualquier URL que no sea Postgres (o que ya tenga driver) se devuelve igual.
    """
    value = (url or "").strip()
    if "://" not in value:
        return value
    scheme, rest = value.split("://", 1)
    if scheme.lower() in {"postgres", "postgresql"}:
        return f"postgresql+psycopg://{rest}"
    return value


def runtime_database_url() -> str:
    """URL del runtime (feedback). ``DATABASE_URL`` o el default local."""
    value = os.environ.get("DATABASE_URL", "").strip()
    return normalize_database_url(value or DEFAULT_RUNTIME_DATABASE_URL)


def training_database_url() -> str:
    """URL de la DB de entrenamiento (``TRAINING_DATABASE_URL``).

    No se define en producción: los datos de entrenamiento se quedan en local.
    """
    value = os.environ.get("TRAINING_DATABASE_URL", "").strip()
    return normalize_database_url(value or DEFAULT_TRAINING_DATABASE_URL)


def feedback_export_token() -> str | None:
    """Token Bearer para ``GET /feedback/export.csv``; ``None`` = endpoint oculto."""
    value = os.environ.get("FEEDBACK_EXPORT_TOKEN", "").strip()
    return value or None


def db_connect_timeout() -> int:
    """Timeout de conexión a Postgres en segundos (default 5)."""
    raw = os.environ.get("DB_CONNECT_TIMEOUT")
    if raw is None:
        return DEFAULT_DB_CONNECT_TIMEOUT
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_DB_CONNECT_TIMEOUT
    return value if value > 0 else DEFAULT_DB_CONNECT_TIMEOUT

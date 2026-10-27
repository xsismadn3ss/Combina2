"""Engines SQLAlchemy (sync, psycopg 3) para runtime y entrenamiento.

La API es síncrona y FastAPI la ejecuta en el threadpool: SQLAlchemy sync es
suficiente y evita el coste/complejidad de async. Cada engine se cachea por
URL (la URL cambia entre entorno local, tests y producción).
"""

from __future__ import annotations

import logging
import os
import time

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.pool import NullPool

from app.config import db_connect_timeout, runtime_database_url, training_database_url

logger = logging.getLogger(__name__)

RUNTIME_APPLICATION_NAME = "combina2-runtime"
TRAINING_APPLICATION_NAME = "combina2-training"
READY_APPLICATION_NAME = "combina2-ready"

READY_CACHE_TTL = 10.0
READY_CONNECT_TIMEOUT = 2

_engines: dict[tuple[str, str, int], Engine] = {}
_ready_cache: dict[str, tuple[float, bool]] = {}


def _connect_args(application_name: str, timeout: int) -> dict:
    return {"connect_timeout": timeout, "application_name": application_name}


def get_engine(url: str, application_name: str, timeout: int) -> Engine:
    """Engine cacheado con ping/recycle para sobrevivir a reinicios de Postgres."""
    key = (url, application_name, timeout)
    engine = _engines.get(key)
    if engine is None:
        engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args=_connect_args(application_name, timeout),
            future=True,
        )
        _engines[key] = engine
    return engine


def runtime_engine() -> Engine:
    return get_engine(
        runtime_database_url(), RUNTIME_APPLICATION_NAME, db_connect_timeout()
    )


def training_engine() -> Engine:
    return get_engine(
        training_database_url(), TRAINING_APPLICATION_NAME, db_connect_timeout()
    )


def dispose_engines() -> None:
    """Cierra los pools (tests de persistencia y shutdown)."""
    for engine in list(_engines.values()):
        engine.dispose()
    _engines.clear()
    _ready_cache.clear()


def database_status() -> dict[str, bool]:
    """Estado informativo de la DB para ``/ready`` (nunca altera el 200/503).

    ``configured`` indica si ``DATABASE_URL`` está definida explícitamente;
    ``reachable`` se comprueba con un timeout corto y caché de 10 s para no
    penalizar cada readiness probe.
    """
    url = runtime_database_url()
    configured = bool(os.environ.get("DATABASE_URL", "").strip())
    now = time.monotonic()
    cached = _ready_cache.get(url)
    if cached is not None and now - cached[0] < READY_CACHE_TTL:
        reachable = cached[1]
    else:
        reachable = _probe(url)
        _ready_cache[url] = (now, reachable)
    return {"configured": configured, "reachable": reachable}


def _probe(url: str) -> bool:
    engine = None
    try:
        engine = create_engine(
            url,
            poolclass=NullPool,
            connect_args=_connect_args(
                READY_APPLICATION_NAME,
                min(READY_CONNECT_TIMEOUT, db_connect_timeout()),
            ),
            future=True,
        )
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001 - /ready nunca debe caerse
        logger.warning("database unreachable (%s)", type(exc).__name__)
        return False
    finally:
        if engine is not None:
            engine.dispose()


def sanitized_url(url: str | None = None) -> str:
    """URL sin contraseña, solo para logs."""
    parsed = make_url(url or runtime_database_url())
    return parsed.render_as_string(hide_password=True)

"""Fixtures de test: PostgreSQL real, Alembic y aislamiento por test.

Requiere Postgres local (``docker compose up -d postgres``). No hay SQLite:
todo el esquema se crea con ``alembic upgrade head`` en las dos bases.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

# --- Entorno ANTES de importar app ------------------------------------------
# Nunca se apunta a la DB de desarrollo: si hay DATABASE_URL en el entorno se
# reutilizan host/credenciales pero con nombre de base *_test.
from sqlalchemy.engine import make_url

DEFAULT_RUNTIME = "postgresql+psycopg://combina2:combina2@localhost:5432/combina2_test"
DEFAULT_TRAINING = (
    "postgresql+psycopg://combina2:combina2@localhost:5432/combina2_training_test"
)


def _with_database(url: str, database: str) -> str:
    return make_url(url).set(database=database).render_as_string(hide_password=False)


def _test_url(explicit_env: str, base_env: str, database: str, default: str) -> str:
    explicit = os.environ.get(explicit_env, "").strip()
    if explicit:
        return explicit
    base = os.environ.get(base_env, "").strip()
    if base:
        return _with_database(base, database)
    return default


TEST_DATABASE_URL = _test_url(
    "TEST_DATABASE_URL", "DATABASE_URL", "combina2_test", DEFAULT_RUNTIME
)
TEST_TRAINING_DATABASE_URL = _test_url(
    "TEST_TRAINING_DATABASE_URL",
    "TRAINING_DATABASE_URL",
    "combina2_training_test",
    DEFAULT_TRAINING,
)
TEST_EXPORT_TOKEN = "test-export-token"

os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["TRAINING_DATABASE_URL"] = TEST_TRAINING_DATABASE_URL
os.environ["FEEDBACK_EXPORT_TOKEN"] = TEST_EXPORT_TOKEN
os.environ.setdefault("DB_CONNECT_TIMEOUT", "5")

# --- Imports de la app (ya con el entorno forzado) --------------------------
import pytest  # noqa: E402
import psycopg  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from psycopg import sql  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db.engine import dispose_engines  # noqa: E402
from app.db.session import get_session, runtime_session, training_session  # noqa: E402
from app.main import app  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parents[1]
TABLES = [
    "feedback",
    "palette_colors",
    "palettes",
    "extracted_colors",
    "images",
    "training_runs",
]


def _psycopg_dsn(url: str, database: str | None = None) -> str:
    parsed = make_url(url)
    if database is not None:
        parsed = parsed.set(database=database)
    return parsed.set(drivername="postgresql").render_as_string(hide_password=False)


def _ensure_database(url: str) -> None:
    parsed = make_url(url)
    database = parsed.database or ""
    with psycopg.connect(
        _psycopg_dsn(url, "postgres"), autocommit=True, connect_timeout=5
    ) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (database,)
        ).fetchone()
        if not exists:
            conn.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
            )


def _upgrade(url: str) -> None:
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    previous = os.environ.get("ALEMBIC_DATABASE_URL")
    os.environ["ALEMBIC_DATABASE_URL"] = url
    try:
        command.upgrade(config, "head")
    finally:
        if previous is None:
            os.environ.pop("ALEMBIC_DATABASE_URL", None)
        else:
            os.environ["ALEMBIC_DATABASE_URL"] = previous


def _truncate(url: str) -> None:
    with psycopg.connect(_psycopg_dsn(url), autocommit=True, connect_timeout=5) as conn:
        conn.execute(
            "TRUNCATE TABLE "
            + ", ".join(TABLES)
            + " RESTART IDENTITY CASCADE"
        )


@pytest.fixture(scope="session", autouse=True)
def _database_schema() -> Iterator[None]:
    for url in (TEST_DATABASE_URL, TEST_TRAINING_DATABASE_URL):
        _ensure_database(url)
        _upgrade(url)
    yield
    dispose_engines()


@pytest.fixture(autouse=True)
def _clean_database(_database_schema) -> Iterator[None]:
    """TRUNCATE antes y después de cada test (aislamiento total).

    El teardown deja las bases vacías también al terminar la suite: el job de
    CI ejecuta `eval.run_eval` después de pytest y debe ver el fallback
    sintético, no paletas sueltas del último test.
    """
    _truncate(TEST_DATABASE_URL)
    _truncate(TEST_TRAINING_DATABASE_URL)
    yield
    _truncate(TEST_DATABASE_URL)
    _truncate(TEST_TRAINING_DATABASE_URL)


@pytest.fixture(autouse=True)
def _override_session_dependency(_clean_database) -> Iterator[None]:
    def _session() -> Iterator[Session]:
        with runtime_session() as session:
            yield session

    app.dependency_overrides[get_session] = _session
    yield
    app.dependency_overrides.pop(get_session, None)


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def runtime_db() -> Iterator[Session]:
    """Sesión de runtime que hace commit al salir (visible para la API)."""
    with runtime_session() as session:
        yield session


@pytest.fixture()
def training_db() -> Iterator[Session]:
    with training_session() as session:
        yield session


@pytest.fixture()
def runtime_url() -> str:
    return TEST_DATABASE_URL


@pytest.fixture()
def training_url() -> str:
    return TEST_TRAINING_DATABASE_URL


@pytest.fixture()
def export_token() -> str:
    return TEST_EXPORT_TOKEN


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Cierra pools al terminar la sesión de pytest."""
    dispose_engines()

"""Entorno de Alembic para las dos bases (runtime y entrenamiento).

URL: ``-x url=postgresql+psycopg://...`` > ``ALEMBIC_DATABASE_URL`` >
``DATABASE_URL`` (runtime) > default local. El mismo conjunto de migraciones
crea el esquema completo en ambas bases.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool, text
from sqlmodel import SQLModel

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import (  # noqa: E402
    db_connect_timeout,
    normalize_database_url,
    runtime_database_url,
)
from app.db import models  # noqa: E402,F401  (registra todas las tablas)

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata

ADVISORY_LOCK_KEY = 7_281_920  # constante del proyecto; serializa upgrades


def get_url() -> str:
    x_args = context.get_x_argument(as_dictionary=True)
    return normalize_database_url(
        x_args.get("url")
        or os.environ.get("ALEMBIC_DATABASE_URL")
        or runtime_database_url()
    )


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = get_url()
    if url.startswith("postgresql+pg8000://"):
        connect_args: dict = {"timeout": db_connect_timeout()}
    else:
        connect_args = {
            "connect_timeout": db_connect_timeout(),
            "application_name": "combina2-alembic",
        }
    engine = create_engine(
        url,
        poolclass=pool.NullPool,
        connect_args=connect_args,
        future=True,
    )
    with engine.connect() as connection:
        # Lock de sesión: dos pods arrancando a la vez no migran en paralelo.
        connection.execute(text(f"SELECT pg_advisory_lock({ADVISORY_LOCK_KEY})"))
        connection.commit()
        try:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
            )
            with context.begin_transaction():
                context.run_migrations()
        finally:
            connection.execute(
                text(f"SELECT pg_advisory_unlock({ADVISORY_LOCK_KEY})")
            )
            connection.commit()
            connection.close()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

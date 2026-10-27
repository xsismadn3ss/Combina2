"""Base de SQLAlchemy/SQLModel común a las dos bases.

Un único ``SQLModel.metadata`` describe el esquema completo (runtime y
entrenamiento comparten tablas). Alembic usa esta metadata y la convención de
nombres de abajo para generar constraints deterministas.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import SQLModel

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# `SQLModel.metadata` se crea al importar sqlmodel; se le aplica la convención
# antes de importar los modelos (ver `app/db/models/__init__.py`).
SQLModel.metadata.naming_convention = NAMING_CONVENTION

UTCNOW_SQL = "timezone('utc', now())"


def utcnow() -> datetime:
    """Instante UTC *aware* (los servicios generan ``created_at``)."""
    return datetime.now(timezone.utc)

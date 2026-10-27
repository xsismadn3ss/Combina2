"""Sesiones SQLAlchemy como context managers + dependencia FastAPI."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager

from sqlalchemy.orm import Session

from app.db.engine import runtime_engine, training_engine


@contextmanager
def _session_scope(engine) -> Iterator[Session]:
    """Sesión con commit al salir y rollback ante error."""
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def runtime_session() -> AbstractContextManager[Session]:
    """Sesión de la DB de runtime (feedback de producción)."""
    return _session_scope(runtime_engine())


def training_session() -> AbstractContextManager[Session]:
    """Sesión de la DB de entrenamiento (solo scripts offline/tests)."""
    return _session_scope(training_engine())


def get_session() -> Iterator[Session]:
    """Dependencia FastAPI: una sesión por request (runtime)."""
    with runtime_session() as session:
        yield session

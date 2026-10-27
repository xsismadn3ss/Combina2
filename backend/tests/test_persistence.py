"""Persistencia real: escribir, cerrar el pool, reconectar y leer."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from app.db.engine import dispose_engines
from app.db.models import Feedback
from app.db.session import runtime_session
from app.repositories import feedback as repo


def test_row_survives_engine_dispose():
    created = datetime(2026, 10, 27, 9, 30, tzinfo=timezone.utc)
    with runtime_session() as session:
        row = repo.insert_feedback(
            session,
            palette_json={"type": "triada", "colors": [{"value": "#123456"}]},
            accepted=False,
            score=0.25,
            image_sha256=None,
            created_at=created,
        )
        row_id = int(row.id or 0)

    # Simula reinicio del proceso: se tiran todos los pools.
    dispose_engines()

    with runtime_session() as session:
        stored = session.get(Feedback, row_id)
        assert stored is not None
        assert stored.accepted is False
        assert stored.score == 0.25
        assert stored.palette_type == "triada"
        assert stored.created_at == created
        counted = len(list(session.execute(select(Feedback.id)).scalars()))
        assert counted == 1

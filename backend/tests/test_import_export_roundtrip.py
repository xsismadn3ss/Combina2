"""Roundtrip export CSV (runtime) -> import (entrenamiento)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.db.models import Feedback
from app.repositories import feedback as repo
from scripts import export_feedback_csv, import_feedback_csv


def _seed(session):
    rows = [
        (
            {
                "type": "triada",
                "colors": [
                    {"value": "#FF0000", "role": "primary", "weight": None},
                    {"value": "#00FF00", "role": "secondary", "weight": None},
                ],
                "fallback": False,
            },
            True,
            0.1234567,
            datetime(2026, 10, 1, 8, 0, 0, tzinfo=timezone.utc),
            "a" * 64,
        ),
        (
            {
                "type": "monocromatica",
                "colors": [{"value": "#ABC123", "role": "primary", "weight": None}],
                "fallback": False,
            },
            False,
            None,
            datetime(2026, 10, 2, 9, 30, 15, tzinfo=timezone.utc),
            None,
        ),
    ]
    for palette, accepted, score, created, sha in rows:
        repo.insert_feedback(
            session,
            palette_json=palette,
            accepted=accepted,
            score=score,
            image_sha256=sha,
            created_at=created,
        )
    session.commit()


def test_export_import_roundtrip_is_idempotent(runtime_db, training_db, tmp_path):
    _seed(runtime_db)
    csv_path = Path(tmp_path) / "feedback.csv"

    exported = export_feedback_csv.run(csv_path)
    assert exported["rows"] == 2
    raw = csv_path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")

    dry = import_feedback_csv.run(csv_path, apply=False)
    assert dry["read"] == 2
    assert dry["inserted"] == 2
    assert dry["duplicates"] == 0

    applied = import_feedback_csv.run(csv_path, apply=True)
    assert applied["inserted"] == 2
    assert applied["duplicates"] == 0

    rows = list(
        training_db.execute(select(Feedback).order_by(Feedback.id)).scalars()
    )
    assert len(rows) == 2
    assert rows[0].source == "import"
    assert rows[0].accepted is True
    assert rows[0].score == 0.123457
    assert rows[0].image_sha256 == "a" * 64
    assert rows[0].created_at == datetime(2026, 10, 1, 8, 0, 0, tzinfo=timezone.utc)
    assert rows[1].score is None
    assert rows[1].created_at == datetime(
        2026, 10, 2, 9, 30, 15, tzinfo=timezone.utc
    )

    again = import_feedback_csv.run(csv_path, apply=True)
    assert again["inserted"] == 0
    assert again["duplicates"] == 2
    count = len(
        list(training_db.execute(select(Feedback.id)).scalars())
    )
    assert count == 2

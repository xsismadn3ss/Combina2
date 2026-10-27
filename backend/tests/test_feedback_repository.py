"""Repositorio de feedback sobre PostgreSQL real."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.models import Feedback
from app.repositories import feedback as repo

PALETTE = {
    "type": "triada",
    "colors": [
        {"value": "#FF0000", "role": "primary", "weight": None},
        {"value": "#00FF00", "role": "secondary", "weight": None},
    ],
    "fallback": False,
}


def _insert(session, **overrides):
    values = {
        "palette_json": PALETTE,
        "accepted": True,
        "score": 0.912345678,
        "image_sha256": None,
        "source": "api",
    }
    values.update(overrides)
    return repo.insert_feedback(session, **values)


def test_insert_and_read_feedback(runtime_db):
    row = _insert(runtime_db)
    runtime_db.commit()
    assert row is not None
    assert row.id is not None
    assert row.accepted is True
    assert row.score == pytest.approx(0.912346)  # 6 decimales
    assert row.palette_type == "triada"
    assert row.source == "api"
    assert len(row.fingerprint) == 64
    assert row.palette_json["colors"][0]["value"] == "#FF0000"
    assert row.created_at.tzinfo is not None


def test_duplicate_fingerprint_is_ignored(runtime_db):
    created = datetime(2026, 10, 27, 10, 0, 0, tzinfo=timezone.utc)
    first = _insert(runtime_db, created_at=created)
    runtime_db.commit()
    second = _insert(runtime_db, created_at=created)
    runtime_db.commit()
    assert first is not None
    assert second is None
    assert len(list(runtime_db.execute(select(Feedback)).scalars())) == 1


def test_fingerprint_includes_image_sha(runtime_db):
    sha_a = "a" * 64
    sha_b = "b" * 64
    assert _insert(runtime_db, image_sha256=sha_a) is not None
    runtime_db.commit()
    assert _insert(runtime_db, image_sha256=sha_b) is not None
    runtime_db.commit()
    assert len(list(runtime_db.execute(select(Feedback)).scalars())) == 2


def test_filters_and_ordering(runtime_db):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _insert(runtime_db, accepted=True, created_at=base)
    _insert(runtime_db, accepted=False, created_at=base + timedelta(days=10))
    _insert(runtime_db, accepted=True, created_at=base + timedelta(days=20))
    runtime_db.commit()

    everything = list(repo.iter_feedback(runtime_db))
    assert [row.accepted for row in everything] == [True, False, True]

    accepted_only = list(repo.iter_feedback(runtime_db, accepted=True))
    assert len(accepted_only) == 2
    assert all(row.accepted for row in accepted_only)

    window = list(
        repo.iter_feedback(
            runtime_db,
            start=base + timedelta(days=5),
            end=base + timedelta(days=15),
        )
    )
    assert len(window) == 1
    assert window[0].created_at.day == 11


def test_count_and_purge_before(runtime_db):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _insert(runtime_db, created_at=base)
    _insert(runtime_db, created_at=base + timedelta(days=30))
    runtime_db.commit()

    cutoff = base + timedelta(days=10)
    assert repo.count_before(runtime_db, cutoff) == 1
    runtime_db.commit()
    assert repo.delete_before(runtime_db, cutoff) == 1
    runtime_db.commit()
    assert repo.count_feedback(runtime_db) == 1


def test_invalid_image_sha_is_rejected_by_check(runtime_db):
    with pytest.raises(IntegrityError):
        _insert(runtime_db, image_sha256="not-a-sha")
        runtime_db.commit()
    runtime_db.rollback()

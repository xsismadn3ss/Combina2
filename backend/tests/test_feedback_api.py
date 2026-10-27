"""Contrato HTTP de POST /feedback (200, 422, 503) con Postgres real."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from app.config import APICOnfig
from app.db.models import Feedback
from app.db.session import get_session
from app.main import app

PAYLOAD = {
    "palette": {
        "type": "complementario",
        "colors": [
            {"value": "#FF0000", "role": "primary"},
            {"value": "#00FFFF", "role": "complementary"},
        ],
        "fallback": True,
    },
    "accepted": True,
    "score": 0.9,
    "imageSha256": "a" * 64,
}


def test_post_feedback_persists_and_returns_ok(client, runtime_db):
    response = client.post(f"{APICOnfig.prefix}/feedback", json=PAYLOAD)
    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True}
    row = runtime_db.execute(select(Feedback)).scalars().one()
    assert row.accepted is True
    assert row.image_sha256 == "a" * 64
    assert row.palette_type == "complementario"
    assert row.source == "api"


def test_post_feedback_score_optional(client, runtime_db):
    payload = {**PAYLOAD, "score": None}
    response = client.post(f"{APICOnfig.prefix}/feedback", json=payload)
    assert response.status_code == 200
    row = runtime_db.execute(select(Feedback)).scalars().one()
    assert row.score is None


def test_post_feedback_requires_accepted(client):
    payload = {"palette": PAYLOAD["palette"]}
    response = client.post(f"{APICOnfig.prefix}/feedback", json=payload)
    assert response.status_code == 422


def test_post_feedback_rejects_out_of_range_score(client):
    payload = {**PAYLOAD, "score": 1.5}
    response = client.post(f"{APICOnfig.prefix}/feedback", json=payload)
    assert response.status_code == 422


def test_post_feedback_invalid_image_sha_is_422_and_not_persisted(
    client, runtime_db
):
    payload = {**PAYLOAD, "imageSha256": "zz"}
    response = client.post(f"{APICOnfig.prefix}/feedback", json=payload)
    assert response.status_code == 422
    assert runtime_db.execute(select(Feedback)).scalars().all() == []


def test_post_feedback_normalizes_uppercase_image_sha(client, runtime_db):
    payload = {**PAYLOAD, "imageSha256": "A" * 64}
    response = client.post(f"{APICOnfig.prefix}/feedback", json=payload)
    assert response.status_code == 200, response.text
    row = runtime_db.execute(select(Feedback)).scalars().one()
    assert row.image_sha256 == "a" * 64


def test_post_feedback_returns_503_without_leaking_dsn(client, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import NullPool

    dead_engine = create_engine(
        "postgresql+psycopg://nobody:nobody@127.0.0.1:1/nodb",
        connect_args={"connect_timeout": 1},
        poolclass=NullPool,
    )

    def dead_session():
        session = Session(dead_engine)
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setitem(app.dependency_overrides, get_session, dead_session)
    response = client.post(f"{APICOnfig.prefix}/feedback", json=PAYLOAD)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"
    assert response.json()["detail"] == "Feedback no disponible temporalmente"
    assert "postgres" not in response.text.lower()
    assert "password" not in response.text.lower()
    dead_engine.dispose()


def test_health_is_200_even_when_database_is_down(client, monkeypatch):
    def boom():
        raise OperationalError("SELECT 1", None, Exception("connection refused"))
        yield  # pragma: no cover

    monkeypatch.setitem(app.dependency_overrides, get_session, boom)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

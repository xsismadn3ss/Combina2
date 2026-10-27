"""Contrato de GET /feedback/export.csv (auth, headers, filtros, UTF-8)."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from app.config import APICOnfig
from app.repositories import feedback as repo

PALETTE = {
    "type": "analogo",
    "colors": [
        {"value": "#2A9D8F", "role": "primary", "weight": None},
        {"value": "#1D3557", "role": "secondary", "weight": None},
        {"value": "#E63946", "role": "accent", "weight": None},
    ],
    "fallback": False,
}

EXPECTED_HEADER = [
    "id",
    "created_at",
    "accepted",
    "score",
    "palette_type",
    "palette_json",
    "colors_hex",
    "image_sha256",
]


def _seed(session, *, accepted=True, created_at=None, palette=None, score=0.5, sha=None):
    row = repo.insert_feedback(
        session,
        palette_json=palette or PALETTE,
        accepted=accepted,
        score=score,
        image_sha256=sha,
        created_at=created_at or datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc),
    )
    session.commit()
    return row


def _get(client, token, **params):
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    return client.get(
        f"{APICOnfig.prefix}/feedback/export.csv", headers=headers, params=params
    )


def test_export_404_when_token_not_configured(client, monkeypatch):
    monkeypatch.delenv("FEEDBACK_EXPORT_TOKEN", raising=False)
    response = _get(client, "whatever")
    assert response.status_code == 404


def test_export_401_without_or_with_wrong_token(client):
    assert _get(client, None).status_code == 401
    wrong = _get(client, "nope")
    assert wrong.status_code == 401
    assert wrong.headers["www-authenticate"] == "Bearer"


def test_export_headers_and_utf8(client, export_token, runtime_db):
    _seed(runtime_db, sha="a" * 64)
    response = _get(client, export_token)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "attachment" in response.headers["content-disposition"]
    assert response.content[:3] != b"\xef\xbb\xbf"  # sin BOM

    text = response.content.decode("utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == EXPECTED_HEADER
    assert len(rows) == 2
    data = rows[1]
    assert data[1].endswith("Z")
    assert data[2] == "true"
    assert data[3] == "0.500000"
    assert data[4] == "analogo"
    assert data[6] == "#2A9D8F|#1D3557|#E63946"
    assert data[7] == "a" * 64
    # JSON compacto y sin ensure_ascii forzado.
    assert data[5].startswith('{"colors":') and data[5].endswith("}")


def test_export_no_sha256_column(client, export_token, runtime_db):
    _seed(runtime_db, sha="b" * 64)
    response = _get(client, export_token, include_sha256="false")
    rows = list(csv.reader(io.StringIO(response.content.decode("utf-8"))))
    assert rows[1][7] == ""


def test_export_filters_from_to_and_accepted(client, export_token, runtime_db):
    _seed(
        runtime_db,
        accepted=True,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    _seed(
        runtime_db,
        accepted=False,
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    _seed(
        runtime_db,
        accepted=True,
        created_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
    )
    runtime_db.commit()

    response = _get(
        client,
        export_token,
        **{"from": "2026-05-01", "to": "2026-07-01", "accepted": "false"},
    )
    rows = list(csv.reader(io.StringIO(response.content.decode("utf-8"))))
    assert len(rows) == 2
    assert rows[1][2] == "false"
    assert rows[1][1].startswith("2026-06-01")


def test_export_empty_has_only_header(client, export_token):
    response = _get(client, export_token)
    rows = list(csv.reader(io.StringIO(response.content.decode("utf-8"))))
    assert rows == [EXPECTED_HEADER]


def test_export_mitigates_csv_injection(client, export_token, runtime_db):
    malicious = {
        "type": "=HYPERLINK(\"http://evil\")",
        "colors": [{"value": "+1+1", "role": "@cmd", "weight": None}],
    }
    _seed(runtime_db, palette=malicious)
    response = _get(client, export_token)
    rows = list(csv.reader(io.StringIO(response.content.decode("utf-8"))))
    row = rows[1]
    assert row[4].startswith("'=")
    assert row[6].startswith("'+")
    assert row[5].startswith("{")  # el JSON no es una fórmula: no se altera


def test_export_contains_no_images_or_paths(client, export_token, runtime_db):
    _seed(runtime_db)
    response = _get(client, export_token)
    lower = response.text.lower()
    assert "backend/" not in lower
    assert "c:\\\\" not in lower
    for word in ("image/jpeg", "image/png", ".jpg", ".png"):
        assert word not in lower


def test_count_matches_filters(client, export_token, runtime_db):
    _seed(
        runtime_db,
        accepted=True,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    _seed(
        runtime_db,
        accepted=False,
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    _seed(
        runtime_db,
        accepted=True,
        created_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
    )
    runtime_db.commit()

    url = f"{APICOnfig.prefix}/feedback/count"
    headers = {"Authorization": f"Bearer {export_token}"}

    assert client.get(url, headers=headers).json() == {"count": 3}
    assert client.get(url, headers=headers, params={"accepted": "true"}).json() == {
        "count": 2
    }
    assert client.get(
        url,
        headers=headers,
        params={"from": "2026-05-01", "to": "2026-07-01"},
    ).json() == {"count": 1}
    assert client.get(
        url,
        headers=headers,
        params={"from": "2020-01-01", "to": "2020-12-31"},
    ).json() == {"count": 0}


def test_count_requires_export_token(client):
    url = f"{APICOnfig.prefix}/feedback/count"
    assert client.get(url).status_code == 401
    assert (
        client.get(url, headers={"Authorization": "Bearer nope"}).status_code == 401
    )

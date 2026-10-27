"""Tests unitarios de configuración (normalización de URLs de Postgres)."""

from __future__ import annotations

import pytest

from app.config import (
    normalize_database_url,
    runtime_database_url,
    training_database_url,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "postgres://user:secret@host:5432/db",
            "postgresql+psycopg://user:secret@host:5432/db",
        ),
        (
            "postgresql://user:secret@host:5432/db",
            "postgresql+psycopg://user:secret@host:5432/db",
        ),
        (
            "postgresql+psycopg://user:secret@host:5432/db",
            "postgresql+psycopg://user:secret@host:5432/db",
        ),
        (
            "postgresql://user:p%40ss@host/db?sslmode=require",
            "postgresql+psycopg://user:p%40ss@host/db?sslmode=require",
        ),
        (
            "POSTGRES://user:secret@host/db",
            "postgresql+psycopg://user:secret@host/db",
        ),
        # No es Postgres: se deja intacta.
        ("mysql://user:secret@host/db", "mysql://user:secret@host/db"),
        ("sqlite:///tmp/x.db", "sqlite:///tmp/x.db"),
        # Sin esquema: tampoco se toca.
        ("host:5432/db", "host:5432/db"),
    ],
)
def test_normalize_database_url(raw, expected):
    assert normalize_database_url(raw) == expected


def test_runtime_url_accepts_railway_style(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://u:p@containers.railway.app:5432/railway"
    )
    assert runtime_database_url() == (
        "postgresql+psycopg://u:p@containers.railway.app:5432/railway"
    )


def test_training_url_accepts_heroku_style(monkeypatch):
    monkeypatch.setenv("TRAINING_DATABASE_URL", "postgres://u:p@localhost/training")
    assert training_database_url() == "postgresql+psycopg://u:p@localhost/training"


def test_defaults_are_valid_psycopg_urls(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TRAINING_DATABASE_URL", raising=False)
    assert runtime_database_url().startswith("postgresql+psycopg://")
    assert training_database_url().startswith("postgresql+psycopg://")

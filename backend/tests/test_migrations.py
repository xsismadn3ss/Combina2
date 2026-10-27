"""Alembic: esquema completo, constraints e idempotencia en Postgres real."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import psycopg
from alembic import command
from alembic.config import Config
from psycopg import sql
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

BACKEND_DIR = Path(__file__).resolve().parents[1]

EXPECTED_TABLES = {
    "images",
    "extracted_colors",
    "palettes",
    "palette_colors",
    "training_runs",
    "feedback",
    "alembic_version",
}


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


def _engine(url: str):
    return create_engine(url, future=True)


def _admin_dsn(base_url: str) -> str:
    parsed = make_url(base_url).set(database="postgres", drivername="postgresql")
    return parsed.render_as_string(hide_password=False)


def _create_temp_database(base_url: str) -> tuple[str, str]:
    name = f"combina2_mig_{uuid4().hex[:10]}"
    with psycopg.connect(_admin_dsn(base_url), autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name))
        )
    parsed = make_url(base_url).set(database=name)
    return parsed.render_as_string(hide_password=False), name


def _drop_temp_database(base_url: str, name: str) -> None:
    with psycopg.connect(_admin_dsn(base_url), autocommit=True) as connection:
        connection.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                sql.Identifier(name)
            )
        )


def test_upgrade_creates_full_schema(training_url):
    engine = _engine(training_url)
    try:
        inspector = inspect(engine)
        assert EXPECTED_TABLES <= set(inspector.get_table_names())
        feedback_columns = {
            column["name"]: column
            for column in inspector.get_columns("feedback")
        }
        assert set(feedback_columns) == {
            "id",
            "created_at",
            "accepted",
            "score",
            "palette_type",
            "palette_json",
            "image_sha256",
            "fingerprint",
            "source",
        }
        # timestamptz: la DB guarda zona horaria.
        assert feedback_columns["created_at"]["type"].timezone is True
    finally:
        engine.dispose()


def test_upgrade_is_idempotent(training_url):
    _upgrade(training_url)
    _upgrade(training_url)
    engine = _engine(training_url)
    try:
        with engine.connect() as conn:
            versions = conn.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalars().all()
            assert versions == ["0001_initial"]
    finally:
        engine.dispose()


def test_indexes_and_constraints(training_url):
    engine = _engine(training_url)
    try:
        inspector = inspect(engine)
        indexes = {index["name"] for index in inspector.get_indexes("feedback")}
        assert {
            "ix_feedback_created_at",
            "ix_feedback_accepted",
            "ix_feedback_accepted_created_at",
        } <= indexes
        unique = {
            constraint["name"]
            for constraint in inspector.get_unique_constraints("feedback")
        }
        assert "uq_feedback_fingerprint" in unique
        checks = {
            constraint["name"]
            for constraint in inspector.get_check_constraints("feedback")
        }
        assert {"ck_feedback_score_range", "ck_feedback_image_sha256_hex"} <= checks
        palette_indexes = {
            index["name"] for index in inspector.get_indexes("palettes")
        }
        assert "ix_palettes_source" in palette_indexes
    finally:
        engine.dispose()


def test_jsonb_columns(training_url):
    engine = _engine(training_url)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT table_name, column_name, data_type "
                    "FROM information_schema.columns "
                    "WHERE table_schema='public' AND data_type='jsonb'"
                )
            ).all()
        found = {(table, column) for table, column, _ in rows}
        assert {
            ("feedback", "palette_json"),
            ("training_runs", "params_json"),
            ("training_runs", "metrics_json"),
            ("training_runs", "calibration_json"),
        } <= found
    finally:
        engine.dispose()


def test_runtime_and_training_databases_share_schema(runtime_url, training_url):
    for url in (runtime_url, training_url):
        engine = _engine(url)
        try:
            assert EXPECTED_TABLES <= set(inspect(engine).get_table_names())
        finally:
            engine.dispose()


def test_fresh_database_has_naming_convention_constraints(training_url):
    """Una DB recién creada tiene exactamente los constraints del modelo.

    Evita divergencias entre una DB local "vieja" y una creada desde cero.
    """
    fresh_url, fresh_name = _create_temp_database(training_url)
    engine = None
    try:
        _upgrade(fresh_url)
        engine = _engine(fresh_url)
        inspector = inspect(engine)

        with engine.connect() as connection:
            versions = connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalars().all()
        assert versions == ["0001_initial"]

        checks = {
            constraint["name"]
            for constraint in inspector.get_check_constraints("feedback")
        }
        assert checks == {
            "ck_feedback_score_range",
            "ck_feedback_image_sha256_hex",
        }

        unique = {
            constraint["name"]
            for constraint in inspector.get_unique_constraints("feedback")
        }
        assert unique == {"uq_feedback_fingerprint"}

        indexes = {index["name"] for index in inspector.get_indexes("feedback")}
        assert {
            "ix_feedback_created_at",
            "ix_feedback_accepted",
            "ix_feedback_accepted_created_at",
        } <= indexes

        tables = set(inspector.get_table_names())
        assert EXPECTED_TABLES <= tables
    finally:
        if engine is not None:
            engine.dispose()
        _drop_temp_database(training_url, fresh_name)

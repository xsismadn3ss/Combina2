"""Guard: ningún módulo de ``app/`` depende de SQLite ni de training.db."""

from __future__ import annotations

from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"


def _sources():
    return sorted(APP_DIR.rglob("*.py"))


def test_app_does_not_import_sqlite3():
    offenders = {
        str(path.relative_to(APP_DIR)): index
        for path in _sources()
        for index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if "import sqlite3" in line or "from sqlite3" in line
    }
    assert offenders == {}


def test_app_does_not_reference_training_db():
    offenders = {
        str(path.relative_to(APP_DIR)): index
        for path in _sources()
        for index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if "training.db" in line
    }
    assert offenders == {}


def test_legacy_sqlite_module_is_gone():
    assert not (APP_DIR / "db" / "sqlite.py").exists()

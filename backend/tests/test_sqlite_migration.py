"""Migración one-shot SQLite (training.db) -> PostgreSQL."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select

from app.db.models import (
    ExtractedColor,
    Feedback,
    Image,
    Palette,
    PaletteColor,
    TrainingRun,
)
from app.repositories import training as training_repo
from scripts import migrate_sqlite_to_postgres

LEGACY_SCHEMA = """
CREATE TABLE images(
  id INTEGER PRIMARY KEY,
  sha256 TEXT UNIQUE NOT NULL,
  width INTEGER, height INTEGER,
  source TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE extracted_colors(
  image_id INTEGER NOT NULL REFERENCES images(id),
  rank INTEGER NOT NULL,
  hex TEXT NOT NULL,
  l REAL NOT NULL, a REAL NOT NULL, b REAL NOT NULL,
  weight REAL NOT NULL,
  PRIMARY KEY(image_id, rank)
);
CREATE TABLE palettes(
  id INTEGER PRIMARY KEY,
  source TEXT NOT NULL,
  rating REAL,
  harmony_label TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE palette_colors(
  palette_id INTEGER NOT NULL REFERENCES palettes(id),
  position INTEGER NOT NULL,
  hex TEXT NOT NULL,
  l REAL NOT NULL, a REAL NOT NULL, b REAL NOT NULL,
  PRIMARY KEY(palette_id, position)
);
CREATE TABLE training_runs(
  id INTEGER PRIMARY KEY,
  model TEXT NOT NULL,
  git_sha TEXT, params_json TEXT,
  metrics_json TEXT,
  keras_path TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE feedback(
  id INTEGER PRIMARY KEY,
  image_id INTEGER REFERENCES images(id),
  palette_id INTEGER REFERENCES palettes(id),
  accepted INTEGER,
  score REAL,
  palette_json TEXT,
  image_sha256 TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);
"""


def _make_sqlite(path: Path) -> Path:
    conn = sqlite3.connect(str(path))
    conn.executescript(LEGACY_SCHEMA)
    conn.execute(
        "INSERT INTO images(id, sha256, width, height, source, created_at)"
        " VALUES(1, ?, 100, 50, 'colorpedia', '2026-10-20 10:00:00')",
        ("f" * 64,),
    )
    conn.executemany(
        "INSERT INTO extracted_colors(image_id, rank, hex, l, a, b, weight)"
        " VALUES(1, ?, ?, 50.0, 10.0, -10.0, ?)",
        [(0, "#FF0000", 0.6), (1, "#00FF00", 0.4)],
    )
    conn.execute(
        "INSERT INTO palettes(id, source, rating, harmony_label, created_at)"
        " VALUES(1, 'synthetic', 0.8, 'triada', '2026-10-20 11:00:00')"
    )
    color_rows = [
        (i, f"#1A2B{i:02X}", 10.5 + i, 20.25 - i, -30.75 + i) for i in range(5)
    ]
    conn.executemany(
        "INSERT INTO palette_colors(palette_id, position, hex, l, a, b)"
        " VALUES(1, ?, ?, ?, ?, ?)",
        color_rows,
    )
    conn.execute(
        "INSERT INTO training_runs(id, model, git_sha, params_json, metrics_json,"
        " keras_path, created_at)"
        " VALUES(7, 'extract', 'abc1234', '{\"seed\": 1}', '{\"mse\": 0.01}',"
        " 'backend/models/extract-v1.keras', '2026-10-20 12:00:00')"
    )
    palette_json = (
        '{"type": "triada", "colors": [{"value": "#FF0000", "role": "primary",'
        ' "weight": null}], "fallback": false}'
    )
    conn.execute(
        "INSERT INTO feedback(id, accepted, score, palette_json, image_sha256,"
        " created_at) VALUES(1, 1, 0.9, ?, NULL, '2026-10-20 13:00:00')",
        (palette_json,),
    )
    conn.commit()
    conn.close()
    return path


def test_migration_dry_run_does_not_write(training_db, tmp_path):
    sqlite_path = _make_sqlite(Path(tmp_path) / "training.db")
    report = migrate_sqlite_to_postgres.run(sqlite_path, apply=False)
    assert report["dry_run"] is True
    assert report["totals"]["read"] == 11  # 1+2+1+5+1+1
    assert report["totals"]["inserted"] == 11
    assert report["totals"]["skipped"] == 0
    for model in (Image, Palette, TrainingRun, Feedback):
        assert training_db.execute(select(func.count()).select_from(model)).scalar_one() == 0


def test_migration_apply_preserves_data_and_ids(training_db, tmp_path):
    sqlite_path = _make_sqlite(Path(tmp_path) / "training.db")
    report = migrate_sqlite_to_postgres.run(sqlite_path, apply=True)
    assert report["dry_run"] is False
    assert report["totals"]["inserted"] == 11

    assert training_db.execute(select(func.count()).select_from(Image)).scalar_one() == 1
    assert (
        training_db.execute(select(func.count()).select_from(ExtractedColor)).scalar_one()
        == 2
    )
    assert training_db.execute(select(func.count()).select_from(Palette)).scalar_one() == 1
    assert (
        training_db.execute(select(func.count()).select_from(PaletteColor)).scalar_one()
        == 5
    )
    stored_colors = list(
        training_db.execute(
            select(PaletteColor).order_by(PaletteColor.position)
        ).scalars()
    )
    # Se copian los l/a/b originales (no se recalculan desde HEX).
    assert [(color.l, color.a, color.b) for color in stored_colors] == [
        (10.5 + i, 20.25 - i, -30.75 + i) for i in range(5)
    ]
    assert (
        training_db.execute(select(func.count()).select_from(TrainingRun)).scalar_one()
        == 1
    )

    run = training_db.get(TrainingRun, 7)
    assert run is not None
    assert run.params_json == {"seed": 1}
    assert run.metrics_json == {"mse": 0.01}
    assert run.created_at == datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)

    feedback = training_db.execute(select(Feedback)).scalars().one()
    assert feedback.source == "sqlite"
    assert feedback.accepted is True
    assert feedback.score == 0.9
    assert feedback.created_at == datetime(2026, 10, 20, 13, 0, tzinfo=timezone.utc)

    image = training_db.execute(select(Image)).scalars().one()
    assert image.sha256 == "f" * 64
    assert image.created_at == datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)

    # Secuencia ajustada: el siguiente run no reutiliza el id 7.
    new_run = training_repo.create_training_run(
        training_db, model="classify", keras_path="backend/models/classify-v3.keras"
    )
    training_db.commit()
    assert int(new_run.id or 0) > 7


def test_migration_is_idempotent(training_db, tmp_path):
    sqlite_path = _make_sqlite(Path(tmp_path) / "training.db")
    migrate_sqlite_to_postgres.run(sqlite_path, apply=True)
    second = migrate_sqlite_to_postgres.run(sqlite_path, apply=True)
    assert second["totals"]["inserted"] == 0
    assert second["totals"]["skipped"] == second["totals"]["read"] == 11

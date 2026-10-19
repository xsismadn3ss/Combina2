"""Capa sqlite (`training.db`, nunca en el repo).

Esquema completo en `docs/data.md`. En Fase 1 solo se usa la tabla `feedback`;
el resto lo puebla la Fase 2.
"""

import os
import sqlite3
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]


def get_db_path() -> Path:
    override = os.environ.get("TRAINING_DB")
    if override:
        return Path(override)
    return BACKEND_DIR / "training.db"


DDL = """
CREATE TABLE IF NOT EXISTS images(
  id INTEGER PRIMARY KEY,
  sha256 TEXT UNIQUE NOT NULL,
  width INTEGER, height INTEGER,
  source TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS extracted_colors(
  image_id INTEGER NOT NULL REFERENCES images(id),
  rank INTEGER NOT NULL,
  hex TEXT NOT NULL,
  l REAL NOT NULL, a REAL NOT NULL, b REAL NOT NULL,
  weight REAL NOT NULL,
  PRIMARY KEY(image_id, rank)
);
CREATE TABLE IF NOT EXISTS palettes(
  id INTEGER PRIMARY KEY,
  source TEXT NOT NULL,
  rating REAL,
  harmony_label TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS palette_colors(
  palette_id INTEGER NOT NULL REFERENCES palettes(id),
  position INTEGER NOT NULL,
  hex TEXT NOT NULL,
  l REAL NOT NULL, a REAL NOT NULL, b REAL NOT NULL,
  PRIMARY KEY(palette_id, position)
);
CREATE TABLE IF NOT EXISTS training_runs(
  id INTEGER PRIMARY KEY,
  model TEXT NOT NULL,
  git_sha TEXT, params_json TEXT,
  metrics_json TEXT,
  keras_path TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS feedback(
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


def get_connection(path: Path | None = None) -> sqlite3.Connection:
    db_path = path or get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    return conn


def insert_feedback(
    palette_json: str,
    accepted: bool,
    score: float | None,
    image_sha256: str | None,
    path: Path | None = None,
) -> int:
    conn = get_connection(path)
    try:
        cur = conn.execute(
            "INSERT INTO feedback(accepted, score, palette_json, image_sha256)"
            " VALUES(?, ?, ?, ?)",
            (1 if accepted else 0, score, palette_json, image_sha256),
        )
        conn.commit()
        return int(cur.lastrowid or 0)
    finally:
        conn.close()

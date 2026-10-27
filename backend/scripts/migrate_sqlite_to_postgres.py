"""Migra los datos de ``backend/training.db`` (SQLite) a la DB de entrenamiento.

Uso::

    python scripts/migrate_sqlite_to_postgres.py --sqlite backend/training.db --dry-run
    python scripts/migrate_sqlite_to_postgres.py --sqlite backend/training.db --apply

- ``--dry-run`` es el modo por defecto: no escribe nada.
- Lee SQLite con ``sqlite3`` de la stdlib; el lector queda aislado aquí (ningún
  módulo de ``app/`` depende de SQLite).
- Orden: images → extracted_colors → palettes → palette_colors →
  training_runs → feedback.
- Dedup: images.sha256, palettes.key, feedback.fingerprint, training_runs.id.
- Conserva ids de ``training_runs`` y ajusta secuencias con ``setval``.
- Convierte ``params_json``/``metrics_json``/``calibration_json`` de TEXT a
  JSONB (``json.loads`` si son strings).
- ``training.db`` original nunca se modifica (solo SELECT).
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE = BACKEND_DIR / "training.db"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import select, text  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.config import training_database_url  # noqa: E402
from app.db.base import utcnow  # noqa: E402
from app.db.engine import sanitized_url  # noqa: E402
from app.db.models import (  # noqa: E402
    ExtractedColor,
    Feedback,
    Image,
    Palette,
    PaletteColor,
    TrainingRun,
)
from app.db.session import training_session  # noqa: E402
from app.ml.datasets import normalize_hex, palette_key  # noqa: E402
from app.repositories import feedback as feedback_repo  # noqa: E402

TABLES = [
    "images",
    "extracted_colors",
    "palettes",
    "palette_colors",
    "training_runs",
    "feedback",
]
HEX_RE_LEN = 64


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif value is None:
        return utcnow()
    else:
        raw = str(value).strip()
        if not raw:
            return utcnow()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return utcnow()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_json(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        parsed = json.loads(text_value)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _valid_sha(value: Any) -> str | None:
    if not value:
        return None
    candidate = str(value).strip().lower()
    if len(candidate) == HEX_RE_LEN and all(c in "0123456789abcdef" for c in candidate):
        return candidate
    return None


def _palette_color_lab(color: sqlite3.Row) -> tuple[float, float, float]:
    """Lab guardado en SQLite; conversión desde HEX solo si falta o es inválido.

    Los valores originales son la fuente de verdad aunque difieran del
    conversor actual (p. ej. redondeos o fórmulas antiguas).
    """
    keys = color.keys()
    if all(key in keys for key in ("l", "a", "b")):
        try:
            values = (float(color["l"]), float(color["a"]), float(color["b"]))
        except (TypeError, ValueError):
            values = None
        if values is not None and all(math.isfinite(value) for value in values):
            return values
    from app.ml.color_spaces import ColorConverter

    return ColorConverter.hex_to_lab(str(color["hex"]))


def _read_rows(conn: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    try:
        return conn.execute(f"SELECT * FROM {table}").fetchall()
    except sqlite3.OperationalError:
        return []


def _new_report() -> dict[str, dict[str, int]]:
    return {table: {"read": 0, "inserted": 0, "skipped": 0} for table in TABLES}


def _exists_image(session: Session, sha256: str) -> int | None:
    return session.execute(
        select(Image.id).where(Image.sha256 == sha256)
    ).scalar_one_or_none()


def _exists_palette(session: Session, key: str) -> int | None:
    return session.execute(
        select(Palette.id).where(Palette.key == key)
    ).scalar_one_or_none()


def _migrate_images(
    conn, session, report, apply: bool, batch_size: int
) -> dict[str, int]:
    mapping: dict[int, int | None] = {}
    items = _read_rows(conn, "images")
    report["images"]["read"] = len(items)
    for index, row in enumerate(items, start=1):
        old_id = int(row["id"])
        sha = str(row["sha256"])
        existing = _exists_image(session, sha)
        if existing is not None:
            mapping[old_id] = int(existing)
            report["images"]["skipped"] += 1
            continue
        report["images"]["inserted"] += 1
        if not apply:
            mapping[old_id] = old_id  # pseudo-id: el dry-run no escribe
            continue
        stmt = (
            pg_insert(Image)
            .values(
                sha256=sha,
                width=row["width"],
                height=row["height"],
                source=row["source"] or "unknown",
                created_at=_parse_dt(row["created_at"]),
            )
            .on_conflict_do_nothing(index_elements=["sha256"])
            .returning(Image.id)
        )
        new_id = session.execute(stmt).scalar_one_or_none()
        if new_id is None:
            new_id = _exists_image(session, sha)
        mapping[old_id] = int(new_id) if new_id is not None else None
        if index % batch_size == 0:
            session.commit()
    return mapping


def _migrate_extracted_colors(
    conn, session, report, apply: bool, image_map: dict[int, int | None]
) -> None:
    items = _read_rows(conn, "extracted_colors")
    report["extracted_colors"]["read"] = len(items)
    existing = {
        (int(image_id), int(rank))
        for image_id, rank in session.execute(
            select(ExtractedColor.image_id, ExtractedColor.rank)
        ).all()
    }
    payload = []
    for row in items:
        new_id = image_map.get(int(row["image_id"]))
        if new_id is None:
            report["extracted_colors"]["skipped"] += 1
            continue
        key = (int(new_id), int(row["rank"]))
        if key in existing:
            report["extracted_colors"]["skipped"] += 1
            continue
        existing.add(key)
        report["extracted_colors"]["inserted"] += 1
        if apply:
            payload.append(
                {
                    "image_id": new_id,
                    "rank": int(row["rank"]),
                    "hex": row["hex"],
                    "l": row["l"],
                    "a": row["a"],
                    "b": row["b"],
                    "weight": row["weight"],
                }
            )
    if apply and payload:
        session.execute(
            pg_insert(ExtractedColor).on_conflict_do_nothing(
                index_elements=["image_id", "rank"]
            ),
            payload,
        )
        session.flush()


def _migrate_palettes(
    conn, session, report, apply: bool, batch_size: int
) -> tuple[dict[int, int | None], set[int]]:
    mapping: dict[int, int | None] = {}
    inserted_old_ids: set[int] = set()
    palettes = _read_rows(conn, "palettes")
    report["palettes"]["read"] = len(palettes)
    colors_by_palette: dict[int, list[sqlite3.Row]] = {}
    for row in _read_rows(conn, "palette_colors"):
        colors_by_palette.setdefault(int(row["palette_id"]), []).append(row)

    for index, row in enumerate(palettes, start=1):
        old_id = int(row["id"])
        colors = sorted(
            colors_by_palette.get(old_id, []), key=lambda item: int(item["position"])
        )
        if not colors:
            report["palettes"]["skipped"] += 1
            continue
        try:
            hexes = [normalize_hex(str(color["hex"])) for color in colors]
        except ValueError:
            report["palettes"]["skipped"] += 1
            continue
        source = row["source"] or "unknown"
        key = palette_key(source, hexes)
        existing = _exists_palette(session, key)
        if existing is not None:
            mapping[old_id] = int(existing)
            report["palettes"]["skipped"] += 1
            continue
        report["palettes"]["inserted"] += 1
        inserted_old_ids.add(old_id)
        if not apply:
            mapping[old_id] = old_id  # pseudo-id: no se usa para escribir
            continue
        stmt = (
            pg_insert(Palette)
            .values(
                key=key,
                source=source,
                rating=row["rating"],
                harmony_label=row["harmony_label"],
                created_at=_parse_dt(row["created_at"]),
            )
            .on_conflict_do_nothing(index_elements=["key"])
            .returning(Palette.id)
        )
        new_id = session.execute(stmt).scalar_one_or_none()
        if new_id is None:
            new_id = _exists_palette(session, key)
        mapping[old_id] = int(new_id) if new_id is not None else None
        if new_id is not None:
            color_payload = []
            for color in colors:
                lab = _palette_color_lab(color)
                color_payload.append(
                    {
                        "palette_id": int(new_id),
                        "position": int(color["position"]),
                        "hex": str(color["hex"]).upper(),
                        "l": lab[0],
                        "a": lab[1],
                        "b": lab[2],
                    }
                )
            session.execute(
                pg_insert(PaletteColor).on_conflict_do_nothing(
                    index_elements=["palette_id", "position"]
                ),
                color_payload,
            )
        if index % batch_size == 0:
            session.commit()
    session.flush()
    return mapping, inserted_old_ids


def _migrate_palette_colors(
    conn,
    session,
    report,
    apply: bool,
    palette_map: dict[int, int | None],
    inserted_palette_ids: set[int],
) -> None:
    """Los colores ya se insertan junto con la paleta; aquí solo se reporta."""
    items = _read_rows(conn, "palette_colors")
    report["palette_colors"]["read"] = len(items)
    for row in items:
        old_palette_id = int(row["palette_id"])
        if old_palette_id in inserted_palette_ids:
            report["palette_colors"]["inserted"] += 1
        else:
            report["palette_colors"]["skipped"] += 1


def _migrate_training_runs(conn, session, report, apply: bool) -> list[int]:
    items = _read_rows(conn, "training_runs")
    report["training_runs"]["read"] = len(items)
    inserted_ids: list[int] = []
    for row in items:
        run_id = int(row["id"])
        exists = session.get(TrainingRun, run_id)
        if exists is not None:
            report["training_runs"]["skipped"] += 1
            continue
        report["training_runs"]["inserted"] += 1
        inserted_ids.append(run_id)
        if not apply:
            continue
        session.add(
            TrainingRun(
                id=run_id,
                model=row["model"] or "unknown",
                git_sha=row["git_sha"],
                params_json=_parse_json(row["params_json"]),
                metrics_json=_parse_json(row["metrics_json"]),
                keras_path=row["keras_path"],
                tf_version=row["tf_version"] if "tf_version" in row.keys() else None,
                keras_version=(
                    row["keras_version"] if "keras_version" in row.keys() else None
                ),
                calibration_json=(
                    _parse_json(row["calibration_json"])
                    if "calibration_json" in row.keys()
                    else None
                ),
                created_at=_parse_dt(row["created_at"]),
            )
        )
    if apply:
        session.flush()
    return inserted_ids


def _migrate_feedback(
    conn, session, report, apply: bool, batch_size: int
) -> None:
    items = _read_rows(conn, "feedback")
    report["feedback"]["read"] = len(items)
    existing = set(session.execute(select(Feedback.fingerprint)).scalars())
    for index, row in enumerate(items, start=1):
        palette_json = _parse_json(row["palette_json"])
        if not palette_json:
            report["feedback"]["skipped"] += 1
            continue
        accepted = bool(row["accepted"]) if row["accepted"] is not None else False
        score = row["score"] if "score" in row.keys() else None
        image_sha = _valid_sha(row["image_sha256"])
        created_at = _parse_dt(row["created_at"])
        fingerprint = feedback_repo.feedback_fingerprint(
            created_at=created_at,
            accepted=accepted,
            score=score,
            palette_json=palette_json,
            image_sha256=image_sha,
        )
        if fingerprint in existing:
            report["feedback"]["skipped"] += 1
            continue
        report["feedback"]["inserted"] += 1
        existing.add(fingerprint)
        if not apply:
            continue
        feedback_repo.insert_feedback(
            session,
            palette_json=palette_json,
            accepted=accepted,
            score=score,
            image_sha256=image_sha,
            source="sqlite",
            created_at=created_at,
        )
        if index % batch_size == 0:
            session.commit()


def _reset_sequences(session: Session) -> None:
    for table in ("images", "palettes", "training_runs", "feedback"):
        session.execute(
            text(
                "SELECT setval("
                f"pg_get_serial_sequence('{table}', 'id'), "
                f"COALESCE((SELECT MAX(id) FROM {table}), 0) + 1, false)"
            )
        )


def run(
    sqlite_path: Path,
    *,
    apply: bool = False,
    batch_size: int = 500,
) -> dict:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"No existe {sqlite_path}")
    report = _new_report()
    conn = sqlite3.connect(f"{sqlite_path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        with training_session() as session:
            image_map = _migrate_images(conn, session, report, apply, batch_size)
            _migrate_extracted_colors(conn, session, report, apply, image_map)
            palette_map, inserted_palette_ids = _migrate_palettes(
                conn, session, report, apply, batch_size
            )
            _migrate_palette_colors(
                conn, session, report, apply, palette_map, inserted_palette_ids
            )
            _migrate_training_runs(conn, session, report, apply)
            _migrate_feedback(conn, session, report, apply, batch_size)
            if apply:
                _reset_sequences(session)
    finally:
        conn.close()
    return {
        "sqlite": str(sqlite_path),
        "db": sanitized_url(training_database_url()),
        "dry_run": not apply,
        "tables": report,
        "totals": {
            key: sum(table[key] for table in report.values())
            for key in ("read", "inserted", "skipped")
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", default=str(DEFAULT_SQLITE))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True)
    group.add_argument("--apply", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args(argv)
    try:
        report = run(
            Path(args.sqlite), apply=args.apply, batch_size=max(1, args.batch_size)
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[migrate_sqlite_to_postgres] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"[migrate_sqlite_to_postgres] {json.dumps(report, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

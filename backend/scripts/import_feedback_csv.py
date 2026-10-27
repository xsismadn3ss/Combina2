"""Importa un CSV exportado de feedback a la DB de entrenamiento.

Uso::

    python scripts/import_feedback_csv.py --input feedback.csv --dry-run
    python scripts/import_feedback_csv.py --input feedback.csv --apply

Lee ``TRAINING_DATABASE_URL``. Preserva ``created_at`` y marca
``source='import'``. La idempotencia la garantiza ``fingerprint`` +
``ON CONFLICT DO NOTHING``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import select  # noqa: E402

from app.config import training_database_url  # noqa: E402
from app.db.engine import sanitized_url  # noqa: E402
from app.db.models import Feedback  # noqa: E402
from app.db.session import training_session  # noqa: E402
from app.repositories import feedback as repo  # noqa: E402
from app.services.feedback import CSV_HEADERS  # noqa: E402


def _parse_created_at(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def _parse_row(row: dict[str, str]) -> dict:
    text = (row.get("palette_json") or "").strip()
    if not text:
        raise ValueError("palette_json vacío")
    palette_json = json.loads(text)
    score = (row.get("score") or "").strip()
    created = (row.get("created_at") or "").strip()
    if not created:
        raise ValueError("created_at vacío")
    image_sha = (row.get("image_sha256") or "").strip() or None
    accepted = (row.get("accepted") or "").strip().lower() in {"1", "true", "yes"}
    return {
        "created_at": _parse_created_at(created),
        "accepted": accepted,
        "score": float(score) if score else None,
        "palette_json": palette_json,
        "image_sha256": image_sha,
    }


def run(input_path: Path, *, apply: bool = False) -> dict:
    read = 0
    invalid = 0
    inserted = 0
    duplicates = 0
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            header = []
        handle.seek(0)
        rows = list(csv.DictReader(handle))
    headers_ok = all(column in header for column in CSV_HEADERS)
    with training_session() as session:
        existing = set(session.execute(select(Feedback.fingerprint)).scalars())
        for raw in rows:
            read += 1
            try:
                parsed = _parse_row(raw)
            except (ValueError, json.JSONDecodeError):
                invalid += 1
                continue
            fingerprint = repo.feedback_fingerprint(
                created_at=parsed["created_at"],
                accepted=parsed["accepted"],
                score=parsed["score"],
                palette_json=parsed["palette_json"],
                image_sha256=parsed["image_sha256"],
            )
            if fingerprint in existing:
                duplicates += 1
                continue
            if not apply:
                inserted += 1  # "se insertaría"
                existing.add(fingerprint)
                continue
            row = repo.insert_feedback(
                session,
                palette_json=parsed["palette_json"],
                accepted=parsed["accepted"],
                score=parsed["score"],
                image_sha256=parsed["image_sha256"],
                source="import",
                created_at=parsed["created_at"],
            )
            if row is None:
                duplicates += 1
            else:
                inserted += 1
                existing.add(fingerprint)
    return {
        "db": sanitized_url(training_database_url()),
        "input": str(input_path),
        "dry_run": not apply,
        "headers_ok": headers_ok,
        "read": read,
        "inserted": inserted,
        "duplicates": duplicates,
        "invalid": invalid,
        "csv_headers_expected": CSV_HEADERS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV de entrada")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True)
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    report = run(Path(args.input), apply=args.apply)
    print(f"[import_feedback_csv] {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Borra feedback anterior a una fecha (retención indefinida, purga manual).

Uso::

    python scripts/purge_feedback.py --before 2025-01-01 --dry-run
    python scripts/purge_feedback.py --before 2025-01-01 --apply

Lee ``DATABASE_URL``. Sin imágenes ni PII: solo snapshots de paleta.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import runtime_database_url  # noqa: E402
from app.db.engine import sanitized_url  # noqa: E402
from app.db.session import runtime_session  # noqa: E402
from app.repositories import feedback as repo  # noqa: E402


def _parse_cutoff(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"fecha inválida: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def run(cutoff: datetime, *, apply: bool = False) -> dict:
    with runtime_session() as session:
        pending = repo.count_before(session, cutoff)
        deleted = repo.delete_before(session, cutoff) if apply else 0
    return {
        "db": sanitized_url(runtime_database_url()),
        "before": cutoff.isoformat(),
        "dry_run": not apply,
        "would_delete": pending,
        "deleted": deleted,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, help="YYYY-MM-DD o ISO-8601")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True)
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    report = run(_parse_cutoff(args.before), apply=args.apply)
    print(f"[purge_feedback] {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Exporta el feedback de runtime a CSV (mismo contrato que la API).

Uso::

    python scripts/export_feedback_csv.py --output feedback.csv
    python scripts/export_feedback_csv.py --output f.csv \
        --from 2026-10-01 --to 2026-10-31 --accepted true --no-sha256

Lee ``DATABASE_URL`` (runtime). UTF-8 sin BOM, encabezados exactos y
mitigación de CSV injection.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import runtime_database_url  # noqa: E402
from app.db.engine import sanitized_url  # noqa: E402
from app.db.session import runtime_session  # noqa: E402
from app.services.feedback import write_feedback_csv  # noqa: E402


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "si", "sí"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("valor booleano inválido")


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"fecha inválida: {value!r}") from exc


def run(
    output: Path,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    accepted: bool | None = None,
    include_sha256: bool = True,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    with runtime_session() as session, output.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        rows = write_feedback_csv(
            session,
            handle,
            start=start,
            end=end,
            accepted=accepted,
            include_sha256=include_sha256,
        )
    return {
        "db": sanitized_url(runtime_database_url()),
        "output": str(output),
        "rows": rows,
        "include_sha256": include_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Ruta del CSV de salida")
    parser.add_argument("--from", dest="start", default=None, help="Desde (ISO-8601)")
    parser.add_argument("--to", dest="end", default=None, help="Hasta (ISO-8601)")
    parser.add_argument("--accepted", type=_parse_bool, default=None)
    parser.add_argument("--no-sha256", action="store_true")
    args = parser.parse_args(argv)
    report = run(
        Path(args.output),
        start=_parse_dt(args.start),
        end=_parse_dt(args.end),
        accepted=args.accepted,
        include_sha256=not args.no_sha256,
    )
    print(f"[export_feedback_csv] {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

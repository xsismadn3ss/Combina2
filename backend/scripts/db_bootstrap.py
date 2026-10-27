"""Crea y migra las bases PostgreSQL (local, CI y pre-deploy).

Uso (desde ``backend/``)::

    python scripts/db_bootstrap.py --target all --create --upgrade
    python scripts/db_bootstrap.py --target runtime --upgrade
    python scripts/db_bootstrap.py --target training --create

``--create`` conecta a la DB ``postgres`` con autocommit y crea la base si
falta. ``--upgrade`` ejecuta ``alembic upgrade head`` con la URL de cada base.
Es idempotente: se puede correr varias veces (y en cada arranque de CI).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from psycopg import sql  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402

from app.config import (  # noqa: E402
    db_connect_timeout,
    runtime_database_url,
    training_database_url,
)

TARGETS = ("runtime", "training", "all")


def target_urls(target: str) -> dict[str, str]:
    if target == "runtime":
        return {"runtime": runtime_database_url()}
    if target == "training":
        return {"training": training_database_url()}
    return {
        "runtime": runtime_database_url(),
        "training": training_database_url(),
    }


def _psycopg_dsn(url: str, database: str | None = None) -> str:
    parsed = make_url(url)
    if database is not None:
        parsed = parsed.set(database=database)
    # psycopg no entiende el sufijo de driver de SQLAlchemy.
    parsed = parsed.set(drivername="postgresql")
    return parsed.render_as_string(hide_password=False)


def create_database(url: str) -> str:
    """Crea la base si no existe; devuelve ``created``/``exists``."""
    import psycopg

    parsed = make_url(url)
    database = parsed.database
    if not database:
        raise SystemExit(f"URL sin nombre de base: {parsed.render_as_string()}")
    with psycopg.connect(
        _psycopg_dsn(url, database="postgres"),
        autocommit=True,
        connect_timeout=db_connect_timeout(),
    ) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (database,)
        ).fetchone()
        if exists:
            return "exists"
        conn.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
        )
    return "created"


def upgrade_database(url: str) -> None:
    """Aplica ``alembic upgrade head`` en la URL indicada (advisory lock)."""
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


def run(target: str = "all", *, create: bool = False, upgrade: bool = False) -> dict:
    report: dict[str, dict[str, str]] = {}
    for name, url in target_urls(target).items():
        result: dict[str, str] = {"database": make_url(url).database or ""}
        if create:
            result["create"] = create_database(url)
        if upgrade:
            upgrade_database(url)
            result["upgrade"] = "head"
        report[name] = result
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, default="all")
    parser.add_argument(
        "--create", action="store_true", help="Crea las bases si faltan"
    )
    parser.add_argument(
        "--upgrade", action="store_true", help="Ejecuta alembic upgrade head"
    )
    args = parser.parse_args(argv)
    if not args.create and not args.upgrade:
        parser.error("indica --create y/o --upgrade")
    try:
        report = run(args.target, create=args.create, upgrade=args.upgrade)
    except Exception as exc:  # noqa: BLE001
        print(f"[db_bootstrap] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for name, result in report.items():
        print(f"[db_bootstrap] {name}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI de gestión del registro de modelos (B3).

Uso (desde ``backend/``; también funciona con el intérprete base, sin TF)::

    python scripts/manage_models.py list
    python scripts/manage_models.py verify [--slot classify]
    python scripts/manage_models.py activate --slot classify --file classify-v3.keras --note "..."
    python scripts/manage_models.py rollback --slot classify [--dry-run]

Todos los subcomandos aceptan ``--json`` para salida legible por máquina.
`verify` sale con código 1 si falta un archivo o su sha256 no coincide.
`rollback --dry-run` imprime el intercambio sin modificar el manifiesto.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services import model_registry  # noqa: E402


def _short(sha: str | None) -> str:
    return sha[:12] if sha else "-"


def _slot_payload(slot: str, entry: dict[str, Any]) -> dict[str, Any]:
    files = entry.get("files") or {}
    return {
        "slot": slot,
        "active": entry.get("active"),
        "previous": entry.get("previous"),
        "versions": list(entry.get("versions") or []),
        "files": {
            name: {"sha256": (meta or {}).get("sha256")}
            for name, meta in files.items()
            if isinstance(meta, dict)
        },
    }


def cmd_list(args: argparse.Namespace) -> int:
    manifest = model_registry.load_manifest()
    slots = model_registry.list_slots()
    rows = [_slot_payload(slot, manifest.get(slot) or {}) for slot in slots]
    if args.json:
        print(json.dumps({"slots": rows}, indent=2, ensure_ascii=False))
        return 0
    if not rows:
        print("(manifiesto sin slots)")
        return 0
    for row in rows:
        files = row["files"]
        versions = ", ".join(
            f"{name}({_short(files.get(name, {}).get('sha256'))})"
            for name in row["versions"]
        )
        print(
            f"{row['slot']:<9} active={row['active'] or '-':<24} "
            f"previous={row['previous'] or '-':<24} versions=[{versions}]"
        )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    result = model_registry.verify(args.slot)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        scope = args.slot or ", ".join(model_registry.list_slots())
        print(f"verify({scope}): {'OK' if result['ok'] else 'FAIL'}")
        for item in result["missing"]:
            print(f"  FALTA   {item['slot']}/{item['file']}")
        for item in result["hash_mismatch"]:
            print(
                f"  HASH    {item['slot']}/{item['file']} "
                f"esperado={_short(item['expected'])} real={_short(item['actual'])}"
            )
        for item in result["unhashed"]:
            print(f"  SIN HASH {item['slot']}/{item['file']} (legacy, no verificable)")
    return 0 if result["ok"] else 1


def cmd_activate(args: argparse.Namespace) -> int:
    try:
        result = model_registry.activate(args.slot, args.file, note=args.note)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(
            f"activado {result['slot']}={result['active']} "
            f"(previous={result['previous'] or '-'}, sha={_short(result['sha256'])})"
        )
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    manifest = model_registry.load_manifest()
    entry = manifest.get(args.slot) or {}
    previous = entry.get("previous")
    current = entry.get("active")
    if not previous:
        print(
            f"ERROR: el slot {args.slot!r} no tiene 'previous' para rollback",
            file=sys.stderr,
        )
        return 1
    payload = {"from": current, "to": previous, "dry_run": bool(args.dry_run)}
    if args.dry_run:
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"rollback (dry-run) {args.slot}: {current} -> {previous}")
        return 0
    try:
        result = model_registry.rollback(args.slot)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    payload = {**result, "dry_run": False}
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"rollback {args.slot}: {result['from']} -> {result['to']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="Lista slots, activo, previous y versiones")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_verify = sub.add_parser("verify", help="Comprueba existencia y sha256")
    p_verify.add_argument("--slot", choices=("extract", "generate", "classify"))
    p_verify.add_argument("--json", action="store_true")
    p_verify.set_defaults(func=cmd_verify)

    p_activate = sub.add_parser("activate", help="Activa una versión ya presente")
    p_activate.add_argument("--slot", required=True, choices=("extract", "generate", "classify"))
    p_activate.add_argument("--file", required=True, help="Nombre del .keras en backend/models/")
    p_activate.add_argument("--note")
    p_activate.add_argument("--json", action="store_true")
    p_activate.set_defaults(func=cmd_activate)

    p_rollback = sub.add_parser("rollback", help="Intercambia active/previous")
    p_rollback.add_argument("--slot", required=True, choices=("extract", "generate", "classify"))
    p_rollback.add_argument("--dry-run", action="store_true")
    p_rollback.add_argument("--json", action="store_true")
    p_rollback.set_defaults(func=cmd_rollback)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

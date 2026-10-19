"""Fetch COLOURLovers TOP → tablas `palettes`/`palette_colors` (idempotente).

Recorre offsets 0..900 (paso 100) × 8 hueOptions
(red, orange, yellow, green, aqua, blue, violet, fuchsia) ≈ hasta 8.000
paletas de 5 colores con ranking (`numResults=100` por página).

ESTADO (verificado 2026-10-19): la API responde `410 Gone` en todas las
combinaciones — el endpoint `/api/palettes/top` está caído o cambió sin
aviso. El script lo documenta en el reporte, inserta 0 filas y sale con
código 0 para no romper el resto del pipeline; el volumen lo cubre
`make_synthetic.py`. Solo stdlib (urllib), sin dependencias nuevas.

Uso:
    backend/.venv/bin/python backend/scripts/fetch_colourlovers.py
    TRAINING_DB=/tmp/t.db backend/.venv/bin/python backend/scripts/fetch_colourlovers.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.db.sqlite import get_connection, get_db_path  # noqa: E402
from app.ml.datasets import (  # noqa: E402
    count_palettes_by_source,
    insert_palette,
    load_palette_keys,
)

SOURCE = "colourlovers"
BASE_URL = "https://www.colourlovers.com/api/palettes/top"
HUE_OPTIONS = ["red", "orange", "yellow", "green", "aqua", "blue", "violet", "fuchsia"]
OFFSETS = list(range(0, 1000, 100))  # 0..900
NUM_RESULTS = 100
TIMEOUT = 20
RETRIES = 3


def fetch_page(hue: str, offset: int) -> tuple[list | None, str]:
    """Descarga una página; devuelve (paletas|None, nota de estado)."""
    url = (
        f"{BASE_URL}?format=json&showPaletteWidths=1"
        f"&numResults={NUM_RESULTS}&resultOffset={offset}&hueOption={hue}"
    )
    last_error = ""
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Combina2/0.1 (research dataset fetch)"}
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8")), "ok"
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code in (400, 403, 404, 410):
                break  # permanente: no reintentar
            time.sleep(2 * attempt)
        except Exception as exc:  # red, timeout, JSON roto...
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2 * attempt)
    return None, last_error


def _normalize_rating(value) -> float | None:
    try:
        return float(value) / 5.0 if value is not None else None
    except (TypeError, ValueError):
        return None


def run(limit: int | None = None) -> dict:
    conn = get_connection()
    errors: list[str] = []
    pages_ok = 0
    seen_api = 0
    inserted = 0
    skipped = 0
    try:
        existing = load_palette_keys(conn)
        stop = False
        for hue in HUE_OPTIONS:
            for offset in OFFSETS:
                palettes, status = fetch_page(hue, offset)
                if palettes is None:
                    errors.append(f"hue={hue} offset={offset}: {status}")
                    continue
                pages_ok += 1
                if not palettes:
                    break  # sin más resultados para este hue
                for pal in palettes:
                    colors = pal.get("colors") or []
                    hexes = [c for c in (col.get("hex") if isinstance(col, dict) else col for col in colors) if c]
                    if len(hexes) < 2:
                        skipped += 1
                        continue
                    seen_api += 1
                    pid = insert_palette(
                        conn,
                        source=SOURCE,
                        hexes=[h if h.startswith("#") else f"#{h}" for h in hexes],
                        rating=_normalize_rating(pal.get("ranking") or pal.get("numVotes")),
                        harmony_label=None,
                        existing_keys=existing,
                    )
                    if pid is None:
                        skipped += 1
                    else:
                        inserted += 1
                    if limit is not None and seen_api >= limit:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break
        conn.commit()
        by_source = count_palettes_by_source(conn)
        total = int(conn.execute("SELECT COUNT(*) FROM palettes").fetchone()[0])
    finally:
        conn.close()
    return {
        "db": str(get_db_path()),
        "pages_ok": pages_ok,
        "pages_total": len(HUE_OPTIONS) * len(OFFSETS),
        "seen_api": seen_api,
        "inserted": inserted,
        "skipped_duplicates_or_invalid": skipped,
        "errors": errors,
        "by_source": by_source,
        "total_palettes": total,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Tope de paletas (debug)")
    args = parser.parse_args(argv)
    report = run(limit=args.limit)
    print(
        f"[fetch_colourlovers] db={report['db']} pages_ok={report['pages_ok']}/"
        f"{report['pages_total']} seen={report['seen_api']} inserted={report['inserted']} "
        f"skipped={report['skipped_duplicates_or_invalid']} total={report['total_palettes']} "
        f"by_source={report['by_source']}"
    )
    if report["errors"]:
        print(
            "[fetch_colourlovers] AVISO: la API falló o cambió "
            f"({len(report['errors'])} páginas con error, muestro 5):"
        )
        for err in report["errors"][:5]:
            print(f"  - {err}")
        print("  Se sigue con el resto del pipeline; el sintético cubre el volumen.")
    return report


if __name__ == "__main__":
    main()

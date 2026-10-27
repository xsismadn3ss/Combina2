"""Fetch HF `boltuix/color-pedia` (~50k, Parquet) → `palettes`/`palette_colors`.

El dataset son ~50k colores sueltos (HEX/RGB + nombre + emoción); se agrupan
de 5 en 5 en orden de fila para formar paletas de 5 colores
(`source=colorpedia`, sin rating ni etiqueta). Idempotente vía dedup sha256.
Si no hay acceso a HF o falla la descarga/parseo, lo documenta en el reporte
e inserta 0 filas con código de salida 0 (sigue el resto del pipeline).

Requiere `pyarrow` (ver `backend/pyproject.toml`).

Uso:
    python scripts/fetch_colorpedia.py
    TRAINING_DATABASE_URL=postgresql+psycopg://.../t python scripts/fetch_colorpedia.py --limit 200
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.config import training_database_url  # noqa: E402
from app.db.engine import sanitized_url  # noqa: E402
from app.db.session import training_session  # noqa: E402
from app.ml.datasets import (  # noqa: E402
    count_palettes,
    count_palettes_by_source,
    insert_palette,
    load_palette_keys,
    normalize_hex,
)

SOURCE = "colorpedia"
CHUNK = 5
BATCH_SIZE = 200  # commit por lote (no una transacción gigante)
TIMEOUT = 120
URLS = [
    "https://huggingface.co/datasets/boltuix/color-pedia/resolve/main/color_pedia.parquet",
    # Copia convertida (más ligera) como respaldo:
    "https://huggingface.co/datasets/boltuix/color-pedia/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
]


def download(urls: list[str], dest: Path) -> str:
    """Descarga la primera URL que funcione; devuelve la URL usada o lanza."""
    last_error = ""
    for url in urls:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Combina2/0.1 (research dataset fetch)"}
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp, open(dest, "wb") as fh:
                while True:
                    block = resp.read(1024 * 1024)
                    if not block:
                        break
                    fh.write(block)
            return url
        except Exception as exc:
            last_error = f"{url}: {type(exc).__name__}: {exc}"
    raise RuntimeError(f"No se pudo descargar Color-Pedia. {last_error}")


def _resolve_columns(names: list[str]) -> tuple[str | None, tuple[str, str, str] | None]:
    low = {n.lower().strip(): n for n in names}
    hex_col = next((low[k] for k in ("hex code", "hex", "hexcode", "hex_code") if k in low), None)
    rgb_cols = None
    if all(k in low for k in ("r", "g", "b")):
        rgb_cols = (low["r"], low["g"], low["b"])
    return hex_col, rgb_cols


def iter_color_hexes(path: Path):
    """Genera HEX `#RRGGBB` fila a fila desde el Parquet (streaming)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    hex_col, rgb_cols = _resolve_columns(pf.schema.names)
    if hex_col is None and rgb_cols is None:
        raise RuntimeError(f"Columnas no reconocidas: {pf.schema.names}")
    cols = [c for c in (hex_col, *(rgb_cols or ())) if c is not None]
    for batch in pf.iter_batches(batch_size=5000, columns=cols):
        table = batch.to_pylist()
        for row in table:
            hexv = (row.get(hex_col) or "") if hex_col else ""
            hexv = str(hexv).strip()
            if not hexv and rgb_cols:
                try:
                    hexv = "#%02X%02X%02X" % (
                        int(row[rgb_cols[0]]),
                        int(row[rgb_cols[1]]),
                        int(row[rgb_cols[2]]),
                    )
                except (TypeError, ValueError):
                    continue
            if not hexv:
                continue
            try:
                yield normalize_hex(hexv if hexv.startswith("#") else f"#{hexv}")
            except ValueError:
                continue


def run(limit: int | None = None) -> dict:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return {
            "db": sanitized_url(training_database_url()),
            "inserted": 0,
            "skipped_duplicates_or_invalid": 0,
            "colors_seen": 0,
            "error": "pyarrow no instalado (uv pip install pyarrow); se omite Color-Pedia.",
            "by_source": {},
            "total_palettes": 0,
        }
    with training_session() as session:
        existing = load_palette_keys(session)
        with tempfile.TemporaryDirectory(prefix="colorpedia-") as tmp:
            dest = Path(tmp) / "color_pedia.parquet"
            try:
                used_url = download(URLS, dest)
            except RuntimeError as exc:
                return {
                    "db": sanitized_url(training_database_url()),
                    "inserted": 0,
                    "skipped_duplicates_or_invalid": 0,
                    "colors_seen": 0,
                    "error": str(exc),
                    "by_source": count_palettes_by_source(session),
                    "total_palettes": count_palettes(session),
                }
            inserted = 0
            skipped = 0
            colors_seen = 0
            chunk: list[str] = []
            for hexv in iter_color_hexes(dest):
                colors_seen += 1
                chunk.append(hexv)
                if len(chunk) == CHUNK:
                    pid = insert_palette(
                        session,
                        source=SOURCE,
                        hexes=list(chunk),
                        rating=None,
                        harmony_label=None,
                        existing_keys=existing,
                    )
                    if pid is None:
                        skipped += 1
                    else:
                        inserted += 1
                        if inserted % BATCH_SIZE == 0:
                            session.commit()
                    chunk = []
                    if limit is not None and inserted + skipped >= limit:
                        break
            session.commit()
            by_source = count_palettes_by_source(session)
            total = count_palettes(session)
    return {
        "db": sanitized_url(training_database_url()),
        "url": used_url,
        "colors_seen": colors_seen,
        "inserted": inserted,
        "skipped_duplicates_or_invalid": skipped,
        "by_source": by_source,
        "total_palettes": total,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Tope de paletas (debug)")
    args = parser.parse_args(argv)
    report = run(limit=args.limit)
    print(
        f"[fetch_colorpedia] db={report['db']} colors_seen={report.get('colors_seen', 0)} "
        f"inserted={report['inserted']} skipped={report['skipped_duplicates_or_invalid']} "
        f"total={report['total_palettes']} by_source={report['by_source']}"
    )
    if report.get("error"):
        print(f"[fetch_colorpedia] AVISO: {report['error']} Se sigue con el resto.")
    return report


if __name__ == "__main__":
    main()

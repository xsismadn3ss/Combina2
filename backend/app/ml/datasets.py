"""Preproceso común y helpers reutilizables del dataset (`training.db`).

Fuente de verdad del esquema: `docs/data.md` (DDL) y `app/db/sqlite.py`.
Este módulo lo usan los scripts de Fase 2 (`backend/scripts/`):

- `fetch_colorpedia.py` y `fetch_colourlovers.py` (fuentes públicas).
- `make_synthetic.py` (destilación del motor clásico, 100% offline).

Convenciones:

- Lado mayor de imagen: 300px · submuestreo: 3000 px · RGB→CIELAB vía
  `app/ml/color_spaces.py` · normalización L∈[0,1], a/b∈[-1,1].
- Dedup de imágenes por `sha256` (UNIQUE en `images`).
- Dedup de paletas por `sha256` del contenido canónico
  (`source|HEX,HEX,...` en mayúsculas): la tabla `palettes` no tiene columna
  hash, así que la comprobación se hace en Python cargando las firmas
  existentes una vez (`load_palette_keys`) antes de insertar en lote.
- Split 80/10/10 estratificado por `(source, harmony_label)`, reproducible
  con semilla fija.
"""

from __future__ import annotations

import hashlib
import io
import random
import re
import sqlite3

import numpy as np
from PIL import Image

from app.ml.color_spaces import ColorConverter

MAX_SIDE = 300
SAMPLE_PIXELS = 3000

VALID_HARMONY_LABELS = frozenset(
    {"complementario", "triada", "analogo", "monocromatica"}
)
VALID_SOURCES = frozenset(
    {"colourlovers", "colorpedia", "wikiart", "synthetic", "user"}
)

SPLIT_RATIOS = (0.8, 0.1, 0.1)

_HEX_RE = re.compile(r"^#[0-9A-F]{6}$")


def normalize_hex(hex_str: str) -> str:
    """Valida y normaliza a `#RRGGBB` en mayúsculas."""
    h = hex_str.strip().upper()
    if not h.startswith("#"):
        h = "#" + h
    if not _HEX_RE.match(h):
        raise ValueError(f"Código HEX inválido: {hex_str!r}")
    return h


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def palette_key(source: str, hexes: list[str]) -> str:
    """Firma sha256 canónica de una paleta para el dedup idempotente."""
    canonical = source + "|" + ",".join(normalize_hex(h) for h in hexes)
    return sha256_hex(canonical.encode("utf-8"))


# --- Imágenes: carga + resize + submuestreo + Lab -----------------------------


def load_image_rgb(image_bytes: bytes) -> Image.Image:
    """Decodifica a RGB (RGBA→RGB sobre blanco, como el motor clásico)."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        if image.mode in ("RGBA", "LA", "PA"):
            image = image.convert("RGBA")
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            return background
        return image.convert("RGB")


def resize_max_side(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """Reescala para que el lado mayor mida `max_side` (sin ampliar)."""
    w, h = img.size
    side = max(w, h)
    if side <= max_side:
        return img
    scale = max_side / float(side)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def sample_pixels(rgb: np.ndarray, n: int = SAMPLE_PIXELS, seed: int = 42) -> np.ndarray:
    """Submuestreo determinista de `n` píxeles (con reemplazo si faltan)."""
    flat = rgb.reshape(-1, 3)
    rng = np.random.RandomState(seed)
    idx = rng.choice(flat.shape[0], size=n, replace=flat.shape[0] < n)
    return flat[idx]


def rgb_batch_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (N,3) → CIELAB float64 (N,3) vía `ColorConverter`."""
    return np.array(
        [ColorConverter.rgb_to_lab((int(r), int(g), int(b))) for r, g, b in rgb],
        dtype=np.float64,
    )


def lab_batch_normalize(lab: np.ndarray) -> np.ndarray:
    """CIELAB (N,3) → normalizado L∈[0,1], a/b∈[-1,1] aprox."""
    out = np.empty_like(lab)
    out[:, 0] = lab[:, 0] / 100.0
    out[:, 1] = np.clip(lab[:, 1] / 128.0, -1.0, 1.0)
    out[:, 2] = np.clip(lab[:, 2] / 128.0, -1.0, 1.0)
    return out


def preprocess_image(
    image_bytes: bytes,
    *,
    max_side: int = MAX_SIDE,
    sample_n: int = SAMPLE_PIXELS,
    seed: int = 42,
) -> dict:
    """Preproceso común: resize lado mayor + submuestreo + RGB→Lab normalizado."""
    if not image_bytes:
        raise ValueError("El archivo está vacío")
    img = resize_max_side(load_image_rgb(image_bytes), max_side)
    rgb = np.asarray(img, dtype=np.uint8)
    sampled = sample_pixels(rgb, sample_n, seed)
    lab = lab_batch_normalize(rgb_batch_to_lab(sampled))
    return {
        "sha256": sha256_hex(image_bytes),
        "width": img.size[0],
        "height": img.size[1],
        "pixels_lab": lab,  # (sample_n, 3) float64 normalizado
    }


# --- Escritura idempotente ----------------------------------------------------


def insert_image(
    conn: sqlite3.Connection, *, sha256: str, width: int, height: int, source: str
) -> int | None:
    """Inserta en `images`; devuelve el id o None si ya existía (dedup)."""
    if source not in VALID_SOURCES:
        raise ValueError(f"Fuente desconocida: {source!r}")
    cur = conn.execute(
        "INSERT OR IGNORE INTO images(sha256, width, height, source) VALUES(?, ?, ?, ?)",
        (sha256, width, height, source),
    )
    if cur.rowcount == 0:
        return None
    return int(cur.lastrowid or 0)


def load_palette_keys(conn: sqlite3.Connection) -> set[str]:
    """Firmas sha256 de todas las paletas ya guardadas (para dedup en lote)."""
    keys: set[str] = set()
    for pid, source, hexes in _iter_palettes(conn):
        keys.add(palette_key(source, hexes))
    return keys


def _iter_palettes(conn: sqlite3.Connection):
    rows = conn.execute(
        "SELECT p.id, p.source, c.position, c.hex FROM palettes p"
        " JOIN palette_colors c ON c.palette_id = p.id"
        " ORDER BY p.id, c.position"
    )
    current: tuple[int, str, list[str]] | None = None
    for pid, source, _pos, hexv in rows:
        if current is None or current[0] != pid:
            if current is not None:
                yield current[0], current[1], current[2]
            current = (pid, source, [hexv])
        else:
            current[2].append(hexv)
    if current is not None:
        yield current[0], current[1], current[2]


def insert_palette(
    conn: sqlite3.Connection,
    *,
    source: str,
    hexes: list[str],
    rating: float | None = None,
    harmony_label: str | None = None,
    existing_keys: set[str] | None = None,
) -> int | None:
    """Inserta una paleta + sus colores Lab; None si duplicada (idempotente).

    Si se pasa `existing_keys` (de `load_palette_keys`) se usa y actualiza
    en sitio para insertar lotes sin releer la DB.
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"Fuente desconocida: {source!r}")
    if harmony_label is not None and harmony_label not in VALID_HARMONY_LABELS:
        raise ValueError(f"Etiqueta de armonía inválida: {harmony_label!r}")
    if not hexes:
        raise ValueError("La paleta no tiene colores")
    norm = [normalize_hex(h) for h in hexes]
    key = palette_key(source, norm)
    if existing_keys is not None:
        if key in existing_keys:
            return None
    else:
        if key in load_palette_keys(conn):
            return None
    cur = conn.execute(
        "INSERT INTO palettes(source, rating, harmony_label) VALUES(?, ?, ?)",
        (source, rating, harmony_label),
    )
    pid = int(cur.lastrowid or 0)
    for pos, hexv in enumerate(norm):
        lab = ColorConverter.hex_to_lab(hexv)
        conn.execute(
            "INSERT INTO palette_colors(palette_id, position, hex, l, a, b)"
            " VALUES(?, ?, ?, ?, ?, ?)",
            (pid, pos, hexv, lab[0], lab[1], lab[2]),
        )
    if existing_keys is not None:
        existing_keys.add(key)
    return pid


def count_palettes_by_source(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT source, COUNT(*) FROM palettes GROUP BY source"
        ).fetchall()
    }


# --- Split estratificado reproducible -----------------------------------------


def fetch_palette_strata(conn: sqlite3.Connection) -> list[tuple[int, str, str]]:
    """Lista (id, source, harmony_label|'' ) de todas las paletas."""
    return [
        (row[0], row[1], row[2] or "")
        for row in conn.execute(
            "SELECT id, source, harmony_label FROM palettes"
        ).fetchall()
    ]


def stratified_split(
    strata: list[tuple[int, str]],
    *,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Reparte ids en train/val/test estratificado por clave, reproducible.

    `strata`: lista de (id, clave_de_estrato). Suma exacta: todo id cae en
    exactamente un split.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Los ratios deben sumar 1.0")
    groups: dict[str, list[int]] = {}
    for pid, key in strata:
        groups.setdefault(key, []).append(pid)
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng = random.Random(f"{seed}:{key}")
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        splits["train"].extend(ids[:n_train])
        splits["val"].extend(ids[n_train : n_train + n_val])
        splits["test"].extend(ids[n_train + n_val :])
    return splits


def palette_splits(
    conn: sqlite3.Connection,
    *,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Split 80/10/10 de la DB estratificado por (source, harmony_label)."""
    strata = [(pid, f"{src}\x00{lab}") for pid, src, lab in fetch_palette_strata(conn)]
    return stratified_split(strata, ratios=ratios, seed=seed)

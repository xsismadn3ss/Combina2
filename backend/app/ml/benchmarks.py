"""Builders deterministas de datasets para entrenamiento y evaluación (issue #22).

Centraliza lo que antes vivía duplicado en ``scripts/train_extract.py``,
``scripts/train_generate.py`` y ``scripts/train_classify.py``. Los tests y
``eval/run_eval.py`` importan **este** módulo (nunca ``scripts/`` vía
``importlib``).

Convenciones:

- Todo es determinista con ``seed``; no se consulta red.
- La DB de entrenamiento (PostgreSQL) se usa si tiene paletas; si no (o si
  faltan), se completa con paletas sintéticas offline. El repo actual tiene la
  DB sin paletas, así que los sets son sintéticos y reproducibles.
- Las paletas de G/C son dicts ``{"hexes", "rating"?, "harmony_label"}``.
- Numpy/pillow puro a nivel de módulo. TensorFlow y la inferencia (E) se
  importan de forma perezosa dentro de funciones.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = BACKEND_DIR / ".eval-cache"

HARMONY_LABELS = ["complementario", "triada", "analogo", "monocromatica"]
HARMONY_TO_IDX = {h: i for i, h in enumerate(HARMONY_LABELS)}
FEATURE_DIM_V1 = 40
FEATURE_DIM_V2 = 54

# Set fijo de paridad/eval de E: 10 paletas diversas, deterministas.
EXTRACT_EVAL_PALETTES: list[list[str]] = [
    ["#FF0000", "#00FF00", "#0000FF"],
    ["#FFFFFF", "#000000"],
    ["#123456", "#ABCDEF", "#FEDCBA", "#445566"],
    ["#FF8800", "#0088FF", "#88FF00", "#FF0088", "#00FF88"],
    ["#5A3D2B", "#9C8E82", "#2B3A5A", "#7A9C5A"],
    ["#F5F5DC", "#8B4513", "#2F4F4F"],
    ["#E63946", "#F1FA8C", "#1D3557", "#A8DADC", "#457B9D"],
    ["#111111", "#888888", "#EEEEEE"],
    ["#7B2D26", "#F0E6EF", "#0A3200", "#E0A458"],
    ["#003049", "#D62828", "#F77F00", "#FCBF49", "#EAE2B7"],
]

# Tamaños/semillas de la línea base (eval/baseline.json).
EXTRACT_EVAL_SIZE = 64
EXTRACT_EVAL_JPEG_QUALITY = 80
EXTRACT_EVAL_SEED0 = 5000
GENERATE_EVAL_N = 40
GENERATE_EVAL_SEED = 99
CLASSIFY_SPLIT_SEED = 42
CLASSIFY_N_TRAIN = 600
CLASSIFY_N_VAL = 120
CLASSIFY_N_TEST = 80
CLASSIFY_TEST_SEED_OFFSET = 20_000
# Partición disjunta de las paletas de la DB de entrenamiento entre train/val/test. El
# barajado es fijo y cada split recibe un bucket distinto (seed % 3), así una
# misma paleta de DB nunca aparece en dos splits.
CLASSIFY_DB_PARTITION_SEED = 7_313
CLASSIFY_DB_BUCKETS = 3
CLASSIFY_SYNTHETIC_SEED_OFFSET = 777_000


def git_sha() -> str:
    """SHA corto del repo, o ``"unknown"`` si no está disponible."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=BACKEND_DIR.parent,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --- E: paletas + mosaicos ----------------------------------------------------


def load_extract_palettes(limit: int) -> list[list[str]]:
    """HEX por paleta desde la DB de entrenamiento (vacío si no hay DB/contenido)."""
    from sqlalchemy import select

    from app.db.models import PaletteColor
    from app.db.session import training_session

    try:
        with training_session() as session:
            rows = session.execute(
                select(PaletteColor.palette_id, PaletteColor.hex).order_by(
                    PaletteColor.palette_id, PaletteColor.position
                )
            ).all()
    except Exception:
        return []
    palettes: list[list[str]] = []
    current: list[str] = []
    last_pid = None
    for pid, hexv in rows:
        if last_pid is None or pid != last_pid:
            if current:
                palettes.append(current)
                if len(palettes) >= limit:
                    break
            current = [hexv]
            last_pid = pid
        else:
            current.append(hexv)
    if current and len(palettes) < limit:
        palettes.append(current)
    return palettes[:limit]


def random_palette(rng: np.random.RandomState) -> list[str]:
    n = int(rng.randint(3, 7))
    return [
        "#%02X%02X%02X" % tuple(int(v) for v in rng.randint(0, 256, size=3))
        for _ in range(n)
    ]


def build_extract_palette_list(n: int, seed: int) -> list[list[str]]:
    """Paletas DB (hasta ``n``) + relleno aleatorio offline. Determinista."""
    palettes = load_extract_palettes(n)
    rng = np.random.RandomState(seed + 1_000_000)
    while len(palettes) < n:
        palettes.append(random_palette(rng))
    return palettes[:n]


def synthesize_extract_dataset(
    palettes: list[list[str]],
    *,
    seed: int,
    size: int,
    quality_lo: int = 60,
    quality_hi: int = 95,
) -> list[bytes]:
    """Un mosaico JPEG por paleta (pesos Dirichlet, orden aleatorio)."""
    from app.ml.datasets import synthesize_mosaic

    rng = np.random.RandomState(seed)
    images: list[bytes] = []
    for hexes in palettes:
        order = rng.permutation(len(hexes))
        hexes_s = [hexes[i] for i in order]
        weights = rng.dirichlet(np.ones(len(hexes_s)))
        images.append(
            synthesize_mosaic(
                hexes_s,
                list(weights),
                size=size,
                seed=int(rng.randint(0, 2**31 - 1)),
                jpeg_quality=int(rng.randint(quality_lo, quality_hi + 1)),
            )
        )
    return images


def extract_eval_cases(
    *,
    size: int = EXTRACT_EVAL_SIZE,
    jpeg_quality: int = EXTRACT_EVAL_JPEG_QUALITY,
    seed0: int = EXTRACT_EVAL_SEED0,
) -> list[dict]:
    """Casos fijos de paridad E: 10 mosaicos (seeds 5000..5009, 64 px, JPEG 80)."""
    from app.ml.datasets import synthesize_mosaic

    cases = []
    for i, hexes in enumerate(EXTRACT_EVAL_PALETTES):
        seed = seed0 + i
        cases.append(
            {
                "hexes": list(hexes),
                "seed": seed,
                "image": synthesize_mosaic(
                    hexes, size=size, seed=seed, jpeg_quality=jpeg_quality
                ),
            }
        )
    return cases


# --- G / C: paletas con armonía ----------------------------------------------


def load_generate_palettes(limit: int) -> list[dict]:
    """Paletas con rating + harmony + HEX ordenados desde la DB de entrenamiento."""
    from sqlalchemy import select

    from app.db.models import Palette, PaletteColor
    from app.db.session import training_session

    try:
        with training_session() as session:
            rows = session.execute(
                select(
                    Palette.id,
                    Palette.rating,
                    Palette.harmony_label,
                    PaletteColor.position,
                    PaletteColor.hex,
                )
                .join(PaletteColor, PaletteColor.palette_id == Palette.id)
                .order_by(Palette.id, PaletteColor.position)
            ).all()
    except Exception:
        return []
    pal_map: dict[int, dict] = {}
    for pid, rating, harmony, _pos, hexv in rows:
        if pid not in pal_map:
            pal_map[pid] = {"hexes": [], "rating": rating, "harmony_label": harmony}
        pal_map[pid]["hexes"].append(hexv)
    return list(pal_map.values())[:limit]


def load_classify_palettes(limit: int | None = None) -> list[dict]:
    """Paletas etiquetadas de 5 colores (las únicas usables por C).

    ``limit=None`` carga todas: la partición disjunta de splits necesita ver
    el conjunto completo antes de repartirlo.
    """
    from sqlalchemy import select

    from app.db.models import Palette, PaletteColor
    from app.db.session import training_session

    try:
        with training_session() as session:
            rows = session.execute(
                select(
                    Palette.id,
                    Palette.harmony_label,
                    PaletteColor.position,
                    PaletteColor.hex,
                )
                .join(PaletteColor, PaletteColor.palette_id == Palette.id)
                .where(Palette.harmony_label.in_(tuple(HARMONY_LABELS)))
                .order_by(Palette.id, PaletteColor.position)
            ).all()
    except Exception:
        return []
    pal_map: dict[int, dict] = {}
    for pid, harmony, _pos, hexv in rows:
        if pid not in pal_map:
            pal_map[pid] = {"hexes": [], "harmony_label": harmony}
        pal_map[pid]["hexes"].append(hexv)
    palettes = [
        value
        for value in pal_map.values()
        if len(value["hexes"]) == 5 and value["harmony_label"] in HARMONY_TO_IDX
    ]
    return palettes if limit is None else palettes[:limit]


def synthetic_palette(rng: np.random.RandomState, sigma: float = 2.5) -> dict:
    """Paleta sintética 5 colores + ``harmony_label`` perfecta (offline)."""
    from app.schemas.palette import HarmonyType
    from app.services.classic import ClassicPaletteGenerator
    from app.ml.color_spaces import ColorConverter

    label_to_harmony = {
        "complementario": HarmonyType.COMPLEMENTARY,
        "triada": HarmonyType.TRIADIC,
        "analogo": HarmonyType.ANALOGOUS,
        "monocromatica": HarmonyType.MONOCHROMATIC,
    }
    generator = ClassicPaletteGenerator()
    base_hex = "#%02X%02X%02X" % tuple(int(v) for v in rng.randint(0, 256, size=3))
    label = HARMONY_LABELS[int(rng.randint(0, len(HARMONY_LABELS)))]
    resp = generator.generate_palette([base_hex], label_to_harmony[label])
    hexes = [c.value.upper() for c in resp.colors]
    while len(hexes) < 5:
        idx = int(rng.randint(0, len(hexes)))
        lab = np.array(ColorConverter.hex_to_lab(hexes[idx]), dtype=np.float64)
        lab += rng.normal(0.0, sigma, size=3)
        lab[0] = float(np.clip(lab[0], 0.0, 100.0))
        hexes.append(ColorConverter.lab_to_hex((lab[0], lab[1], lab[2])).upper())
    return {"hexes": hexes[:5], "rating": None, "harmony_label": label}


def build_generate_palette_list(n: int, seed: int) -> list[dict]:
    """Paletas DB (hasta ``n``) + relleno sintético offline. Determinista."""
    palettes = load_generate_palettes(n)
    rng = np.random.RandomState(seed + 999_000)
    while len(palettes) < n:
        palettes.append(synthetic_palette(rng))
    return palettes[:n]


def _classify_class_counts(n: int) -> list[int]:
    """Reparto 80/10/10 de una clase entre ``(train, val, test)``.

    Garantiza al menos una paleta por split cuando la clase tiene al menos 3
    paletas. Con 1-2 paletas el volumen no permite cubrir los tres splits:
    se prioriza train y, con 2, también val.
    """
    if n <= 0:
        return [0, 0, 0]
    if n == 1:
        return [1, 0, 0]
    if n == 2:
        return [1, 1, 0]
    n_val = max(1, int(round(n * 0.1)))
    n_test = max(1, int(round(n * 0.1)))
    # Mantiene al menos una en train (relevante para clases de 3-4 paletas).
    while n_val + n_test >= n and n_test > 0:
        n_test -= 1
    n_train = n - n_val - n_test
    return [n_train, n_val, n_test]


def _partition_db_classify_palettes(
    palettes: list[dict],
    *,
    seed: int = CLASSIFY_DB_PARTITION_SEED,
    buckets: int = CLASSIFY_DB_BUCKETS,
) -> list[list[dict]]:
    """Partición estratificada y disjunta de las paletas de DB.

    Reparte **cada ``harmony_label`` por separado** con proporciones 80/10/10
    (``train``/``val``/``test``) usando una permutación con semilla fija. Al
    asignar cada paleta a un solo bucket, la disyunción entre splits está
    garantizada; con al menos 3 paletas por clase cada split recibe una.
    """
    rng = np.random.RandomState(seed)
    by_label: dict[str, list[dict]] = {}
    for palette in palettes:
        label = str(palette.get("harmony_label"))
        by_label.setdefault(label, []).append(palette)

    partitions: list[list[dict]] = [[] for _ in range(buckets)]
    # Orden estable por etiqueta: el resultado no depende del orden de
    # inserción de la DB.
    for label in sorted(by_label):
        class_palettes = by_label[label]
        order = rng.permutation(len(class_palettes))
        shuffled = [class_palettes[int(index)] for index in order]
        if buckets == CLASSIFY_DB_BUCKETS:
            counts = _classify_class_counts(len(shuffled))
            if len(shuffled) < buckets:
                # Clases con 1-2 paletas: van a los buckets menos cargados para
                # no desequilibrar el reparto global (sin romper la disyunción).
                loads = [len(partition) for partition in partitions]
                chosen = sorted(range(buckets), key=lambda b: (loads[b], b))[
                    : len(shuffled)
                ]
                counts = [1 if b in chosen else 0 for b in range(buckets)]
        else:  # reparto uniforme para usos con otro número de buckets
            counts = [
                len(shuffled) // buckets
                + (1 if index < len(shuffled) % buckets else 0)
                for index in range(buckets)
            ]
        start = 0
        for bucket, count in enumerate(counts):
            partitions[bucket].extend(shuffled[start : start + count])
            start += count
    return partitions


def _build_classify_split(n: int, seed: int, db_palettes: list[dict]) -> list[dict]:
    """Paletas de un split: bucket de DB + relleno sintético de esa semilla.

    El relleno usa ``seed + CLASSIFY_SYNTHETIC_SEED_OFFSET``, distinto para
    cada split, así que los sintéticos tampoco se comparten.
    """
    rng = np.random.RandomState(seed + CLASSIFY_SYNTHETIC_SEED_OFFSET)
    palettes = list(db_palettes)
    while len(palettes) < n:
        palettes.append(synthetic_palette(rng))
    rng.shuffle(palettes)
    return palettes[:n]


def _classify_db_bucket(
    seed: int, db_palettes: list[dict] | None = None
) -> list[dict]:
    """Bucket de DB asignado al split identificado por ``seed`` (``seed % 3``)."""
    partitions = _partition_db_classify_palettes(
        load_classify_palettes(None) if db_palettes is None else db_palettes
    )
    return list(partitions[seed % CLASSIFY_DB_BUCKETS])


def build_classify_palette_list(n: int, seed: int) -> list[dict]:
    """Paletas etiquetadas + relleno sintético. Determinista y sin fuga.

    Las paletas de la DB de entrenamiento se cargan una vez por llamada, se barajan
    con semilla fija y solo se usa el bucket ``seed % 3``. Con ello
    ``seed=42`` (train), ``seed+10000`` (val) y ``seed+20000`` (test) reciben
    paletas de DB disjuntas; el relleno sintético también usa una semilla
    distinta por split. Si la DB está vacía el resultado es idéntico al
    histórico (solo sintéticos con ``seed + 777000``), preservando la línea
    base de ``classify-v2``.
    """
    bucket = _classify_db_bucket(seed)
    return _build_classify_split(n, seed, bucket)


def classify_split_palettes(
    *,
    n_train: int = CLASSIFY_N_TRAIN,
    n_val: int = CLASSIFY_N_VAL,
    n_test: int = CLASSIFY_N_TEST,
    seed: int = CLASSIFY_SPLIT_SEED,
) -> dict[str, list[dict]]:
    """Paletas de train/val/test disjuntas (DB y sintéticas).

    La DB se carga **una sola vez** y se reparte con semilla fija entre los
    tres splits. Las semillas usadas al entrenar v2 (42, 10042, 20042) siguen
    produciendo exactamente el mismo set sintético que antes cuando la DB no
    tiene paletas.
    """
    db_palettes = load_classify_palettes(None)
    partitions = _partition_db_classify_palettes(db_palettes)
    seeds = {
        "train": seed,
        "val": seed + 10_000,
        "test": seed + CLASSIFY_TEST_SEED_OFFSET,
    }
    sizes = {"train": n_train, "val": n_val, "test": n_test}
    return {
        name: _build_classify_split(
            sizes[name], seeds[name], partitions[seeds[name] % CLASSIFY_DB_BUCKETS]
        )
        for name in ("train", "val", "test")
    }


def palette_signatures(palettes: list[dict]) -> set[tuple[str, ...]]:
    """Firma canónica de cada paleta (HEX en orden) para comprobar solapes."""
    return {tuple(h.upper() for h in pal["hexes"]) for pal in palettes}


# --- G: dataset de entrenamiento/eval ----------------------------------------


def palette_to_generate_dataset(palettes: list[dict], seed: int):
    """Convierte paletas ground truth a arrays de entrenamiento/eval G."""
    from app.ml.color_spaces import ColorConverter

    rng = np.random.RandomState(seed)
    n = len(palettes)
    x_lab = np.zeros((n, 10, 3), dtype=np.float32)
    x_mask = np.zeros((n, 10), dtype=np.float32)
    x_harm = np.zeros((n, 4), dtype=np.float32)
    y_pal = np.zeros((n, 15), dtype=np.float32)
    y_score = np.zeros((n, 1), dtype=np.float32)
    for i, pal in enumerate(palettes):
        hexes = [h.upper() for h in pal["hexes"][:5]]
        while len(hexes) < 5:
            hexes.append(hexes[-1])
        labs = [ColorConverter.lab_normalize(ColorConverter.hex_to_lab(h)) for h in hexes]
        y_pal[i] = np.array(labs, dtype=np.float32).reshape(-1)
        rating = pal.get("rating")
        if rating is None:
            y_score[i, 0] = 1.0
        else:
            try:
                y_score[i, 0] = float(np.clip(float(rating), 0.0, 1.0))
            except Exception:
                y_score[i, 0] = 1.0
        k = int(rng.randint(1, 4))  # 1..3 colores base (prefijo determinista)
        for j, base_hex in enumerate(hexes[:k]):
            nlab = ColorConverter.lab_normalize(ColorConverter.hex_to_lab(base_hex))
            x_lab[i, j] = np.array(nlab, dtype=np.float32)
            x_mask[i, j] = 1.0
        harmony = pal.get("harmony_label")
        if harmony in HARMONY_TO_IDX:
            x_harm[i, HARMONY_TO_IDX[harmony]] = 1.0
        else:
            x_harm[i, int(rng.randint(0, 4))] = 1.0
    return (x_lab, x_mask, x_harm), (y_pal, y_score)


def generate_eval_dataset(
    n: int = GENERATE_EVAL_N, seed: int = GENERATE_EVAL_SEED
):
    """Dataset fijo de evaluación G (40 paletas seed 99 por defecto)."""
    return palette_to_generate_dataset(build_generate_palette_list(n, seed), seed)


# --- C: features legacy + pipeline E→C ----------------------------------------


def palette_to_legacy_features(
    hexes: list[str], rng: np.random.RandomState | None = None
) -> np.ndarray:
    """Vector 40 dims (10 Lab + 10 pesos) simulando la salida de E (v1).

    Pesos descendentes fijos (0.30..0.10 + 0.02) con ruido leve, preservando
    el orden original. Réplica del builder histórico de ``classify-v1``.
    """
    from app.ml.color_spaces import ColorConverter

    labs5 = [ColorConverter.lab_normalize(ColorConverter.hex_to_lab(h)) for h in hexes]
    labs10 = []
    for i in range(10):
        if i < 5:
            labs10.append(labs5[i])
        else:
            base = np.array(labs5[-1], dtype=np.float64)
            if rng is not None:
                base = base + rng.normal(0.0, 0.02, size=3)
                base[0] = float(np.clip(base[0], 0.0, 1.0))
                base[1] = float(np.clip(base[1], -1.0, 1.0))
                base[2] = float(np.clip(base[2], -1.0, 1.0))
            labs10.append(tuple(base))
    labs10_arr = np.array(labs10, dtype=np.float32)
    w10 = np.array(
        [0.30, 0.25, 0.20, 0.15, 0.10, 0.02, 0.02, 0.02, 0.02, 0.02],
        dtype=np.float64,
    )
    if rng is not None:
        w10[:5] += rng.normal(0.0, 0.008, size=5)
        w10[5:] += rng.normal(0.0, 0.002, size=5)
        w10 = np.clip(w10, 0.01, 0.5)
    w10 = w10 / w10.sum()
    order = np.argsort(-w10, kind="stable")
    labs10_arr = labs10_arr[order]
    w10 = w10[order]
    return np.concatenate([labs10_arr.reshape(-1), w10.astype(np.float32)]).astype(
        np.float32
    )


def palette_to_legacy_dataset(palettes: list[dict], seed: int):
    """Dataset 40 dims (modo ``synthetic`` de C)."""
    rng = np.random.RandomState(seed)
    n = len(palettes)
    x = np.zeros((n, FEATURE_DIM_V1), dtype=np.float32)
    y = np.zeros((n, 4), dtype=np.float32)
    for i, pal in enumerate(palettes):
        x[i] = palette_to_legacy_features(pal["hexes"], rng)
        y[i, HARMONY_TO_IDX[pal["harmony_label"]]] = 1.0
    return x, y


def pipeline_mosaic(
    hexes: list[str], rng: np.random.RandomState
) -> tuple[bytes, np.ndarray, np.ndarray]:
    """Mosaico sintético + pesos/orden usados (simula una foto real)."""
    from app.ml.datasets import synthesize_mosaic

    weights = rng.dirichlet(np.ones(len(hexes)))
    weights = np.maximum(weights, 0.08)
    weights = weights / weights.sum()
    order = rng.permutation(len(hexes))
    ordered_hexes = [hexes[int(i)] for i in order]
    ordered_weights = [float(weights[int(i)]) for i in order]
    size = int(rng.randint(96, 161))
    image_seed = int(rng.randint(0, 2**31 - 1))
    quality = int(rng.randint(60, 96))
    mosaic = synthesize_mosaic(
        ordered_hexes,
        ordered_weights,
        size=size,
        seed=image_seed,
        jpeg_quality=quality,
    )
    return mosaic, np.asarray(ordered_hexes), np.asarray(ordered_weights)


def extract_cache_tag() -> str | None:
    """Identificador de la E resuelta: ``archivo + 8 primeros del sha256``.

    Reentrenar o activar otra E cambia la clave de caché y por tanto invalida
    automáticamente las features guardadas en ``.eval-cache``. Devuelve
    ``None`` si no hay una E resoluble.
    """
    from app.services import model_registry

    try:
        manifest = model_registry.load_manifest()
        path = model_registry.resolve("extract", manifest=manifest)
    except Exception:
        return None
    if path is None:
        return None
    sha = model_registry.sha256_file(path)
    digest = (sha or "nohash")[:8]
    return f"{path.name}-{digest}"


def split_content_hash(palettes: list[dict]) -> str:
    """Hash estable del contenido exacto de un split.

    Usa los HEX canónicos (mayúsculas) **en orden** más ``harmony_label`` de
    cada paleta. Cambiar una paleta en la DB de entrenamiento cambia el hash y, con
    él, la clave del ``.npz``: nunca se reutilizan features de otro contenido.
    """
    payload = [
        {
            "hexes": [str(hexv).upper() for hexv in palette.get("hexes", [])],
            "harmony_label": palette.get("harmony_label"),
        }
        for palette in palettes
    ]
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_cache_filename(
    split: str,
    n: int,
    seed: int,
    *,
    tag: str | None = None,
    content_hash: str | None = None,
) -> str:
    """Nombre del ``.npz`` de features.

    Incluye ``split/n/seed``, el hash del contenido del split (si se conoce) y
    la versión/hash de E. Dos splits con las mismas paletas reutilizan el
    ``.npz``; cambiar paletas o E lo invalida.
    """
    resolved_tag = tag if tag is not None else extract_cache_tag()
    content = f"-h{content_hash[:12]}" if content_hash else ""
    return (
        f"classify-v2-{split}-n{n}-s{seed}{content}"
        f"-e{resolved_tag or 'unversioned'}.npz"
    )


def extract_pipeline_features(
    palettes: list[dict],
    seed: int,
    *,
    split: str = "train",
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Ejecuta E sobre mosaicos y devuelve ``(X54, X40, y, stats)``.

    X54/X40 se calculan de los MISMOS colores extraídos por E, así la
    comparación v1 vs v2 en test usa exactamente el mismo set de entrada.
    Cachea por ``(split, n, seed, hash del contenido, versión/hash de E)`` en
    ``cache_dir`` (gitignored): reentrenar E o cambiar las paletas del split
    invalida la caché.
    """
    import app.services.inference as inference
    from app.ml.color_features import palette_features, palette_features_legacy

    tag = extract_cache_tag()
    if tag is None:
        raise RuntimeError("No hay red E activa: el modo pipeline requiere extract")

    n = len(palettes)
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / extract_cache_filename(
            split, n, seed, tag=tag, content_hash=split_content_hash(palettes)
        )
        if cache_path.exists():
            data = np.load(cache_path)
            return (
                data["X54"].astype(np.float32),
                data["X40"].astype(np.float32),
                data["y"].astype(np.float32),
                {"cache_hit": True, "fallback_extractions": int(data["fallback"])},
            )

    x54 = np.zeros((n, FEATURE_DIM_V2), dtype=np.float32)
    x40 = np.zeros((n, FEATURE_DIM_V1), dtype=np.float32)
    y = np.zeros((n, 4), dtype=np.float32)
    fallbacks = 0
    for i, palette in enumerate(palettes):
        sample_rng = np.random.RandomState((seed * 1_000_003 + i * 7919) % (2**32 - 1))
        colors = None
        response = None
        for attempt in range(3):
            mosaic, _hexes, _weights = pipeline_mosaic(palette["hexes"], sample_rng)
            response = inference.extract_image_colors(mosaic)
            if not response.fallback:
                colors = response.colors
                break
            fallbacks += 1
            print(
                f"[benchmarks] aviso: E cayó al clásico en {split}[{i}] "
                f"(intento {attempt + 1}/3)",
                flush=True,
            )
        if colors is None:
            # Se conserva el comportamiento real de producción (fallback).
            colors = response.colors if response is not None else []
        x54[i] = palette_features(colors, input_dim=FEATURE_DIM_V2)
        x40[i] = palette_features_legacy(colors)
        y[i, HARMONY_TO_IDX[palette["harmony_label"]]] = 1.0

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path, X54=x54, X40=x40, y=y, fallback=np.asarray(fallbacks)
        )
    return x54, x40, y, {"cache_hit": False, "fallback_extractions": int(fallbacks)}


def classify_eval_split(
    *,
    n: int = CLASSIFY_N_TEST,
    seed: int = CLASSIFY_SPLIT_SEED,
    test_seed: int | None = None,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
):
    """Split de test de C (80 paletas, seed 20042) ya proyectado por E.

    ``test_seed`` explícito manda (``eval/thresholds.json``); si se omite se
    deriva de ``seed + 20_000``, el mismo split usado al entrenar v2.
    Devuelve ``(X54, X40, y, palettes, stats)``.

    El split de test usa **siempre** el bucket estratificado de test (índice
    2), nunca ``test_seed % 3``: así cualquier ``test_seed`` arbitrario sigue
    siendo disjunto de las paletas de DB de train (bucket 0) y val (bucket 1).
    """
    if test_seed is None:
        test_seed = seed + CLASSIFY_TEST_SEED_OFFSET
    test_seed = int(test_seed)
    partitions = _partition_db_classify_palettes(load_classify_palettes(None))
    palettes = _build_classify_split(n, test_seed, partitions[2])
    x54, x40, y, stats = extract_pipeline_features(
        palettes, test_seed, split="test", cache_dir=cache_dir
    )
    return x54, x40, y, palettes, stats


__all__ = [
    "CLASSIFY_DB_BUCKETS",
    "CLASSIFY_DB_PARTITION_SEED",
    "CLASSIFY_N_TEST",
    "CLASSIFY_N_TRAIN",
    "CLASSIFY_N_VAL",
    "CLASSIFY_SPLIT_SEED",
    "CLASSIFY_SYNTHETIC_SEED_OFFSET",
    "CLASSIFY_TEST_SEED_OFFSET",
    "DEFAULT_CACHE_DIR",
    "EXTRACT_EVAL_PALETTES",
    "EXTRACT_EVAL_SEED0",
    "EXTRACT_EVAL_SIZE",
    "GENERATE_EVAL_N",
    "GENERATE_EVAL_SEED",
    "HARMONY_LABELS",
    "HARMONY_TO_IDX",
    "build_classify_palette_list",
    "build_extract_palette_list",
    "build_generate_palette_list",
    "classify_eval_split",
    "classify_split_palettes",
    "extract_cache_filename",
    "extract_cache_tag",
    "extract_eval_cases",
    "extract_pipeline_features",
    "generate_eval_dataset",
    "git_sha",
    "load_classify_palettes",
    "load_extract_palettes",
    "load_generate_palettes",
    "palette_signatures",
    "palette_to_generate_dataset",
    "palette_to_legacy_dataset",
    "palette_to_legacy_features",
    "pipeline_mosaic",
    "random_palette",
    "split_content_hash",
    "synthetic_palette",
    "synthesize_extract_dataset",
]

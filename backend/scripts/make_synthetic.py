"""Dataset sintético por destilación del motor clásico (100% offline).

Genera pares (base, armonía) → paleta con `ClassicPaletteGenerator` y ruido
gaussiano en Lab; la etiqueta `harmony_label` es perfecta por construcción
(día-0 para la clasificadora C y bootstrap de la generadora G).

Cada paleta tiene 5 colores: los 2-3 del motor clásico (exactos) más
variantes con ruido Lab hasta completar 5. Idempotente: con la misma semilla
se genera la misma secuencia y el dedup por sha256 (`app.ml.datasets`)
omite las ya guardadas. Si las fuentes públicas fallan, este script cubre el
volumen (por defecto 10.000 paletas).

Uso:
    backend/.venv/bin/python backend/scripts/make_synthetic.py --n 10000
    TRAINING_DATABASE_URL=postgresql+psycopg://.../t \
        backend/.venv/bin/python backend/scripts/make_synthetic.py --n 200 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.config import training_database_url  # noqa: E402
from app.db.engine import sanitized_url  # noqa: E402
from app.db.session import training_session  # noqa: E402
from app.ml.color_spaces import ColorConverter  # noqa: E402
from app.ml.datasets import (  # noqa: E402
    VALID_HARMONY_LABELS,
    count_palettes,
    count_palettes_by_source,
    insert_palette,
    load_palette_keys,
)
from app.schemas.palette import HarmonyType  # noqa: E402
from app.services.classic import ClassicPaletteGenerator  # noqa: E402

SOURCE = "synthetic"
PALETTE_SIZE = 5
BATCH_SIZE = 500

LABEL_TO_HARMONY: dict[str, HarmonyType] = {
    "complementario": HarmonyType.COMPLEMENTARY,
    "triada": HarmonyType.TRIADIC,
    "analogo": HarmonyType.ANALOGOUS,
    "monocromatica": HarmonyType.MONOCHROMATIC,
}
LABELS = sorted(VALID_HARMONY_LABELS)


def _noisy_variant(hexv: str, rng: np.random.RandomState, sigma: float) -> str:
    lab = np.array(ColorConverter.hex_to_lab(hexv), dtype=np.float64)
    lab += rng.normal(0.0, sigma, size=3)
    lab[0] = float(np.clip(lab[0], 0.0, 100.0))
    return ColorConverter.lab_to_hex((lab[0], lab[1], lab[2])).upper()


def generate_palette(
    rng: np.random.RandomState, generator: ClassicPaletteGenerator, sigma: float
) -> tuple[list[str], str]:
    """Una paleta sintética de 5 colores + su etiqueta perfecta."""
    base_hex = "#%02X%02X%02X" % tuple(int(v) for v in rng.randint(0, 256, size=3))
    label = LABELS[int(rng.randint(0, len(LABELS)))]
    response = generator.generate_palette([base_hex], LABEL_TO_HARMONY[label])
    hexes = [c.value.upper() for c in response.colors]
    # Relleno con ruido Lab hasta 5 colores (round-robin sobre los existentes).
    i = 0
    while len(hexes) < PALETTE_SIZE:
        hexes.append(_noisy_variant(hexes[i % len(hexes)], rng, sigma))
        i += 1
    return hexes[:PALETTE_SIZE], label


def run(n: int = 10_000, seed: int = 42, sigma: float = 2.5) -> dict:
    """Genera `n` paletas en la DB de entrenamiento; devuelve el reporte."""
    rng = np.random.RandomState(seed)
    generator = ClassicPaletteGenerator()
    with training_session() as session:
        existing = load_palette_keys(session)
        inserted = 0
        skipped = 0
        for _ in range(n):
            hexes, label = generate_palette(rng, generator, sigma)
            pid = insert_palette(
                session,
                source=SOURCE,
                hexes=hexes,
                rating=None,
                harmony_label=label,
                existing_keys=existing,
            )
            if pid is None:
                skipped += 1
            else:
                inserted += 1
                if inserted % BATCH_SIZE == 0:
                    session.commit()
        session.commit()
        by_source = count_palettes_by_source(session)
        total = count_palettes(session)
    return {
        "db": sanitized_url(training_database_url()),
        "requested": n,
        "seed": seed,
        "inserted": inserted,
        "skipped_duplicates": skipped,
        "by_source": by_source,
        "total_palettes": total,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10_000, help="Nº de paletas a generar")
    parser.add_argument("--seed", type=int, default=42, help="Semilla (reproducible)")
    parser.add_argument(
        "--noise", type=float, default=2.5, help="Sigma del ruido gaussiano en Lab"
    )
    args = parser.parse_args(argv)
    if args.n <= 0:
        raise SystemExit("--n debe ser > 0")
    report = run(n=args.n, seed=args.seed, sigma=args.noise)
    print(
        f"[make_synthetic] db={report['db']} requested={report['requested']} "
        f"seed={report['seed']} inserted={report['inserted']} "
        f"skipped_duplicates={report['skipped_duplicates']} "
        f"total={report['total_palettes']} by_source={report['by_source']}"
    )
    return report


if __name__ == "__main__":
    main()

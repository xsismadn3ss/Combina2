# Tests Fase 2 (issue #18): sintético + dedup idempotente + split reproducible.
import io
import re

import pytest
from PIL import Image
from sqlalchemy import select

from app.db.models import Palette, PaletteColor
from app.db.session import training_session
from app.ml.datasets import (
    VALID_HARMONY_LABELS,
    palette_splits,
    preprocess_image,
)
from scripts.make_synthetic import run as synthetic_run

HEX_RE = re.compile(r"^#[0-9A-F]{6}$")


def _palettes():
    with training_session() as session:
        return list(session.execute(select(Palette).order_by(Palette.id)).scalars())


def test_synthetic_generates_valid_labeled_palettes(_clean_database):
    report = synthetic_run(n=200, seed=7)
    assert report["inserted"] == 200
    assert report["total_palettes"] == 200
    palettes = _palettes()
    assert len(palettes) == 200
    assert {row.source for row in palettes} == {"synthetic"}
    assert {row.harmony_label for row in palettes} == VALID_HARMONY_LABELS
    with training_session() as session:
        for row in palettes:
            colors = list(
                session.execute(
                    select(PaletteColor)
                    .where(PaletteColor.palette_id == row.id)
                    .order_by(PaletteColor.position)
                ).scalars()
            )
            assert len(colors) == 5
            assert all(HEX_RE.match(color.hex) for color in colors)


def test_synthetic_rerun_is_idempotent(_clean_database):
    first = synthetic_run(n=200, seed=7)
    second = synthetic_run(n=200, seed=7)
    assert first["inserted"] == 200
    assert second["inserted"] == 0
    assert second["skipped_duplicates"] == 200
    assert second["total_palettes"] == 200


def test_split_sums_100_percent_and_is_reproducible(_clean_database):
    synthetic_run(n=400, seed=11)
    with training_session() as session:
        total = len(list(session.execute(select(Palette.id)).scalars()))
        assert total == 400
        first = palette_splits(session)
        second = palette_splits(session)
    assert first == second  # reproducible
    sizes = {k: len(v) for k, v in first.items()}
    assert sum(sizes.values()) == total  # suma el 100%
    assert len(set(first["train"]) & set(first["val"]) & set(first["test"])) == 0
    assert len(set(first["train"]) | set(first["val"]) | set(first["test"])) == total
    # Proporciones aproximadas 80/10/10 (tolerancia por redondeo de estratos).
    assert sizes["train"] / total == pytest.approx(0.8, abs=0.05)
    assert sizes["val"] / total == pytest.approx(0.1, abs=0.05)
    assert sizes["test"] / total == pytest.approx(0.1, abs=0.05)


def test_preprocess_image_common_contract():
    img = Image.new("RGB", (640, 100), (10, 200, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out = preprocess_image(buf.getvalue())
    assert max(out["width"], out["height"]) == 300  # lado mayor 300px
    assert out["pixels_lab"].shape == (3000, 3)  # submuestreo 3000 px
    assert 0.0 <= float(out["pixels_lab"][:, 0].min()) <= 1.0
    assert len(out["sha256"]) == 64

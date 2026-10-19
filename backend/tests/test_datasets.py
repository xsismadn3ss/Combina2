# Tests Fase 2 (issue #18): sintético + dedup idempotente + split reproducible.
import io
import re

import pytest
from PIL import Image

from app.db.sqlite import get_connection
from app.ml.datasets import (
    VALID_HARMONY_LABELS,
    palette_splits,
    preprocess_image,
)
from scripts.make_synthetic import run as synthetic_run

HEX_RE = re.compile(r"^#[0-9A-F]{6}$")


@pytest.fixture()
def training_db(tmp_path, monkeypatch):
    db = tmp_path / "phase2.db"
    monkeypatch.setenv("TRAINING_DB", str(db))
    return db


def _palettes(conn):
    return conn.execute(
        "SELECT id, source, harmony_label FROM palettes"
    ).fetchall()


def test_synthetic_generates_valid_labeled_palettes(training_db):
    report = synthetic_run(n=200, seed=7)
    assert report["inserted"] == 200
    assert report["total_palettes"] == 200
    conn = get_connection()
    try:
        rows = _palettes(conn)
        assert len(rows) == 200
        assert {r["source"] for r in rows} == {"synthetic"}
        assert set(r["harmony_label"] for r in rows) <= VALID_HARMONY_LABELS
        assert set(r["harmony_label"] for r in rows) == VALID_HARMONY_LABELS
        for row in rows:
            colors = conn.execute(
                "SELECT hex, l, a, b FROM palette_colors WHERE palette_id=? ORDER BY position",
                (row["id"],),
            ).fetchall()
            assert len(colors) == 5
            assert all(HEX_RE.match(c["hex"]) for c in colors)
    finally:
        conn.close()


def test_synthetic_rerun_is_idempotent(training_db):
    first = synthetic_run(n=200, seed=7)
    second = synthetic_run(n=200, seed=7)
    assert first["inserted"] == 200
    assert second["inserted"] == 0
    assert second["skipped_duplicates"] == 200
    assert second["total_palettes"] == 200


def test_split_sums_100_percent_and_is_reproducible(training_db):
    synthetic_run(n=400, seed=11)
    conn = get_connection()
    try:
        total = int(conn.execute("SELECT COUNT(*) FROM palettes").fetchone()[0])
        assert total == 400
        first = palette_splits(conn)
        second = palette_splits(conn)
    finally:
        conn.close()
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

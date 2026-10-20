# Tests Fase 3 (issue #19): red E extractora + POST /extract.
#
# Sin TF (venv runtime 3.14) corren todos menos la paridad, que usa el
# `.keras` si existe y si no hace `pytest.skip`. Nada en este módulo importa
# tensorflow/sklearn a nivel superior.
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.services.inference as inference
from app.config import APICOnfig
from app.main import app
from app.ml import benchmarks
from app.ml.color_spaces import ColorConverter
from app.ml.losses import (
    distillation_loss_numpy,
    mean_delta_e,
    reorder_matched,
)

client = TestClient(app)
HEX_RE = re.compile(r"^#[0-9A-F]{6}$")

THRESHOLDS = json.loads(
    (Path(__file__).resolve().parents[1] / "eval" / "thresholds.json").read_text()
)


def _png_bytes(hexes: list[str], size: int = 90) -> bytes:
    """Mosaico PNG determinista de franjas (para el endpoint)."""
    strip = size // len(hexes)
    img = Image.new("RGB", (strip * len(hexes), 30), (0, 0, 0))
    px = img.load()
    assert px is not None
    for i, h in enumerate(hexes):
        rgb = ColorConverter.hex_to_rgb(h)
        for x in range(i * strip, (i + 1) * strip):
            for y in range(30):
                px[x, y] = rgb
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- losses.py (puro numpy, corre en ambos venvs) ------------------------------


def test_mean_delta_e_identical_is_zero():
    pal = np.array([[50.0, 10.0, -20.0], [80.0, 0.0, 0.0], [20.0, -30.0, 40.0]])
    assert mean_delta_e(pal, pal) == pytest.approx(0.0)


def test_mean_delta_e_black_vs_white_is_100():
    black = np.array([[0.0, 0.0, 0.0]])
    white = np.array([[100.0, 0.0, 0.0]])
    assert mean_delta_e(black, white) == pytest.approx(100.0)


def test_mean_delta_e_matching_is_order_invariant():
    a = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [50.0, 50.0, 50.0]])
    b = a[::-1].copy()
    assert mean_delta_e(a, b) == pytest.approx(0.0)


def test_distillation_loss_zero_on_perfect_match():
    rng = np.random.RandomState(0)
    c = rng.rand(10, 3).astype(np.float64)
    w = np.full(10, 0.1)
    assert distillation_loss_numpy(c, w, c, w, entropy_lambda=0.0) == pytest.approx(0.0)


def test_distillation_loss_entropy_term_sharpens():
    c = np.zeros((10, 3))
    uniform = np.full(10, 0.1)
    sharp = np.zeros(10)
    sharp[0] = 1.0
    lam = 0.01
    l_uniform = distillation_loss_numpy(c, uniform, c, uniform, entropy_lambda=lam)
    l_sharp = distillation_loss_numpy(c, sharp, c, sharp, entropy_lambda=lam)
    assert l_uniform == pytest.approx(lam * np.log(10))
    assert l_sharp < l_uniform  # bins nítidos = menor entropía


def test_reorder_matched_aligns_target():
    pred_c = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    pred_w = np.array([0.7, 0.3])
    tgt_c = np.array([[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    tgt_w = np.array([0.3, 0.7])
    mc, mw = reorder_matched(pred_c, pred_w, tgt_c, tgt_w)
    assert np.allclose(mc, pred_c)
    assert np.allclose(mw, pred_w)


# --- Endpoint /extract (fallback o red, ambos cumplen el contrato) -------------


def test_extract_endpoint_weighted_colors_sum_one():
    response = client.post(
        f"{APICOnfig.prefix}/palette/extract",
        files={"file": ("mosaic.png", _png_bytes(["#FF0000", "#00FF00", "#0000FF"]), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert 1 <= len(data["colors"]) <= 10
    assert all(c["role"] == "predominant" for c in data["colors"])
    assert all(HEX_RE.match(c["value"]) for c in data["colors"])
    total = sum(c["weight"] for c in data["colors"])
    assert total == pytest.approx(1.0)


def test_fallback_when_no_active_model(monkeypatch):
    monkeypatch.setattr(inference, "load_manifest", lambda: {})
    monkeypatch.delitem(inference._keras_cache, "extract", raising=False)
    monkeypatch.setattr(inference, "_identifier", None)
    response = client.post(
        f"{APICOnfig.prefix}/palette/extract",
        files={"file": ("red.png", _png_bytes(["#FF0000"]), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["fallback"] is True
    assert data["colors"][0]["value"] == "#FF0000"


# --- Path con red: sin sklearn -------------------------------------------------


class _FakeExtractModel:
    """Stub de la red E: devuelve centroides+pesos fijos (duck-typing)."""

    def __init__(self, hexes: list[str], weights: list[float]):
        self._c = np.array(
            [ColorConverter.lab_normalize(ColorConverter.hex_to_lab(h)) for h in hexes]
        )[None, ...]
        w = np.asarray(weights, dtype=np.float64)
        self._w = (w / w.sum())[None, ...]

    def predict(self, x, verbose=0):
        assert x.shape[1:] == (3000, 3)
        return [self._c, self._w]


def _purge_from_modules(monkeypatch):
    doomed = [
        m
        for m in list(sys.modules)
        if m == "sklearn" or m.startswith("sklearn.") or m == "app.services.classic"
    ]
    for m in doomed:
        monkeypatch.delitem(sys.modules, m, raising=False)
    return doomed


def test_model_path_returns_sorted_colors_without_sklearn(monkeypatch):
    hexes = ["#FF0000", "#00FF00", "#0000FF"]
    weights = [0.5, 0.3, 0.2]
    monkeypatch.setattr(
        inference, "load_keras_model", lambda slot: _FakeExtractModel(hexes, weights)
    )
    _purge_from_modules(monkeypatch)
    colors = inference.extract_image_colors(_png_bytes(hexes)).colors
    assert [c.value for c in colors] == hexes  # ordenadas por peso
    assert [c.weight for c in colors] == pytest.approx(weights)
    assert all(c.role == "predominant" for c in colors)
    assert "sklearn" not in sys.modules
    assert "app.services.classic" not in sys.modules
    assert not any(m.startswith("sklearn.") for m in sys.modules)


def test_model_failure_falls_back(monkeypatch):
    class _Broken:
        def predict(self, x, verbose=0):
            raise RuntimeError("GPU en llamas")

    monkeypatch.setattr(inference, "load_keras_model", lambda slot: _Broken())
    monkeypatch.setattr(inference, "_identifier", None)
    resp = inference.extract_image_colors(_png_bytes(["#FF0000"]))
    assert resp.fallback is True
    assert resp.colors[0].value == "#FF0000"


# --- Paridad ΔE < 5 vs KMeans (requiere TF + .keras; si no, skip) --------------


def test_parity_delta_e_vs_kmeans():
    pytest.importorskip("tensorflow")
    keras_path = inference.MODELS_DIR / "extract-v1.keras"
    if not keras_path.exists():
        pytest.skip("sin backend/models/extract-v1.keras entrenado")
    from app.services.classic import ClassicPaletteIdentifier

    classic = ClassicPaletteIdentifier()
    cases = benchmarks.extract_eval_cases(
        size=benchmarks.EXTRACT_EVAL_SIZE,
        jpeg_quality=benchmarks.EXTRACT_EVAL_JPEG_QUALITY,
        seed0=benchmarks.EXTRACT_EVAL_SEED0,
    )
    deltas = []
    for case in cases:
        ref_lab = np.array(
            [ColorConverter.hex_to_lab(c.value) for c in classic.identify_palette(case["image"])]
        )
        result = inference.extract_image_colors(case["image"])
        assert not result.fallback
        got_lab = np.array([ColorConverter.hex_to_lab(c.value) for c in result.colors])
        deltas.append(mean_delta_e(got_lab, ref_lab))
    mean_de = float(np.mean(deltas))
    max_de = float(np.max(deltas))
    print(f"\n[parity] ΔE medio={mean_de:.3f} max={max_de:.3f}")
    cfg = THRESHOLDS["extract"]
    assert mean_de <= cfg["delta_e76_mean_max"]
    assert max_de <= cfg["delta_e76_max_max"]

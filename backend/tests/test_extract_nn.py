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


# --- sample_pixels estratificado (opción A, sin tocar modelos) ------------------


def test_sample_pixels_shape_and_determinism():
    from app.ml.datasets import SAMPLE_PIXELS, sample_pixels

    rng = np.random.RandomState(0)
    rgb = rng.randint(0, 256, size=(90, 60, 3)).astype(np.uint8)
    a = sample_pixels(rgb)
    b = sample_pixels(rgb)
    assert a.shape == (SAMPLE_PIXELS, 3) == (3000, 3)
    assert np.array_equal(a, b)  # misma entrada → misma salida
    c = sample_pixels(rgb, seed=43)
    assert not np.array_equal(a, c)  # el seed discrimina


def test_sample_pixels_preserves_small_vivid_block():
    from app.ml.datasets import sample_pixels

    green = np.array([0, 255, 0], dtype=np.uint8)
    red = np.array([255, 0, 0], dtype=np.uint8)
    rgb = np.full((100, 100, 3), green, dtype=np.uint8)
    rgb[0:10, 0:10] = red  # bloque rojo del 1% del área
    sampled = sample_pixels(rgb, n=3000, seed=42)
    assert sampled.shape == (3000, 3)
    n_red = int(np.all(sampled == red, axis=1).sum())
    # Cobertura proporcional al área: ~1% de 3000 ≈ 30 (tolerancia amplia).
    assert n_red >= 10, f"el bloque rojo minoritario se perdió: {n_red}/3000"


def test_sample_pixels_tiny_image_with_replacement():
    from app.ml.datasets import sample_pixels

    rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)  # 1×2 px
    sampled = sample_pixels(rgb, n=3000, seed=42)
    assert sampled.shape == (3000, 3)
    assert np.array_equal(sample_pixels(rgb, n=3000, seed=42), sampled)
    assert set(map(tuple, sampled.tolist())) <= {(1, 2, 3), (4, 5, 6)}


# --- Rescate de minorías (post-proceso E: fusión ΔE<5 + rescate 1%) --------

def _lab_norm(hexv: str) -> list[float]:
    lab = ColorConverter.hex_to_lab(hexv)
    n = ColorConverter.lab_normalize(lab)
    return [float(n[0]), float(n[1]), float(n[2])]


def _delta_e_hex(a: str, b: str) -> float:
    la = np.asarray(ColorConverter.hex_to_lab(a), dtype=np.float64)
    lb = np.asarray(ColorConverter.hex_to_lab(b), dtype=np.float64)
    return float(np.linalg.norm(la - lb))


def _closest_delta(target: str, colors) -> float:
    return min(_delta_e_hex(target, c.value) for c in colors)


def test_merge_near_duplicate_centroids_sums_weights():
    blue_a = np.asarray(_lab_norm("#0000FF"), dtype=np.float64)
    # Perturbación pequeña en Lab crudo (≈ΔE 2) → mismo grupo tras fusionar.
    blue_b_raw = np.asarray(ColorConverter.hex_to_lab("#0000FF"), dtype=np.float64)
    blue_b_raw = blue_b_raw + np.array([1.0, 1.0, 0.5])
    blue_b = ColorConverter.lab_normalize(tuple(blue_b_raw))
    green = _lab_norm("#00FF00")
    centroids = np.array([blue_a, list(blue_b), green])
    weights = np.array([0.5, 0.3, 0.2])
    colors = inference.extract_colors_from_outputs(centroids, weights)
    assert len(colors) == 2  # los dos azules fusionados
    assert sum(c.weight for c in colors) == pytest.approx(1.0)
    assert colors[0].weight == pytest.approx(0.8)
    assert [c.weight for c in colors] == sorted(
        [c.weight for c in colors], reverse=True
    )


def test_rescue_minority_red_block_1pct():
    from app.ml.datasets import synthesize_mosaic

    # Mosaico limpio (ruido/luz desactivados): el bloque rojo del 1%
    # conserva un núcleo compacto; la red simulada lo absorbe igual que
    # el HardKMeans real (10 centroides azules fragmentados, sin rojo).
    mosaic = synthesize_mosaic(
        ["#0000FF", "#FF0000"],
        weights=[0.99, 0.01],
        seed=7,
        size=256,
        noise_sigma=0.0,
        light_range=(1.0, 1.0),
        jpeg_quality=95,
    )
    from app.ml.datasets import preprocess_image

    samples = np.asarray(preprocess_image(mosaic)["pixels_lab"], dtype=np.float64)
    # Red E simulada con el fallo: 10 centroides azules fragmentados, sin rojo.
    rng = np.random.RandomState(11)
    blue = np.asarray(_lab_norm("#0000FF"), dtype=np.float64)
    centroids = np.clip(blue + rng.normal(0.0, 0.004, size=(10, 3)), -1.0, 1.0)
    weights = np.full(10, 0.1)
    colors = inference.extract_colors_from_outputs(
        centroids, weights, samples_lab=samples
    )
    assert 1 <= len(colors) <= 10
    assert sum(c.weight for c in colors) == pytest.approx(1.0)
    assert _closest_delta("#FF0000", colors) < 20.0, (
        f"rojo 1% no rescatado: {[c.value for c in colors]}"
    )


def test_rescue_end_to_end_through_extract_with_model():
    from app.ml.datasets import synthesize_mosaic

    mosaic = synthesize_mosaic(
        ["#0000FF", "#FF0000"],
        weights=[0.99, 0.01],
        seed=7,
        size=256,
        noise_sigma=0.0,
        light_range=(1.0, 1.0),
        jpeg_quality=95,
    )
    rng = np.random.RandomState(11)
    blue = np.asarray(_lab_norm("#0000FF"), dtype=np.float64)
    centroids = np.clip(blue + rng.normal(0.0, 0.004, size=(10, 3)), -1.0, 1.0)

    class _AllBlueModel:
        def predict(self, x, verbose=0):
            assert x.shape[1:] == (3000, 3)
            return [centroids[None, ...], np.full((1, 10), 0.1)]

    colors = inference._extract_with_model(_AllBlueModel(), mosaic)
    assert colors is not None
    assert _closest_delta("#FF0000", colors) < 20.0


def test_rescue_fragmented_crumbs_real_kmeans_path():
    """El rojo fragmentado en migajas < EXTRACT_MIN_WEIGHT debe rescatarse.

    Repro del punto ciego: mosaico real (azul `#2D3E5F` + franja roja 1%)
    por el path numpy real (`preprocess_image` → `_batch_kmeans_lab_to_rgb`
    → `extract_colors_from_outputs`). El KMeans real fragmenta el rojo en
    migajas rojizas bajo el umbral que, evaluadas contra TODOS los
    mergeados, ocultaban la minoría al rescate (las muestras rojas quedan
    a d≈0 de su migaja). El rescate debe evaluarse contra los centroides
    KEPT (post-filtro): el rojo (masa total ~1%) reaparece en el top-10.
    Sin TF (sin `_batch_kmeans_lab_to_rgb` importable) se replica el mismo
    escenario con los centroides fragmentados de la evidencia
    (azul w=0.982 + 3 migajas rojizas + 2 oscuras, todas < umbral).
    """
    from app.ml.datasets import preprocess_image, synthesize_mosaic

    mosaic = synthesize_mosaic(
        ["#2D3E5F", "#FF0000"],
        weights=[0.99, 0.01],
        seed=7,
        size=256,
        noise_sigma=0.0,
        light_range=(1.0, 1.0),
        jpeg_quality=95,
    )
    samples = np.asarray(preprocess_image(mosaic)["pixels_lab"], dtype=np.float64)

    try:
        from app.ml.extract_model import _batch_kmeans_lab_to_rgb

        x = np.asarray(samples, dtype=np.float64).reshape(1, -1, 3)
        centroids, weights, _ = _batch_kmeans_lab_to_rgb(x)
        centroids, weights = np.asarray(centroids[0]), np.asarray(weights[0])
    except Exception:
        # Sin TF: réplica fiel del escenario del contenedor. Las migajas
        # salen del KMeans sobre estas mismas muestras, luego coinciden
        # con ellas por construcción: se derivan de la media roja real
        # (offsets ΔE=7: no fusionan entre sí, pero toda muestra roja
        # queda a d<15 de alguna migaja → la lógica vieja no veía "far").
        s_raw = np.asarray(samples * [100.0, 128.0, 128.0], dtype=np.float64)
        pure_red = np.array([53.2, 80.1, 67.2])
        core = s_raw[np.linalg.norm(s_raw - pure_red, axis=1) < 15.0]
        assert len(core) >= 15, "el mosaico no dejó núcleo rojo rescatable"
        m = core.mean(axis=0)
        labs_raw = np.vstack(
            [
                np.asarray(ColorConverter.hex_to_lab("#2D3E5F"), dtype=np.float64),
                m,
                m + np.array([7.0, 0.0, 0.0]),
                m + np.array([0.0, 7.0, 0.0]),
                np.array([10.0, 5.0, 5.0]),
                np.array([15.0, -5.0, -10.0]),
            ]
        )
        centroids = np.array(
            [ColorConverter.lab_normalize(tuple(v)) for v in labs_raw],
            dtype=np.float64,
        )
        weights = np.array([0.982, 0.0049, 0.004, 0.004, 0.003, 0.0021])
        weights = weights / weights.sum()
        # La lógica vieja (rescate vs TODOS los mergeados) es ciega aquí.
        assert (
            inference._rescue_minority_color(s_raw, labs_raw) is None
        ), "el escenario ya no reproduce el punto ciego (migajas d≈0)"
        # Y ninguna migaja supera el umbral: solo el azul hace keep.
        assert sum(1 for w in weights if w >= inference.EXTRACT_MIN_WEIGHT) == 1

    colors = inference.extract_colors_from_outputs(
        centroids, weights, samples_lab=samples
    )
    assert 1 <= len(colors) <= 10
    assert sum(c.weight for c in colors) == pytest.approx(1.0)
    assert _closest_delta("#FF0000", colors) < 20.0, (
        f"rojo 1% fragmentado no rescatado: {[c.value for c in colors]}"
    )


def test_rescue_skips_tiny_outlier_seed_for_big_minority():
    """Un outlier diminuto aún más lejano no debe robar el único rescate.

    Geometría del caso real (`backend/prubea-anime.jpeg`): minoría grande
    (camiseta roja, ~2%) + grupo diminuto aún más lejano (píxel de neón,
    0.1%) que gana la primera semilla. Con una sola semilla el núcleo del
    neón no alcanza RESCUE_MIN_MASS → None y la camiseta se pierde; con
    varias semillas se descarta el núcleo del neón y se rescata el rojo.
    """
    rng = np.random.RandomState(0)
    gray = np.array([50.0, 0.0, 0.0])
    red = np.array([32.0, 42.0, 30.0])
    neon = np.array([90.0, 100.0, -80.0])
    samples = np.vstack(
        [
            gray + rng.normal(0.0, 1.0, size=(2937, 3)),
            red + rng.normal(0.0, 1.0, size=(60, 3)),
            neon + rng.normal(0.0, 0.5, size=(3, 3)),
        ]
    )
    centroids = gray.reshape(1, 3)
    d = np.linalg.norm(samples - gray, axis=1)
    first_seed = int(np.argmax(d))
    # El outlier gana la primera semilla (es lo más lejano)...
    assert first_seed >= 2997, "el outlier debe ser lo más lejano"
    # ...pero su núcleo no alcanza masa (con una sola semilla → None).
    neon_core_mass = float((np.linalg.norm(samples - samples[first_seed], axis=1) <= 12.0).sum()) / len(samples)
    assert neon_core_mass < inference.RESCUE_MIN_MASS
    # Con el fix multi-semilla se rescata el rojo.
    rescued = inference._rescue_minority_color(samples, centroids)
    assert rescued is not None, "el rojo (minoría grande) debe rescatarse"
    mean, mass = rescued
    assert mass == pytest.approx(0.02, abs=0.005)
    assert float(np.linalg.norm(mean - red)) < 12.0
    # Determinista: segunda llamada idéntica.
    again = inference._rescue_minority_color(samples, centroids)
    assert again is not None
    assert np.allclose(again[0], mean) and again[1] == mass


def test_rescue_real_anime_shirt_red():
    """La camiseta roja de `backend/prubea-anime.jpeg` reaparece en el top-10.

    Path numpy real (`preprocess_image` → `_batch_kmeans_lab_to_rgb` →
    `extract_colors_from_outputs`): el KMeans real deja 9 centroides
    azules/grises y el rescate multi-semilla debe traer el rojo de la
    camiseta (familia `#8D261F`) al top-10. Sin TF (sin kmeans) hace skip,
    como el test de paridad.
    """
    pytest.importorskip("tensorflow")
    try:
        from app.ml.extract_model import _batch_kmeans_lab_to_rgb
    except Exception:
        pytest.skip("sin _batch_kmeans_lab_to_rgb (requiere TF/keras)")
    from app.ml.datasets import preprocess_image

    img_path = Path(__file__).resolve().parents[1] / "prubea-anime.jpeg"
    if not img_path.exists():
        pytest.skip("sin backend/prubea-anime.jpeg")
    samples = np.asarray(preprocess_image(img_path.read_bytes())["pixels_lab"])
    x = np.asarray(samples, dtype=np.float64).reshape(1, -1, 3)
    centroids, weights, _ = _batch_kmeans_lab_to_rgb(x)
    colors = inference.extract_colors_from_outputs(
        np.asarray(centroids[0]), np.asarray(weights[0]), samples_lab=samples
    )
    assert 1 <= len(colors) <= 10
    assert sum(c.weight for c in colors) == pytest.approx(1.0)
    assert _closest_delta("#8D261F", colors) < 20.0, (
        f"camiseta roja no rescatada: {[c.value for c in colors]}"
    )
    # Determinista: el rescate no depende del orden de las muestras.
    colors2 = inference.extract_colors_from_outputs(
        np.asarray(centroids[0]), np.asarray(weights[0]), samples_lab=samples
    )
    assert [c.value for c in colors2] == [c.value for c in colors]


def test_rescue_ignores_uniform_image_without_minority():
    blue = np.asarray(_lab_norm("#0000FF"), dtype=np.float64)
    rng = np.random.RandomState(3)
    centroids = np.clip(blue + rng.normal(0.0, 0.004, size=(10, 3)), -1.0, 1.0)
    weights = np.full(10, 0.1)
    samples = np.clip(
        blue + rng.normal(0.0, 0.002, size=(3000, 3)), -1.0, 1.0
    )
    colors = inference.extract_colors_from_outputs(
        centroids, weights, samples_lab=samples
    )
    # Sin minoría lejana no se inventa un color: todo sigue siendo azul.
    assert _closest_delta("#0000FF", colors) < 15.0
    assert all(_delta_e_hex("#FF0000", c.value) > 30.0 for c in colors)


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

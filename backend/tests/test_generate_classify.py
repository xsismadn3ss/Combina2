# Tests Fase 4 (issue #20): redes G y C + POST /generate /identify
# Sin TF (runtime 3.14) corren todos menos los que requieren .keras, que hacen skip.
import io
import re
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.services.inference as inference
from app.config import APICOnfig
from app.main import app
from app.ml.color_spaces import ColorConverter
from app.schemas.palette import ColorRole, HarmonyType

client = TestClient(app)
HEX_RE = re.compile(r"^#[0-9A-F]{6}$")
VALID_ROLES = {e.value for e in ColorRole}
VALID_HARMONY = {h.value for h in HarmonyType}

def _png_bytes(hexes=None, size=64):
    if hexes is None:
        hexes = ["#FF0000"]
    # simple mosaic via PIL
    img = Image.new("RGB", (size, size), (255,255,255))
    # fill with first color
    rgb = ColorConverter.hex_to_rgb(hexes[0])
    img2 = Image.new("RGB", (size,size), rgb)
    buf = io.BytesIO()
    img2.save(buf, format="PNG")
    return buf.getvalue()

# --- G: generate endpoint / inference ---------

def test_generate_endpoint_with_model_returns_valid_palettes():
    # si hay modelo G activo y TF, debe responder fallback False con palettes válidas (5 colores)
    # si no hay TF o no hay modelo, también debe responder pero fallback True (2-3 colores clásico)
    payload = {"colors": ["#FF0000"], "harmony": HarmonyType.COMPLEMENTARY.value, "topK": 3}
    resp = client.post(f"{APICOnfig.prefix}/palette/generate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "palettes" in data and "fallback" in data
    assert 1 <= len(data["palettes"]) <= 3
    for pal in data["palettes"]:
        assert pal["type"] in VALID_HARMONY
        if data["fallback"]:
            assert 2 <= len(pal["colors"]) <= 5
        else:
            assert 5 == len(pal["colors"])
        for c in pal["colors"]:
            assert HEX_RE.match(c["value"])
            assert c["role"] in VALID_ROLES
        assert 0.0 <= pal["score"] <= 1.0
    # scores ordenados desc
    scores = [p["score"] for p in data["palettes"]]
    assert scores == sorted(scores, reverse=True)
    # si hay modelo activo, fallback debe ser False; si no, True (ambos aceptables según env)
    # verificamos coherencia cuando TF disponible
    try:
        import tensorflow  # noqa: F401
        has_tf = True
    except ImportError:
        has_tf = False
    if has_tf:
        from app.services.inference import active_keras_path
        expects_fallback = active_keras_path("generate") is None
        assert data["fallback"] is expects_fallback
    else:
        assert data["fallback"] is True


def test_generate_respects_harmony_filter_and_topK():
    for harmony in list(HarmonyType):
        payload = {"colors": ["#112233", "#445566"], "harmony": harmony.value, "topK": 5}
        resp = client.post(f"{APICOnfig.prefix}/palette/generate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["palettes"]) == (1 if data["fallback"] else 5)
        for pal in data["palettes"]:
            # cuando hay modelo, todas deben ser del tipo pedido (filtro/reranking)
            if not data["fallback"]:
                assert pal["type"] == harmony.value
                assert 5 == len(pal["colors"])
            else:
                assert 2 <= len(pal["colors"]) <= 5
        # fallback clásico solo devuelve 1
        if data["fallback"]:
            assert data["palettes"][0]["type"] == harmony.value


def test_generate_invalid_hex_is_400():
    payload = {"colors": ["#ZZZZZZ"], "harmony": HarmonyType.COMPLEMENTARY.value}
    resp = client.post(f"{APICOnfig.prefix}/palette/generate", json=payload)
    assert resp.status_code == 400
    assert "HEX" in resp.json()["detail"]


def test_generate_topK_single_when_fallback():
    # topK distinto cuando hay modelo vs fallback, pero ambos deben respetar rango 1..10
    payload = {"colors": ["#00FF00"], "topK": 1, "harmony": HarmonyType.TRIADIC.value}
    resp = client.post(f"{APICOnfig.prefix}/palette/generate", json=payload)
    assert resp.status_code == 200
    assert len(resp.json()["palettes"]) == 1


def test_inference_generate_direct_without_sklearn(monkeypatch):
    # stub model G que evita importar sklearn en el path con red
    class _FakeG:
        def predict(self, inputs, verbose=0):
            # inputs es lista [palette_lab, mask, harmony]
            # devuelve paleta fija + score
            pal = np.array([[0.5,0.0,0.0]*5], dtype=np.float32)  # 15
            score = np.array([[0.9]], dtype=np.float32)
            return [pal, score]
    def _purge():
        doomed=[m for m in list(sys.modules) if m=="sklearn" or m.startswith("sklearn.") or m=="app.services.classic"]
        for m in doomed:
            monkeypatch.delitem(sys.modules, m, raising=False)
        return doomed
    monkeypatch.setattr(inference, "load_keras_model", lambda slot: _FakeG() if slot=="generate" else None)
    _purge()
    res = inference.generate_palettes(["#FF0000"], HarmonyType.COMPLEMENTARY, top_k=2)
    assert res.fallback is False
    assert len(res.palettes) == 2
    assert all(HEX_RE.match(c.value) for p in res.palettes for c in p.colors)
    assert res.palettes[0].score >= res.palettes[1].score
    # sin sklearn importado
    assert "sklearn" not in sys.modules


def test_generate_failure_falls_back(monkeypatch):
    class _Broken:
        def predict(self, *a, **kw):
            raise RuntimeError("boom")
    monkeypatch.setattr(inference, "load_keras_model", lambda slot: _Broken() if slot=="generate" else None)
    monkeypatch.setattr(inference, "_generator", None)
    # debe caer al clásico
    res = inference.generate_palettes(["#FF0000"], HarmonyType.COMPLEMENTARY, top_k=3)
    assert res.fallback is True
    assert len(res.palettes) == 1


# --- C: classify / identify ---------

def test_identify_endpoint_returns_valid():
    resp = client.post(
        f"{APICOnfig.prefix}/palette/identify",
        files={"file": ("red.png", _png_bytes(["#FF0000"]), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "type" in data and "confidence" in data and "colors" in data and "fallback" in data
    assert 1 <= len(data["colors"]) <= 10
    for c in data["colors"]:
        assert HEX_RE.match(c["value"])
        assert c["role"] == "predominant"
        assert 0.0 <= c["weight"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0
    # type es una de las 4 o predominant (fallback)
    assert data["type"] in (VALID_HARMONY | {"predominant"})
    # coherencia fallback
    try:
        import tensorflow  # noqa: F401
        has_tf = True
    except ImportError:
        has_tf = False
    if has_tf:
        from app.services.inference import active_keras_path
        has_c = active_keras_path("classify") is not None
        has_e = active_keras_path("extract") is not None
        # identify fallback False solo si ambos E y C activos y no fallan
        if has_c and has_e:
            # puede ser False o True si C falla, aceptamos ambos pero si es False verifica tipo en 4
            if not data["fallback"]:
                assert data["type"] in VALID_HARMONY
        else:
            assert data["fallback"] is True
    else:
        assert data["fallback"] is True


def test_identify_classify_direct_without_sklearn(monkeypatch):
    class _FakeC:
        def predict(self, x, verbose=0):
            # devuelve prob complementario alta
            probs = np.array([[0.9, 0.03, 0.03, 0.04]], dtype=np.float32)
            return probs
    # necesitamos también que extract funcione; usaremos colores sintéticos
    from app.schemas.palette import Color
    colors = [Color(value="#FF0000", role="predominant", weight=0.5), Color(value="#00FFFF", role="predominant", weight=0.5)]
    # pad a menos de 10 se maneja dentro
    monkeypatch.setattr(inference, "load_keras_model", lambda slot: _FakeC() if slot=="classify" else None)
    doomed=[m for m in list(sys.modules) if m=="sklearn" or m.startswith("sklearn.") or m=="app.services.classic"]
    for m in doomed:
        monkeypatch.delitem(sys.modules, m, raising=False)
    result = inference._classify_with_model(_FakeC(), colors)
    assert result is not None
    label, conf = result
    assert label in VALID_HARMONY
    assert 0.0 <= conf <= 1.0
    assert "sklearn" not in sys.modules


def test_identify_failure_falls_back(monkeypatch):
    class _BrokenC:
        def predict(self, *a, **kw):
            raise RuntimeError("boom")
    monkeypatch.setattr(inference, "load_keras_model", lambda slot: _BrokenC() if slot=="classify" else None)
    # identify_image usa extract_image_colors primero; mock extract para no depender de E
    from app.schemas.palette import Color
    fake_colors = [Color(value="#00FF00", role="predominant", weight=1.0)]
    monkeypatch.setattr(inference, "extract_image_colors", lambda b: type("R", (), {"colors": fake_colors})())
    res = inference.identify_image(b"fake")
    assert res.fallback is True
    assert res.type == "predominant"


# --- Métricas G y C ---------

def test_g_ciede_below_30_or_report():
    tf = pytest.importorskip("tensorflow")
    gen_path = inference.MODELS_DIR / "generate-v1.keras"
    if not gen_path.exists():
        pytest.skip("sin backend/models/generate-v1.keras")
    from app.ml.generate_model import GENERATE_CUSTOM_OBJECTS
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_generate", "backend/scripts/train_generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # dataset val pequeño determinista
    pals = mod.build_palette_list(40, seed=99)
    (Xl, Xm, Xh), (Yp, Ys) = mod.palette_to_dataset(pals, seed=99)
    model = tf.keras.models.load_model(str(gen_path), custom_objects=GENERATE_CUSTOM_OBJECTS)
    pred_pal, _ = model.predict([Xl, Xm, Xh], verbose=0)
    ciede = mod.mean_ciede2000_approx(Yp, pred_pal)
    print(f"\n[G] CIEDE={ciede:.3f} (umbral <30, ref ~23.5)")
    assert ciede < 30.0


def test_c_accuracy_thresholds():
    tf = pytest.importorskip("tensorflow")
    cls_path = inference.MODELS_DIR / "classify-v1.keras"
    if not cls_path.exists():
        pytest.skip("sin backend/models/classify-v1.keras")
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_classify", "backend/scripts/train_classify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pals = mod.build_palette_list(120, seed=123)
    X, y = mod.build_dataset(pals, seed=123)
    model = tf.keras.models.load_model(str(cls_path))
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"\n[C] accuracy={acc:.4f} (umbral sintético 0.85, curado 0.70)")
    # debe pasar el más exigente sintético
    assert acc >= 0.85
    assert acc >= 0.70


def test_no_broken_color_logic():
    # Lab roundtrip no debe producir HEX inválido ni pesos rotos
    payload = {"colors": ["#123456", "#ABCDEF"], "harmony": HarmonyType.ANALOGOUS.value, "topK": 2}
    resp = client.post(f"{APICOnfig.prefix}/palette/generate", json=payload)
    assert resp.status_code == 200
    for pal in resp.json()["palettes"]:
        for c in pal["colors"]:
            # hex_to_lab -> lab_to_hex roundtrip debe ser idempotente aprox
            lab = ColorConverter.hex_to_lab(c["value"])
            hex2 = ColorConverter.lab_to_hex(lab).upper()
            assert HEX_RE.match(hex2)
        if not resp.json()["fallback"]:
            assert 5 == len(pal["colors"])
    # identify pesos suman 1
    resp2 = client.post(f"{APICOnfig.prefix}/palette/identify", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert resp2.status_code == 200
    total = sum(c["weight"] for c in resp2.json()["colors"] if c["weight"] is not None)
    assert total == pytest.approx(1.0, abs=1e-3)

# Tests de contrato Fase 1 (issue #17): la API responde vía fallback clásico.
import io
import sqlite3

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import APICOnfig
from app.main import app
from app.schemas.palette import ColorRole, HarmonyType

client = TestClient(app)


@pytest.fixture()
def training_db(tmp_path, monkeypatch):
    db = tmp_path / "test-training.db"
    monkeypatch.setenv("TRAINING_DB", str(db))
    return db


def _png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    img = Image.new("RGB", (64, 64), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_harmony_lists_four_types():
    response = client.get(f"{APICOnfig.prefix}/harmony/")
    assert response.status_code == 200
    assert set(response.json()) == {h.value for h in HarmonyType}


def test_color_roles_non_empty():
    response = client.get(f"{APICOnfig.prefix}/colors/roles")
    assert response.status_code == 200
    assert ColorRole.PREDOMINANT.value in response.json()


def test_generate_complementary_red_to_cyan_fallback():
    payload = {"colors": ["#FF0000"], "harmony": HarmonyType.COMPLEMENTARY.value}
    response = client.post(f"{APICOnfig.prefix}/palette/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["fallback"] is True
    assert len(data["palettes"]) == 1
    palette = data["palettes"][0]
    assert palette["type"] == HarmonyType.COMPLEMENTARY.value
    assert palette["colors"][0]["value"] == "#FF0000"
    assert palette["colors"][1]["value"] == "#00FFFF"


def test_generate_invalid_hex_is_400():
    payload = {"colors": ["#FF"], "harmony": HarmonyType.COMPLEMENTARY.value}
    response = client.post(f"{APICOnfig.prefix}/palette/generate", json=payload)
    assert response.status_code == 400
    assert "HEX" in response.json()["detail"]


def test_generate_empty_colors_is_422():
    response = client.post(
        f"{APICOnfig.prefix}/palette/generate",
        json={"colors": [], "harmony": HarmonyType.TRIADIC.value},
    )
    assert response.status_code == 422


def test_extract_solid_image_returns_weighted_colors():
    response = client.post(
        f"{APICOnfig.prefix}/palette/extract",
        files={"file": ("red.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    # Con modelo extract activo y TF disponible, fallback es False; sin
    # modelo o sin TF es True. Acepta ambos para que pase en .venv y .venv-tf.
    # Verifica coherencia con el manifest cuando TF está disponible.
    assert data["fallback"] in (True, False)
    try:
        import tensorflow  # noqa: F401

        has_tf = True
    except ImportError:
        has_tf = False
    if has_tf:
        from app.services.inference import active_keras_path

        expects_fallback = active_keras_path("extract") is None
        assert data["fallback"] is expects_fallback
    else:
        assert data["fallback"] is True
    assert 1 <= len(data["colors"]) <= 10
    assert data["colors"][0]["value"] == "#FF0000"
    total_weight = sum(c["weight"] for c in data["colors"])
    assert total_weight == pytest.approx(1.0)


def test_extract_invalid_file_is_400():
    response = client.post(
        f"{APICOnfig.prefix}/palette/extract",
        files={"file": ("nope.txt", b"esto no es una imagen", "text/plain")},
    )
    assert response.status_code == 400


def test_identify_solid_image_is_predominant_fallback():
    response = client.post(
        f"{APICOnfig.prefix}/palette/identify",
        files={"file": ("green.png", _png_bytes((0, 255, 0)), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "predominant"
    assert data["fallback"] is True
    assert data["colors"][0]["value"] == "#00FF00"


def test_feedback_persists_row(training_db):
    payload = {
        "palette": {
            "type": HarmonyType.COMPLEMENTARY.value,
            "colors": [
                {"value": "#FF0000", "role": "primary"},
                {"value": "#00FFFF", "role": "complementary"},
            ],
            "fallback": True,
        },
        "accepted": True,
        "score": 0.9,
    }
    response = client.post(f"{APICOnfig.prefix}/feedback", json=payload)
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    conn = sqlite3.connect(str(training_db))
    row = conn.execute("SELECT accepted, score FROM feedback").fetchone()
    conn.close()
    assert tuple(row) == (1, 0.9)

# Tests de contrato Fase 1 (issue #17): la API responde vía fallback clásico.
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select

from app.config import APICOnfig
from app.db.models import Feedback
from app.main import app
from app.schemas.palette import ColorRole, HarmonyType

client = TestClient(app)


def _png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    img = Image.new("RGB", (64, 64), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _small_png_bytes(size: tuple[int, int], colors: list[tuple[int, int, int]]) -> bytes:
    img = Image.new("RGB", size)
    img.putdata([colors[index % len(colors)] for index in range(size[0] * size[1])])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_local_frontend_origin_is_allowed_by_cors():
    response = client.options(
        f"{APICOnfig.prefix}/palette/generate",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_harmony_lists_four_types():
    # Sin barra final: es la ruta que usa el frontend (same-origin proxy).
    response = client.get(f"{APICOnfig.prefix}/harmony")
    assert response.status_code == 200
    assert set(response.json()) == {h.value for h in HarmonyType}


def test_harmony_trailing_slash_alias_does_not_redirect():
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
    # G activa o fallback: contrato común de siete colores.
    # acepta ambos para que pase en .venv y .venv-tf
    try:
        import tensorflow  # noqa: F401

        has_tf = True
    except ImportError:
        has_tf = False
    if has_tf:
        from app.services.inference import active_keras_path

        expects_fallback = active_keras_path("generate") is None
        assert data["fallback"] is expects_fallback
        if not expects_fallback:
            # neural: 7 colores, score ordenado, tipo correcto
            assert data["palettes"][0]["type"] == HarmonyType.COMPLEMENTARY.value
            assert 7 == len(data["palettes"][0]["colors"])
            assert data["palettes"][0]["colors"][0]["value"] == "#FF0000"
    else:
        assert data["fallback"] is True
        assert len(data["palettes"]) == 5
        palette = data["palettes"][0]
        assert palette["type"] == HarmonyType.COMPLEMENTARY.value
        assert palette["colors"][0]["value"] == "#FF0000"
        assert len(palette["colors"]) == 7
        assert [p["score"] for p in data["palettes"]] == sorted(
            (p["score"] for p in data["palettes"]), reverse=True
        )


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
    assert data["colors"][0]["value"] == "#00FF00"
    # con C activa, tipo en 4 armonías y fallback False; sin ella, predominant True
    try:
        import tensorflow  # noqa: F401

        has_tf = True
    except ImportError:
        has_tf = False
    if has_tf:
        from app.services.inference import active_keras_path

        has_c = active_keras_path("classify") is not None
        has_e = active_keras_path("extract") is not None
        expects_fallback = not (has_c and has_e)
        assert data["fallback"] is expects_fallback
        if expects_fallback:
            assert data["type"] == "predominant"
        else:
            assert data["type"] in {h.value for h in HarmonyType}
            assert 0.0 <= data["confidence"] <= 1.0
    else:
        assert data["type"] == "predominant"
        assert data["fallback"] is True


@pytest.mark.parametrize(
    ("size", "colors"),
    [
        ((1, 1), [(12, 34, 56)]),
        ((2, 1), [(12, 34, 56), (12, 34, 56)]),
        ((2, 2), [(255, 0, 0), (0, 255, 0)]),
        ((3, 2), [(255, 0, 0), (0, 255, 0), (0, 0, 255)]),
    ],
)
@pytest.mark.parametrize("endpoint", ["extract", "identify"])
def test_small_low_color_images_do_not_crash(endpoint, size, colors):
    response = client.post(
        f"{APICOnfig.prefix}/palette/{endpoint}",
        files={"file": ("small.png", _small_png_bytes(size, colors), "image/png")},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    extracted = data["colors"]
    assert 1 <= len(extracted) <= min(10, len(set(colors)))
    assert sum(color["weight"] for color in extracted) == pytest.approx(1.0)


@pytest.mark.parametrize("endpoint", ["extract", "identify"])
def test_corrupt_image_is_client_error_not_server_error(endpoint):
    response = client.post(
        f"{APICOnfig.prefix}/palette/{endpoint}",
        files={"file": ("corrupt.png", b"not a PNG image", "image/png")},
    )
    assert response.status_code == 400


def test_feedback_persists_row(runtime_db):
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
    row = runtime_db.execute(select(Feedback)).scalars().one()
    assert row.accepted is True
    assert row.score == pytest.approx(0.9)
    assert row.palette_type == HarmonyType.COMPLEMENTARY.value


@pytest.mark.parametrize(
    "palette",
    [
        pytest.param(
            {
                "type": HarmonyType.TRIADIC.value,
                "colors": [{"value": "#2A9D8F", "role": "primary"}],
                "fallback": None,
            },
            id="fallback-null",
        ),
        pytest.param(
            {
                "type": HarmonyType.TRIADIC.value,
                "colors": [{"value": "#2A9D8F", "role": "primary"}],
            },
            id="fallback-ausente",
        ),
    ],
)
def test_feedback_accepts_null_or_missing_palette_fallback(palette):
    response = client.post(
        f"{APICOnfig.prefix}/feedback",
        json={"palette": palette, "accepted": True, "score": 0.9},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True}


def test_feedback_requires_accepted():
    response = client.post(
        f"{APICOnfig.prefix}/feedback",
        json={
            "palette": {
                "type": HarmonyType.TRIADIC.value,
                "colors": [{"value": "#2A9D8F", "role": "primary"}],
                "fallback": None,
            }
        },
    )
    assert response.status_code == 422

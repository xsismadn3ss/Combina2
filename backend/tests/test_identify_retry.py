"""Tests del aviso de reintento y la confianza calibrada en `/identify` (B2)."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import DEFAULT_IDENTIFY_MIN_CONFIDENCE, identify_min_confidence
from app.main import app
from app.ml.calibration import apply_temperature
from app.schemas.palette import Color, ExtractResponse, IdentifyResponse
import app.services.inference as inference


COLORS = [
    Color(value="#FF0000", role="predominant", weight=0.6),
    Color(value="#00FF00", role="predominant", weight=0.4),
]

# Colores extraídos de una imagen gris/monocroma: croma ~0.
GRAY_COLORS = [
    Color(value="#808080", role="predominant", weight=0.6),
    Color(value="#B0B0B0", role="predominant", weight=0.4),
]


class _FakeC:
    def __init__(self, probs):
        self._probs = np.asarray([probs], dtype=np.float32)

    def predict(self, x, verbose=0):
        return self._probs


def _setup(monkeypatch, probs, *, calibration=None, colors=None):
    monkeypatch.setattr(
        inference,
        "load_keras_model",
        lambda slot: _FakeC(probs) if slot == "classify" else None,
    )
    extracted = COLORS if colors is None else colors
    monkeypatch.setattr(
        inference,
        "extract_image_colors",
        lambda image_bytes: ExtractResponse(colors=extracted, fallback=False),
    )
    monkeypatch.setattr(
        inference, "load_calibration", lambda slot: dict(calibration or {})
    )


def test_low_confidence_requires_retry_with_suggestion(monkeypatch):
    monkeypatch.delenv("IDENTIFY_MIN_CONFIDENCE", raising=False)
    _setup(monkeypatch, [0.40, 0.30, 0.20, 0.10])
    response = inference.identify_image(b"fake")
    assert response.fallback is False
    assert response.needsRetry is True
    assert response.suggestion
    assert "foto" in response.suggestion.lower()
    assert response.confidenceCalibrated is False


def test_high_confidence_does_not_require_retry(monkeypatch):
    monkeypatch.delenv("IDENTIFY_MIN_CONFIDENCE", raising=False)
    _setup(monkeypatch, [0.90, 0.04, 0.03, 0.03])
    response = inference.identify_image(b"fake")
    assert response.needsRetry is False
    assert response.suggestion is None


def _png_solid(rgb, size=32) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (size, size), rgb).save(buf, format="PNG")
    return buf.getvalue()


def _png_two_tone(left_rgb, right_rgb, size=32) -> bytes:
    import io

    from PIL import Image

    image = Image.new("RGB", (size, size), left_rgb)
    for x in range(size // 2, size):
        for y in range(size):
            image.putpixel((x, y), right_rgb)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_grayscale_png_requires_retry_even_with_high_confidence(monkeypatch):
    from app.services.classic import ClassicPaletteIdentifier

    gray_png = _png_solid((128, 128, 128))
    gray_colors = ClassicPaletteIdentifier().identify_palette(gray_png)
    _setup(monkeypatch, [0.90, 0.04, 0.03, 0.03], colors=gray_colors)

    response = inference.identify_image(gray_png)
    assert response.fallback is False
    assert response.confidence > 0.70  # la confianza calibrada es alta...
    assert response.needsRetry is True  # ...pero la imagen es gris
    assert response.suggestion
    assert "monocroma" in response.suggestion.lower()


def test_colorful_png_does_not_require_retry(monkeypatch):
    from app.services.classic import ClassicPaletteIdentifier

    colorful_png = _png_two_tone((220, 20, 20), (20, 200, 20))
    colors = ClassicPaletteIdentifier().identify_palette(colorful_png)
    _setup(monkeypatch, [0.90, 0.04, 0.03, 0.03], colors=colors)

    response = inference.identify_image(colorful_png)
    assert response.fallback is False
    assert response.needsRetry is False
    assert response.suggestion is None


def test_dim_duo_with_different_hues_does_not_require_retry(monkeypatch):
    from app.schemas.palette import Color

    dim_duo = [
        Color(value="#E6DCD2", role="predominant", weight=0.5),
        Color(value="#D2E1EB", role="predominant", weight=0.5),
    ]
    _setup(monkeypatch, [0.90, 0.04, 0.03, 0.03], colors=dim_duo)
    response = inference.identify_image(b"fake")
    assert response.fallback is False
    assert response.needsRetry is False
    assert response.suggestion is None


def test_fallback_with_grayscale_never_requires_retry(monkeypatch):
    monkeypatch.setattr(inference, "load_keras_model", lambda slot: None)
    monkeypatch.setattr(
        inference,
        "extract_image_colors",
        lambda image_bytes: ExtractResponse(colors=GRAY_COLORS, fallback=False),
    )
    response = inference.identify_image(b"fake")
    assert response.fallback is True
    assert response.type == "predominant"
    assert response.needsRetry is False
    assert response.suggestion is None


def test_fallback_never_requires_retry(monkeypatch):
    monkeypatch.setenv("IDENTIFY_MIN_CONFIDENCE", "0.99")
    monkeypatch.setattr(inference, "load_keras_model", lambda slot: None)
    monkeypatch.setattr(
        inference,
        "extract_image_colors",
        lambda image_bytes: ExtractResponse(colors=COLORS, fallback=False),
    )
    response = inference.identify_image(b"fake")
    assert response.fallback is True
    assert response.type == "predominant"
    assert response.needsRetry is False
    assert response.suggestion is None
    assert response.confidenceCalibrated is False


def test_env_override_of_min_confidence(monkeypatch):
    _setup(monkeypatch, [0.40, 0.30, 0.20, 0.10])
    monkeypatch.setenv("IDENTIFY_MIN_CONFIDENCE", "0.30")
    assert inference.identify_image(b"fake").needsRetry is False
    monkeypatch.setenv("IDENTIFY_MIN_CONFIDENCE", "0.95")
    assert inference.identify_image(b"fake").needsRetry is True
    # Valor inválido -> default.
    monkeypatch.setenv("IDENTIFY_MIN_CONFIDENCE", "no-es-un-numero")
    assert identify_min_confidence() == DEFAULT_IDENTIFY_MIN_CONFIDENCE


def test_confidence_calibrated_flag_and_value(monkeypatch):
    monkeypatch.delenv("IDENTIFY_MIN_CONFIDENCE", raising=False)
    raw = [0.45, 0.25, 0.20, 0.10]
    calibration = {"method": "temperature", "calibrated": True, "temperature": 0.5}
    _setup(monkeypatch, raw, calibration=calibration)
    response = inference.identify_image(b"fake")
    expected = float(np.max(apply_temperature(np.asarray(raw), 0.5)))
    assert response.confidenceCalibrated is True
    assert response.confidence == pytest.approx(expected)
    assert response.type == "complementario"

    # Calibración inactiva -> confianza cruda y flag false.
    _setup(monkeypatch, raw, calibration={"method": "temperature", "calibrated": False})
    raw_response = inference.identify_image(b"fake")
    assert raw_response.confidenceCalibrated is False
    assert raw_response.confidence == pytest.approx(max(raw))


def test_identify_response_schema_and_openapi_contract():
    response = IdentifyResponse(
        type="triada", confidence=0.8, colors=COLORS, fallback=False
    )
    payload = response.model_dump()
    assert payload["confidenceCalibrated"] is False
    assert payload["needsRetry"] is False
    assert payload["suggestion"] is None

    schema = app.openapi()["components"]["schemas"]["IdentifyResponse"]["properties"]
    for field in ("confidenceCalibrated", "needsRetry", "suggestion"):
        assert field in schema
    assert schema["confidenceCalibrated"]["type"] == "boolean"
    assert schema["needsRetry"]["type"] == "boolean"

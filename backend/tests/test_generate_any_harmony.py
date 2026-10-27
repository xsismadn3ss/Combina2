"""Issue #37: `harmony` omitido/null = "cualquier estilo" con tipos variados.

El backend ya no fuerza `COMPLEMENTARY` cuando el cliente no elige estilo:
resuelve la armonía de cada propuesta por rotación determinista de las 4
clases, iniciando en un índice derivado del hash estable de las anclas.
"""

import json
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.services.inference as inference
from app.config import APICOnfig
from app.main import app
from app.schemas.palette import ColorRole, HarmonyType

client = TestClient(app)
HEX_RE = re.compile(r"^#[0-9A-F]{6}$")
VALID_HARMONY = {h.value for h in HarmonyType}
VALID_ROLES = {role.value for role in ColorRole}
ANCHORS = ["#E63946", "#F1C40F", "#1D3557"]


class _FakeG:
    """Red G falsa: paleta fija + score fijo, sin TF ni sklearn."""

    def __init__(self):
        self.harmonies: list[tuple[float, ...]] = []

    def predict(self, inputs, verbose=0):
        harmony = np.asarray(inputs[2], dtype=np.float32)
        self.harmonies.append(tuple(float(v) for v in harmony.reshape(-1)))
        palette = np.array([[0.55, 0.08, 0.05] * 5], dtype=np.float32)
        score = np.array([[0.82]], dtype=np.float32)
        return [palette, score]


def _post(payload: dict) -> dict:
    response = client.post(f"{APICOnfig.prefix}/palette/generate", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _assert_valid(data: dict, expected_count: int, anchors: list[str]) -> None:
    assert len(data["palettes"]) == expected_count
    scores = [palette["score"] for palette in data["palettes"]]
    assert scores == sorted(scores, reverse=True)
    for palette in data["palettes"]:
        assert palette["type"] in VALID_HARMONY
        assert len(palette["colors"]) == inference.PALETTE_SIZE
        assert 0.0 <= palette["score"] <= 1.0
        values = [color["value"].upper() for color in palette["colors"]]
        assert all(HEX_RE.match(value) for value in values)
        assert all(color["role"] in VALID_ROLES for color in palette["colors"])
        assert set(anchors).issubset(values)


@pytest.mark.parametrize("include_harmony", [False, True])
def test_any_harmony_without_model_covers_four_types(monkeypatch, include_harmony):
    """Omitido y `null` son equivalentes: 4 tipos reales, 7 colores."""
    monkeypatch.setattr(inference, "load_keras_model", lambda _slot: None)
    payload = {"colors": ANCHORS, "topK": 5}
    if include_harmony:
        payload["harmony"] = None

    data = _post(payload)

    assert data["fallback"] is True
    _assert_valid(data, 5, ANCHORS)
    assert {palette["type"] for palette in data["palettes"]} == VALID_HARMONY


def test_any_harmony_with_fake_model_keeps_diversity(monkeypatch):
    fake = _FakeG()
    monkeypatch.setattr(
        inference, "load_keras_model", lambda slot: fake if slot == "generate" else None
    )

    data = _post({"colors": ANCHORS, "topK": 5, "harmony": None})

    assert data["fallback"] is False
    _assert_valid(data, 5, ANCHORS)
    assert {palette["type"] for palette in data["palettes"]} == VALID_HARMONY
    # Una inferencia por armonía de la rotación (máximo 4), nunca un one-hot
    # todo ceros.
    assert 1 <= len(fake.harmonies) <= 4
    assert all(sum(harmony) == pytest.approx(1.0) for harmony in fake.harmonies)


@pytest.mark.parametrize("harmony", list(HarmonyType))
def test_explicit_harmony_still_filters_without_model(monkeypatch, harmony):
    monkeypatch.setattr(inference, "load_keras_model", lambda _slot: None)

    data = _post({"colors": ANCHORS, "topK": 4, "harmony": harmony.value})

    assert data["fallback"] is True
    _assert_valid(data, 4, ANCHORS)
    assert all(palette["type"] == harmony.value for palette in data["palettes"])


@pytest.mark.parametrize("harmony", list(HarmonyType))
def test_explicit_harmony_still_filters_with_model(monkeypatch, harmony):
    fake = _FakeG()
    monkeypatch.setattr(
        inference, "load_keras_model", lambda slot: fake if slot == "generate" else None
    )

    data = _post({"colors": ANCHORS, "topK": 4, "harmony": harmony.value})

    assert data["fallback"] is False
    _assert_valid(data, 4, ANCHORS)
    assert all(palette["type"] == harmony.value for palette in data["palettes"])


def test_any_harmony_is_deterministic(monkeypatch):
    monkeypatch.setattr(inference, "load_keras_model", lambda _slot: None)
    payload = {"colors": ANCHORS, "topK": 5, "harmony": None}

    first = _post(payload)
    second = _post(payload)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_any_harmony_small_topk_returns_rotation_subset(monkeypatch, top_k):
    monkeypatch.setattr(inference, "load_keras_model", lambda _slot: None)

    data = _post({"colors": ANCHORS, "topK": top_k, "harmony": None})

    assert data["fallback"] is True
    _assert_valid(data, top_k, ANCHORS)
    types = [palette["type"] for palette in data["palettes"]]
    # Subconjunto sin repeticiones de la rotación de 4 clases.
    assert len(set(types)) == len(types)
    assert set(types).issubset(VALID_HARMONY)
    # La secuencia coincide con la rotación documentada para estas anclas.
    expected = [h.value for h in inference.harmony_sequence(ANCHORS, None, top_k)]
    assert types == expected


def test_classic_fallback_without_harmony_still_works(monkeypatch):
    """Regresión: el fallback clásico sigue dando 7 colores y anclas exactas."""
    monkeypatch.setattr(inference, "load_keras_model", lambda _slot: None)

    result = inference.generate_palettes(ANCHORS, None, top_k=5)

    assert result.fallback is True
    assert len(result.palettes) == 5
    for palette in result.palettes:
        values = [color.value.upper() for color in palette.colors]
        assert len(values) == inference.PALETTE_SIZE
        assert set(ANCHORS).issubset(values)

"""Issue #40: seleccionar 5 colores no debe devolver la misma paleta.

La paleta generada tiene siempre ``PALETTE_SIZE`` (7) colores: las anclas
elegidas se conservan al inicio y el resto son acentos generados. La UI limita
la selección a 4, pero la API sigue aceptando hasta 5 y debe completar la
paleta en vez de repetir las anclas.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.services.inference as inference
from app.config import APICOnfig
from app.main import app
from app.schemas.palette import ColorRole, HarmonyType

client = TestClient(app)
ANCHORS = ["#E63946", "#F1C40F", "#1D3557", "#2A9D8F", "#E76F51"]
VALID_ROLES = {role.value for role in ColorRole}


class _FakeG:
    """Red G falsa: paleta fija + score fijo, sin TF ni sklearn."""

    def predict(self, inputs, verbose=0):
        palette = np.array([[0.55, 0.08, 0.05] * 5], dtype=np.float32)
        score = np.array([[0.82]], dtype=np.float32)
        return [palette, score]


def _post(payload: dict) -> dict:
    response = client.post(f"{APICOnfig.prefix}/palette/generate", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("anchor_count", [1, 2, 3, 4, 5])
def test_palette_is_completed_to_seven_colors(monkeypatch, anchor_count):
    monkeypatch.setattr(inference, "load_keras_model", lambda _slot: None)
    anchors = ANCHORS[:anchor_count]

    data = _post({"colors": anchors, "topK": 5, "harmony": None})

    assert data["fallback"] is True
    assert len(data["palettes"]) == 5
    signatures = []
    for palette in data["palettes"]:
        values = [color["value"].upper() for color in palette["colors"]]
        assert len(values) == inference.PALETTE_SIZE
        assert len(set(values)) == inference.PALETTE_SIZE
        assert set(anchors).issubset(values)
        assert values[:anchor_count] == anchors
        assert all(color["role"] in VALID_ROLES for color in palette["colors"])
        signatures.append(tuple(values))
    assert len(set(signatures)) == len(signatures)


def test_five_anchors_are_not_returned_unchanged(monkeypatch):
    monkeypatch.setattr(inference, "load_keras_model", lambda _slot: None)

    data = _post({"colors": ANCHORS, "topK": 5, "harmony": None})

    for palette in data["palettes"]:
        values = [color["value"].upper() for color in palette["colors"]]
        generated = [value for value in values if value not in ANCHORS]
        assert len(values) == inference.PALETTE_SIZE
        assert len(generated) == inference.PALETTE_SIZE - len(ANCHORS)


def test_five_anchors_with_fake_model_are_not_returned_unchanged(monkeypatch):
    fake = _FakeG()
    monkeypatch.setattr(
        inference, "load_keras_model", lambda slot: fake if slot == "generate" else None
    )

    data = _post({"colors": ANCHORS, "topK": 5, "harmony": HarmonyType.ANALOGOUS.value})

    assert data["fallback"] is False
    signatures = []
    for palette in data["palettes"]:
        values = [color["value"].upper() for color in palette["colors"]]
        assert len(values) == inference.PALETTE_SIZE
        assert len(set(values)) == inference.PALETTE_SIZE
        assert set(ANCHORS).issubset(values)
        signatures.append(tuple(values))
    assert len(set(signatures)) == len(signatures)


def test_five_anchors_generation_is_deterministic(monkeypatch):
    monkeypatch.setattr(inference, "load_keras_model", lambda _slot: None)
    payload = {"colors": ANCHORS, "topK": 3, "harmony": None}

    first = _post(payload)
    second = _post(payload)

    assert first == second

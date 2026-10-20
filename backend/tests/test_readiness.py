"""Readiness real (`GET /ready`) sin TensorFlow: monkeypatch de carga y registro."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import inference, model_registry

client = TestClient(app)

FILES = {
    "extract": "extract-v1.keras",
    "generate": "generate-v1.keras",
    "classify": "classify-v2.keras",
}


@pytest.fixture(autouse=True)
def _clear_cache():
    inference.clear_model_cache()
    yield
    inference.clear_model_cache()


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Manifiesto mínimo con los 3 slots y archivos falsos hasheados."""
    models = tmp_path / "models"
    models.mkdir()
    manifest: dict = {"schemaVersion": 2, "updatedAt": "2026-10-20T00:00:00Z"}
    for slot, name in FILES.items():
        path = models / name
        path.write_bytes(f"fake-{slot}".encode())
        manifest[slot] = {
            "active": name,
            "versions": [name],
            "files": {name: {"sha256": model_registry.sha256_file(path)}},
        }
    manifest_path = models / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(model_registry, "MODELS_DIR", models)
    monkeypatch.setattr(model_registry, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(inference, "MODELS_DIR", models)
    return models


def _patch_loader(monkeypatch, failing: set[str] | None = None) -> None:
    failing = failing or set()

    def fake_load(slot: str):
        return None if slot in failing else object()

    monkeypatch.setattr(inference, "load_keras_model", fake_load)


def test_ready_200_when_all_slots_load(sandbox, monkeypatch):
    _patch_loader(monkeypatch)

    response = client.get("/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["fallback"] is False
    assert data["storage"] == "models_only"
    assert set(data["models"]) == {"extract", "generate", "classify"}
    for slot, name in FILES.items():
        assert data["models"][slot]["active"] == name
        assert data["models"][slot]["loaded"] is True
        assert data["models"][slot]["fallback"] is False
        assert "error" not in data["models"][slot]


def test_ready_503_when_a_slot_cannot_load(sandbox, monkeypatch):
    _patch_loader(monkeypatch, failing={"classify"})

    response = client.get("/ready")

    assert response.status_code == 503
    data = response.json()
    assert data["ready"] is False
    assert data["fallback"] is True
    assert data["models"]["extract"]["loaded"] is True
    assert data["models"]["generate"]["loaded"] is True
    assert data["models"]["classify"]["loaded"] is False
    assert data["models"]["classify"]["fallback"] is True
    assert data["models"]["classify"]["error"]


def test_ready_503_when_active_file_is_missing(sandbox, monkeypatch):
    _patch_loader(monkeypatch)
    (sandbox / FILES["extract"]).unlink()

    response = client.get("/ready")

    assert response.status_code == 503
    data = response.json()
    assert data["models"]["extract"]["loaded"] is False
    assert data["models"]["extract"]["active"] == FILES["extract"]
    assert "verificable" in data["models"]["extract"]["error"]


def test_ready_never_raises_and_reports_error(sandbox, monkeypatch):
    def boom(slot: str):
        raise RuntimeError("fallo simulado")

    monkeypatch.setattr(inference, "load_keras_model", boom)

    response = client.get("/ready")

    assert response.status_code == 503
    data = response.json()
    assert data["storage"] == "models_only"
    assert all(not item["loaded"] for item in data["models"].values())
    assert all(item["error"] for item in data["models"].values())


def test_health_stays_200_without_loading_models(monkeypatch):
    def boom():
        raise AssertionError("/health no debe calcular readiness")

    monkeypatch.setattr(inference, "readiness_report", boom)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

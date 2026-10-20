"""Rollback en caliente: inferencia y calibración siguen la versión resuelta (B3)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from app.services import inference, model_registry
from app.schemas.palette import Color

REAL_MODELS = Path(model_registry.MODELS_DIR)


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(model_registry, "MODELS_DIR", models)
    monkeypatch.setattr(model_registry, "MANIFEST_PATH", models / "manifest.json")
    monkeypatch.setattr(inference, "MODELS_DIR", models)
    monkeypatch.setattr(model_registry, "git_sha", lambda: "testsha")
    inference.clear_model_cache()
    yield models
    inference.clear_model_cache()


def _copy_models(models_dir: Path, *names: str) -> None:
    for name in names:
        shutil.copy2(REAL_MODELS / name, models_dir / name)


def _entry(models_dir: Path, active: str, previous: str | None, calibration: dict) -> dict:
    files = {}
    for name in {active, previous} - {None}:
        files[name] = {
            "sha256": model_registry.sha256_file(models_dir / name),
            "metrics": {},
            "calibration": calibration if name == active else {},
        }
    entry = {
        "active": active,
        "versions": sorted(files),
        "files": files,
    }
    if previous:
        entry["previous"] = previous
    return entry


def test_classify_rollback_v2_to_v1_uses_40_dims_and_calibration(sandbox, monkeypatch):
    pytest.importorskip("tensorflow")
    if not (REAL_MODELS / "classify-v1.keras").exists():
        pytest.skip("falta classify-v1.keras")
    if not (REAL_MODELS / "classify-v2.keras").exists():
        pytest.skip("falta classify-v2.keras")
    _copy_models(sandbox, "classify-v1.keras", "classify-v2.keras")
    manifest = {
        "schemaVersion": 2,
        "classify": _entry(
            sandbox,
            "classify-v2.keras",
            "classify-v1.keras",
            {"method": "temperature", "calibrated": True, "temperature": 0.266489},
        ),
    }
    model_registry.save_manifest(manifest)

    model_v2 = inference.load_keras_model("classify")
    assert model_v2 is not None
    assert inference._classify_input_dim(model_v2) == 54
    assert inference.load_calibration("classify")["calibrated"] is True

    colors = [
        Color(value="#FF0000", role="predominant", weight=0.3),
        Color(value="#00FFFF", role="predominant", weight=0.7),
    ]
    result_v2 = inference._classify_with_model(model_v2, colors)
    assert result_v2 is not None
    assert result_v2[0] in {"complementario", "triada", "analogo", "monocromatica"}

    rollback = model_registry.rollback("classify")
    assert rollback == {
        "from": "classify-v2.keras",
        "to": "classify-v1.keras",
    }
    assert inference.active_keras_path("classify").name == "classify-v1.keras"
    # La calibración debe leerse de la versión realmente resuelta (v1 legacy).
    assert inference.load_calibration("classify") == {}

    model_v1 = inference.load_keras_model("classify")
    assert model_v1 is not None
    assert model_v1 is not model_v2
    assert inference._classify_input_dim(model_v1) == 40

    seen: dict[str, tuple] = {}
    original = inference.palette_features_legacy

    def _spy(colors):
        features = original(colors)
        seen["shape"] = features.shape
        return features

    monkeypatch.setattr(inference, "palette_features_legacy", _spy)
    result_v1 = inference._classify_with_model(model_v1, colors)
    assert seen["shape"] == (40,)
    assert result_v1 is not None
    assert result_v1[0] in {"complementario", "triada", "analogo", "monocromatica"}
    assert 0.0 <= result_v1[1] <= 1.0
    assert "sklearn" not in sys.modules


def test_extract_rollback_v1_to_previous_v0(sandbox):
    pytest.importorskip("tensorflow")
    if not (REAL_MODELS / "extract-v1.keras").exists():
        pytest.skip("falta extract-v1.keras")
    _copy_models(sandbox, "extract-v1.keras")
    shutil.copy2(sandbox / "extract-v1.keras", sandbox / "extract-v0.keras")
    manifest = {
        "schemaVersion": 2,
        "extract": _entry(sandbox, "extract-v1.keras", "extract-v0.keras", {}),
    }
    model_registry.save_manifest(manifest)

    model_v1 = inference.load_keras_model("extract")
    assert model_v1 is not None
    assert inference.active_keras_path("extract").name == "extract-v1.keras"

    rollback = model_registry.rollback("extract")
    assert rollback == {"from": "extract-v1.keras", "to": "extract-v0.keras"}
    assert inference.active_keras_path("extract").name == "extract-v0.keras"
    model_v0 = inference.load_keras_model("extract")
    assert model_v0 is not None
    assert model_v0 is not model_v1


def test_resolve_falls_back_to_previous_when_active_file_disappears(sandbox):
    pytest.importorskip("tensorflow")
    if not (REAL_MODELS / "extract-v1.keras").exists():
        pytest.skip("falta extract-v1.keras")
    _copy_models(sandbox, "extract-v1.keras")
    shutil.copy2(sandbox / "extract-v1.keras", sandbox / "extract-v0.keras")
    manifest = {
        "schemaVersion": 2,
        "extract": _entry(sandbox, "extract-v1.keras", "extract-v0.keras", {}),
    }
    model_registry.save_manifest(manifest)
    (sandbox / "extract-v1.keras").unlink()

    assert inference.active_keras_path("extract").name == "extract-v0.keras"
    assert inference.load_keras_model("extract") is not None

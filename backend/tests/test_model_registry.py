"""Tests del registro de modelos (B3): versiones, hash, activación, rollback."""

from __future__ import annotations

import json

import pytest

from app.services import model_registry


@pytest.fixture()
def models_dir(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(model_registry, "MODELS_DIR", models)
    monkeypatch.setattr(model_registry, "MANIFEST_PATH", models / "manifest.json")
    monkeypatch.setattr(model_registry, "git_sha", lambda: "testsha")
    return models


def _model(models_dir, name: str, payload: bytes = b"keras-bytes") -> None:
    (models_dir / name).write_bytes(payload)


def _manifest_path(models_dir):
    return models_dir / "manifest.json"


def test_activate_and_rollback_are_reversible(models_dir):
    _model(models_dir, "classify-v1.keras", b"v1")
    _model(models_dir, "classify-v2.keras", b"v2")

    model_registry.activate("classify", "classify-v1.keras")
    result = model_registry.activate("classify", "classify-v2.keras")
    assert result["active"] == "classify-v2.keras"
    assert result["previous"] == "classify-v1.keras"

    first = model_registry.rollback("classify")
    assert first == {"from": "classify-v2.keras", "to": "classify-v1.keras"}
    assert model_registry.active_version("classify") == "classify-v1.keras"
    assert model_registry.previous_version("classify") == "classify-v2.keras"

    second = model_registry.rollback("classify")
    assert second == {"from": "classify-v1.keras", "to": "classify-v2.keras"}
    assert model_registry.active_version("classify") == "classify-v2.keras"
    # Ninguna versión se borra del registro.
    assert model_registry.load_manifest()["classify"]["versions"] == [
        "classify-v1.keras",
        "classify-v2.keras",
    ]


def test_activate_missing_file_raises(models_dir):
    with pytest.raises(FileNotFoundError):
        model_registry.activate("extract", "extract-v9.keras")


def test_verify_ok_and_hash_recorded(models_dir):
    _model(models_dir, "extract-v1.keras", b"abc")
    model_registry.activate("extract", "extract-v1.keras")
    manifest = model_registry.load_manifest()
    metadata = manifest["extract"]["files"]["extract-v1.keras"]
    assert metadata["sha256"] == model_registry.sha256_file(
        models_dir / "extract-v1.keras"
    )
    result = model_registry.verify()
    assert result["ok"] is True
    assert result["missing"] == []
    assert result["hash_mismatch"] == []


def test_verify_detects_missing_file(models_dir):
    manifest = {
        "schemaVersion": 2,
        "extract": {
            "active": "extract-v2.keras",
            "versions": ["extract-v2.keras"],
            "files": {"extract-v2.keras": {"sha256": "0" * 64}},
        },
    }
    _manifest_path(models_dir).write_text(json.dumps(manifest))
    result = model_registry.verify("extract")
    assert result["ok"] is False
    assert result["missing"] == [
        {"slot": "extract", "file": "extract-v2.keras"}
    ]


def test_verify_detects_corrupted_hash(models_dir):
    _model(models_dir, "generate-v1.keras", b"original")
    model_registry.activate("generate", "generate-v1.keras")
    _model(models_dir, "generate-v1.keras", b"corrupted")
    result = model_registry.verify("generate")
    assert result["ok"] is False
    assert len(result["hash_mismatch"]) == 1
    entry = result["hash_mismatch"][0]
    assert entry["slot"] == "generate"
    assert entry["file"] == "generate-v1.keras"


def test_resolve_falls_back_to_previous(models_dir):
    _model(models_dir, "classify-v1.keras", b"v1")
    sha = model_registry.sha256_file(models_dir / "classify-v1.keras")
    manifest = {
        "schemaVersion": 2,
        "classify": {
            "active": "classify-v2.keras",  # no existe
            "previous": "classify-v1.keras",
            "versions": ["classify-v1.keras", "classify-v2.keras"],
            "files": {
                "classify-v1.keras": {"sha256": sha},
                "classify-v2.keras": {"sha256": "f" * 64},
            },
        },
    }
    _manifest_path(models_dir).write_text(json.dumps(manifest))
    assert model_registry.resolve("classify") == models_dir / "classify-v1.keras"


def test_resolve_falls_back_when_active_hash_mismatch(models_dir):
    _model(models_dir, "extract-v1.keras", b"v1")
    _model(models_dir, "extract-v2.keras", b"changed")
    sha_v1 = model_registry.sha256_file(models_dir / "extract-v1.keras")
    manifest = {
        "schemaVersion": 2,
        "extract": {
            "active": "extract-v2.keras",
            "previous": "extract-v1.keras",
            "versions": ["extract-v1.keras", "extract-v2.keras"],
            "files": {
                "extract-v1.keras": {"sha256": sha_v1},
                "extract-v2.keras": {"sha256": "0" * 64},  # hash incorrecto
            },
        },
    }
    _manifest_path(models_dir).write_text(json.dumps(manifest))
    assert model_registry.resolve("extract") == models_dir / "extract-v1.keras"


def test_resolve_returns_none_when_active_and_previous_missing(models_dir):
    manifest = {
        "schemaVersion": 2,
        "classify": {
            "active": "classify-v9.keras",
            "previous": "classify-v8.keras",
            "versions": ["classify-v8.keras", "classify-v9.keras"],
            "files": {},
        },
    }
    _manifest_path(models_dir).write_text(json.dumps(manifest))
    assert model_registry.resolve("classify") is None


def test_legacy_manifest_without_files_does_not_break(models_dir):
    _model(models_dir, "extract-v1.keras", b"legacy")
    _manifest_path(models_dir).write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "extract": {
                    "active": "extract-v1.keras",
                    "versions": ["extract-v1.keras"],
                },
            }
        )
    )
    assert model_registry.resolve("extract") == models_dir / "extract-v1.keras"
    result = model_registry.verify("extract")
    assert result["ok"] is True
    assert result["unhashed"] == [
        {"slot": "extract", "file": "extract-v1.keras"}
    ]


def test_register_version_refuses_to_overwrite_without_force(models_dir, tmp_path):
    source = tmp_path / "classify-v3.keras"
    source.write_bytes(b"new-model")

    first = model_registry.register_version(
        "classify",
        source,
        metrics={"accuracy": 0.99},
        params={"seed": 1},
        training_run_id=42,
    )
    assert first["filename"] == "classify-v3.keras"
    assert first["active"] == "classify-v3.keras"
    assert model_registry.active_version("classify") == "classify-v3.keras"
    metadata = model_registry.load_manifest()["classify"]["files"][
        "classify-v3.keras"
    ]
    assert metadata["trainingRunId"] == 42
    assert metadata["metrics"] == {"accuracy": 0.99}
    assert metadata["gitSha"] == "testsha"

    with pytest.raises(FileExistsError):
        model_registry.register_version(
            "classify", source, metrics={}, training_run_id=43
        )

    forced = model_registry.register_version(
        "classify", source, metrics={"accuracy": 1.0}, training_run_id=43, force=True
    )
    assert forced["filename"] == "classify-v3.keras"


def test_register_version_copies_temp_file_and_uses_final_name(models_dir, tmp_path):
    source = tmp_path / ".tmp-generate-v4.keras"
    source.write_bytes(b"trained")
    result = model_registry.register_version(
        "generate",
        source,
        metrics={"mse": 0.01},
        params={},
        training_run_id=7,
        filename="generate-v4.keras",
    )
    assert result["filename"] == "generate-v4.keras"
    assert (models_dir / "generate-v4.keras").read_bytes() == b"trained"
    assert not (models_dir / ".tmp-generate-v4.keras").exists()


def test_next_free_version_scans_models_and_manifest(models_dir):
    assert model_registry.next_free_version("extract") == 1
    _model(models_dir, "extract-v2.keras")
    assert model_registry.next_free_version("extract") == 3
    assert model_registry.version_filename("extract", "v5") == "extract-v5.keras"
    assert model_registry.version_filename("extract", 5) == "extract-v5.keras"
    with pytest.raises(ValueError):
        model_registry.parse_version("latest")


def test_rollback_without_previous_raises(models_dir):
    _model(models_dir, "extract-v1.keras")
    model_registry.activate("extract", "extract-v1.keras")
    with pytest.raises(ValueError):
        model_registry.rollback("extract")


def test_save_manifest_is_atomic_and_stamps_metadata(models_dir):
    manifest = {"extract": {"active": None, "versions": [], "files": {}}}
    model_registry.save_manifest(manifest)
    data = json.loads(_manifest_path(models_dir).read_text())
    assert data["schemaVersion"] == 2
    assert data["updatedAt"].endswith("Z")
    assert not list(models_dir.glob(".manifest-*"))


def test_manage_models_rollback_dry_run_does_not_write(models_dir):
    _model(models_dir, "classify-v1.keras", b"v1")
    _model(models_dir, "classify-v2.keras", b"v2")
    model_registry.activate("classify", "classify-v1.keras")
    model_registry.activate("classify", "classify-v2.keras")
    before = _manifest_path(models_dir).read_text()

    from scripts import manage_models

    assert manage_models.main(["rollback", "--slot", "classify", "--dry-run"]) == 0
    assert _manifest_path(models_dir).read_text() == before
    assert model_registry.active_version("classify") == "classify-v2.keras"
    assert model_registry.previous_version("classify") == "classify-v1.keras"


def test_manage_models_verify_exit_code(models_dir):
    _model(models_dir, "extract-v1.keras", b"v1")
    model_registry.activate("extract", "extract-v1.keras")
    from scripts import manage_models

    assert manage_models.main(["verify", "--slot", "extract"]) == 0
    (models_dir / "extract-v1.keras").unlink()
    assert manage_models.main(["verify", "--slot", "extract"]) == 1


def test_models_endpoint_contract_has_no_absolute_paths(models_dir):
    from fastapi.testclient import TestClient

    from app.config import APICOnfig
    from app.main import app

    _model(models_dir, "classify-v1.keras", b"v1")
    _model(models_dir, "classify-v2.keras", b"v2")
    model_registry.activate("classify", "classify-v1.keras")
    model_registry.activate("classify", "classify-v2.keras")

    response = TestClient(app).get(f"{APICOnfig.prefix}/models")
    assert response.status_code == 200
    data = response.json()
    assert data["classify"]["active"] == "classify-v2.keras"
    assert data["classify"]["previous"] == "classify-v1.keras"
    assert data["classify"]["versions"] == [
        "classify-v1.keras",
        "classify-v2.keras",
    ]
    files = data["classify"]["files"]
    assert files["classify-v2.keras"]["sha256"] == model_registry.sha256_file(
        models_dir / "classify-v2.keras"
    )
    assert "trainingRunId" in files["classify-v2.keras"]
    assert "metrics" in files["classify-v2.keras"]
    assert "calibration" in files["classify-v2.keras"]

    raw = json.dumps(data)
    assert str(models_dir) not in raw
    assert "C:\\\\" not in raw
    assert "/models/" not in raw


def test_training_db_migration_adds_b3_columns_idempotently(tmp_path):
    import sqlite3

    from app.db.sqlite import get_connection

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE training_runs(
          id INTEGER PRIMARY KEY,
          model TEXT NOT NULL,
          git_sha TEXT, params_json TEXT,
          metrics_json TEXT,
          keras_path TEXT NOT NULL,
          created_at TEXT DEFAULT (datetime('now'))
        );
        INSERT INTO training_runs(model, keras_path) VALUES('extract', 'x.keras');
        """
    )
    conn.commit()
    conn.close()

    for _ in range(2):  # idempotente
        conn = get_connection(db)
        try:
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(training_runs)")
            }
            assert {"tf_version", "keras_version", "calibration_json"} <= columns
            count = conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
            assert count == 1
        finally:
            conn.close()



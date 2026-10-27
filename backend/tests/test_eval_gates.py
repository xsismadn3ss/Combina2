"""Tests de los gates de eval (issue #22 / B1).

No requieren TensorFlow: los slots se simulan/monkeypatchean. El skip real
sin TF se valida explícitamente (exit 0 no estricto, exit 1 estricto).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import numpy as np

from app.ml import benchmarks
from eval import run_eval


def _thresholds() -> dict:
    return json.loads(run_eval.DEFAULT_THRESHOLDS.read_text())


def _passing_classify() -> dict:
    return {
        "status": "ok",
        "slot": "classify",
        "metrics": {
            "accuracy": 0.975,
            "macro_f1": 0.9757,
            "per_class_recall": {
                "complementario": 0.947,
                "triada": 0.957,
                "analogo": 1.0,
                "monocromatica": 1.0,
            },
            "recall_min": 0.947,
            "confusion": [[18, 0, 0, 1], [1, 22, 0, 0], [0, 0, 16, 0], [0, 0, 0, 22]],
            "nll": 0.1324,
            "ece": 0.0916,
            "n": 80,
        },
        "calibration": {
            "informational": True,
            "nll_raw": 0.1324,
            "ece_raw": 0.0916,
            "t_opt": 0.48,
            "nll_at_t_opt": 0.0717,
            "ece_at_t_opt": 0.0292,
        },
        "gates": [
            run_eval._gate("accuracy", 0.975, mode="min", threshold=0.90),
            run_eval._gate("macro_f1", 0.9757, mode="min", threshold=0.90),
            run_eval._gate("recall/complementario", 0.947, mode="min", threshold=0.85),
        ],
    }


def _failing_classify() -> dict:
    result = _passing_classify()
    result["metrics"]["accuracy"] = 0.40
    result["gates"] = [
        run_eval._gate("accuracy", 0.40, mode="min", threshold=0.90)
    ]
    return result


def test_thresholds_json_is_the_single_source():
    thresholds = _thresholds()
    assert thresholds["extract"] == {
        "delta_e76_mean_max": 5.0,
        "delta_e76_max_max": 20.0,
        "n": 10,
        "seed": 5000,
        "size": 64,
        "jpegQuality": 80,
    }
    assert thresholds["generate"]["ciede2000_mean_max"] == 14.5
    assert thresholds["generate"]["mse_max"] == 0.05
    assert thresholds["generate"]["n"] == 40
    assert thresholds["generate"]["seed"] == 99
    assert thresholds["classify"]["pipeline_accuracy_min"] == 0.90
    assert thresholds["classify"]["pipeline_macro_f1_min"] == 0.90
    assert thresholds["classify"]["pipeline_per_class_recall_min"] == 0.85
    assert thresholds["classify"]["n"] == 80
    assert thresholds["classify"]["seed"] == 20042
    assert thresholds["calibration"] == {
        "informational": False,
        "ece_max": 0.10,
        "t_min": 0.2,
        "t_max": 5.0,
        "require_auc_min": 0.55,
    }


def test_gate_comparisons():
    assert run_eval._gate("x", 5.0, mode="max", threshold=5.0)["passed"]
    assert not run_eval._gate("x", 5.01, mode="max", threshold=5.0)["passed"]
    assert run_eval._gate("x", 0.90, mode="min", threshold=0.90)["passed"]
    assert not run_eval._gate("x", 0.89, mode="min", threshold=0.90)["passed"]


def test_missing_tensorflow_skips_slots_and_exit_codes(monkeypatch):
    monkeypatch.setattr(run_eval, "tensorflow_version", lambda: None)
    manifest = run_eval.load_json(run_eval.DEFAULT_MANIFEST)

    soft = run_eval.build_report(
        thresholds=_thresholds(),
        manifest=manifest,
        slots=["extract", "generate", "classify"],
        check_manifest_enabled=False,
        strict=False,
    )
    assert all(slot["status"] == "skipped" for slot in soft["slots"].values())
    assert soft["summary"]["tensorflowAvailable"] is False
    assert soft["summary"]["exitCode"] == 0

    hard = run_eval.build_report(
        thresholds=_thresholds(),
        manifest=manifest,
        slots=["extract", "generate", "classify"],
        check_manifest_enabled=False,
        strict=True,
    )
    assert hard["summary"]["exitCode"] == 1


def test_report_files_generated_and_false_gate_exits_nonzero(monkeypatch, tmp_path):
    monkeypatch.setattr(run_eval, "tensorflow_version", lambda: "9.9.9")
    monkeypatch.setattr(
        run_eval, "evaluate_classify", lambda thresholds, has_tf: _failing_classify()
    )
    json_path = tmp_path / "eval-report.json"
    md_path = tmp_path / "eval-report.md"

    exit_code = run_eval.main(
        [
            "--slot",
            "classify",
            "--strict",
            "--json",
            str(json_path),
            "--md",
            str(md_path),
        ]
    )

    assert exit_code != 0
    assert json_path.exists() and md_path.exists()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["summary"]["exitCode"] == 1
    assert report["summary"]["gatesFailed"] == 1
    assert "FAIL" in md_path.read_text(encoding="utf-8")


def test_passing_report_exits_zero_and_renders_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(run_eval, "tensorflow_version", lambda: "9.9.9")
    monkeypatch.setattr(
        run_eval, "evaluate_classify", lambda thresholds, has_tf: _passing_classify()
    )
    json_path = tmp_path / "eval-report.json"
    md_path = tmp_path / "eval-report.md"

    exit_code = run_eval.main(
        [
            "--slot",
            "classify",
            "--strict",
            "--json",
            str(json_path),
            "--md",
            str(md_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["summary"]["gatesFailed"] == 0
    assert "PASS" in md_path.read_text(encoding="utf-8")


def test_check_manifest_detects_drift():
    report = {"slots": {"classify": _passing_classify()}}
    manifest = {
        "classify": {
            "active": "classify-v2.keras",
            "files": {
                "classify-v2.keras": {
                    "metrics": {
                        "accuracy": 0.975,
                        "macro_f1": 0.9757,
                        "per_class_recall": {
                            "complementario": 0.947,
                            "triada": 0.957,
                        },
                    }
                }
            },
        }
    }
    ok = run_eval.check_manifest(report, manifest)
    assert ok["ok"] and not ok["drift"]
    assert len(ok["checked"]) == 4

    manifest["classify"]["files"]["classify-v2.keras"]["metrics"]["accuracy"] = 0.50
    drifted = run_eval.check_manifest(report, manifest)
    assert not drifted["ok"]
    assert any(item["metric"] == "accuracy" for item in drifted["drift"])


def test_check_manifest_tolerance_is_ten_percent():
    report = {"slots": {"classify": _passing_classify()}}
    manifest = {
        "classify": {
            "active": "classify-v2.keras",
            "files": {"classify-v2.keras": {"metrics": {"accuracy": 0.90}}},
        }
    }
    # 0.975 vs 0.90 => 8.33% (< 10%) no es drift.
    assert run_eval.check_manifest(report, manifest)["ok"]
    manifest["classify"]["files"]["classify-v2.keras"]["metrics"]["accuracy"] = 1.20
    # 18.75% (> 10%) sí es drift.
    assert not run_eval.check_manifest(report, manifest)["ok"]


def test_classify_test_split_is_disjoint_from_train_and_val():
    splits = benchmarks.classify_split_palettes()
    signatures = {
        name: benchmarks.palette_signatures(palettes)
        for name, palettes in splits.items()
    }
    assert len(signatures["train"]) == benchmarks.CLASSIFY_N_TRAIN
    assert len(signatures["val"]) == benchmarks.CLASSIFY_N_VAL
    assert len(signatures["test"]) == benchmarks.CLASSIFY_N_TEST
    assert signatures["train"].isdisjoint(signatures["test"])
    assert signatures["val"].isdisjoint(signatures["test"])
    assert signatures["train"].isdisjoint(signatures["val"])


def _populate_classify_db(session, n: int = 9) -> None:
    """Crea ``n`` paletas etiquetadas de 5 colores en la DB de entrenamiento."""
    from app.ml.datasets import insert_palette

    labels = ["complementario", "triada", "analogo", "monocromatica"]
    for i in range(n):
        base = i * 23
        hexes = [
            "#%02X%02X%02X" % ((base + j * 17) % 256, (base * 3 + j * 29) % 256, (base * 7 + j * 41) % 256)
            for j in range(5)
        ]
        insert_palette(
            session,
            source="synthetic",
            hexes=hexes,
            harmony_label=labels[i % len(labels)],
        )
    session.commit()


def _populate_classify_db_balanced(session, per_class: int) -> None:
    """DB de entrenamiento con ``per_class`` paletas de 5 colores por etiqueta."""
    from app.ml.datasets import insert_palette

    labels = ["complementario", "triada", "analogo", "monocromatica"]
    for label_idx, label in enumerate(labels):
        for i in range(per_class):
            base = label_idx * 61 + i * 7
            hexes = [
                "#%02X%02X%02X"
                % (
                    (base + j * 19) % 256,
                    (base * 3 + j * 31) % 256,
                    (base * 7 + j * 43) % 256,
                )
                for j in range(5)
            ]
            insert_palette(session, source="synthetic", hexes=hexes, harmony_label=label)
    session.commit()


def _db_label_by_signature() -> dict[tuple[str, ...], str]:
    return {
        tuple(hexv.upper() for hexv in pal["hexes"]): pal["harmony_label"]
        for pal in benchmarks.load_classify_palettes(None)
    }


@pytest.mark.parametrize("per_class", [3, 10])
def test_stratified_partition_is_disjoint_and_balanced(training_db, per_class):
    _populate_classify_db_balanced(training_db, per_class=per_class)

    splits = benchmarks.classify_split_palettes(
        n_train=600, n_val=120, n_test=80
    )
    signatures = {
        name: benchmarks.palette_signatures(palettes)
        for name, palettes in splits.items()
    }
    # Intersecciones vacías entre los tres splits.
    assert signatures["train"].isdisjoint(signatures["val"])
    assert signatures["train"].isdisjoint(signatures["test"])
    assert signatures["val"].isdisjoint(signatures["test"])
    # Todas las paletas de la DB aparecen exactamente una vez.
    db_signatures = benchmarks.palette_signatures(
        benchmarks.load_classify_palettes(None)
    )
    used = signatures["train"] | signatures["val"] | signatures["test"]
    assert db_signatures.issubset(used)
    assert sum(len(db_signatures & signatures[name]) for name in signatures) == len(
        db_signatures
    )
    assert len(db_signatures) == 4 * per_class

    # Estratificación: al menos 1 por clase en cada split y train mayoritario.
    label_by_sig = _db_label_by_signature()
    for name, sigs in signatures.items():
        counts = {
            label: sum(1 for sig in sigs if label_by_sig.get(sig) == label)
            for label in benchmarks.HARMONY_LABELS
        }
        assert all(count >= 1 for count in counts.values()), (name, counts)
    for label in benchmarks.HARMONY_LABELS:
        counts = {
            name: sum(1 for sig in signatures[name] if label_by_sig.get(sig) == label)
            for name in ("train", "val", "test")
        }
        assert counts["train"] >= counts["val"] >= 1
        assert counts["train"] >= counts["test"] >= 1


@pytest.mark.parametrize("test_seed", [43, 7, 10_042, 20_043])
def test_eval_split_test_bucket_disjoint_for_any_test_seed(
    training_db, monkeypatch, test_seed
):
    _populate_classify_db_balanced(training_db, per_class=3)

    captured: dict[str, list] = {}

    def _fake_extract(palettes, seed, *, split="train", cache_dir=None):
        captured["palettes"] = palettes
        return (
            np.zeros((len(palettes), benchmarks.FEATURE_DIM_V2), dtype=np.float32),
            np.zeros((len(palettes), benchmarks.FEATURE_DIM_V1), dtype=np.float32),
            np.zeros((len(palettes), 4), dtype=np.float32),
            {"cache_hit": False, "fallback_extractions": 0},
        )

    monkeypatch.setattr(benchmarks, "extract_pipeline_features", _fake_extract)
    _x54, _x40, _y, palettes, _stats = benchmarks.classify_eval_split(
        n=80, seed=42, test_seed=test_seed, cache_dir=None
    )
    db_signatures = benchmarks.palette_signatures(
        benchmarks.load_classify_palettes(None)
    )
    test_db = benchmarks.palette_signatures(palettes) & db_signatures
    assert test_db  # el bucket de test no está vacío

    train_val = benchmarks.classify_split_palettes(
        n_train=600, n_val=120, n_test=0, seed=42
    )
    train_val_db = benchmarks.palette_signatures(
        train_val["train"] + train_val["val"]
    ) & db_signatures
    assert test_db.isdisjoint(train_val_db)
    # En particular, test_seed=43 no cae en el bucket de validación (antes 43 % 3 == 1).
    val_db = benchmarks.palette_signatures(train_val["val"]) & db_signatures
    assert test_db.isdisjoint(val_db)


def test_extract_cache_invalidates_when_split_palettes_change(monkeypatch, tmp_path):
    import app.services.inference as inference
    from app.schemas.palette import Color, ExtractResponse

    monkeypatch.setattr(
        benchmarks, "extract_cache_tag", lambda: "extract-v1.keras-aaaaaaaa"
    )
    monkeypatch.setattr(
        inference,
        "extract_image_colors",
        lambda image_bytes: ExtractResponse(
            colors=[Color(value="#FF8800", role="predominant", weight=1.0)],
            fallback=False,
        ),
    )
    palettes_a = [
        {"hexes": ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"],
         "harmony_label": "triada"},
        {"hexes": ["#123456", "#ABCDEF", "#FEDCBA", "#445566", "#778899"],
         "harmony_label": "analogo"},
    ]
    palettes_b = [
        {"hexes": ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFE"],
         "harmony_label": "triada"},
        {"hexes": ["#123456", "#ABCDEF", "#FEDCBA", "#445566", "#778899"],
         "harmony_label": "analogo"},
    ]
    assert benchmarks.split_content_hash(palettes_a) != benchmarks.split_content_hash(
        palettes_b
    )
    assert benchmarks.extract_cache_filename(
        "test", len(palettes_a), 20042,
        tag="extract-v1.keras-aaaaaaaa",
        content_hash=benchmarks.split_content_hash(palettes_a),
    ) != benchmarks.extract_cache_filename(
        "test", len(palettes_b), 20042,
        tag="extract-v1.keras-aaaaaaaa",
        content_hash=benchmarks.split_content_hash(palettes_b),
    )

    first = benchmarks.extract_pipeline_features(
        palettes_a, 20042, split="test", cache_dir=tmp_path
    )
    assert first[3]["cache_hit"] is False
    again = benchmarks.extract_pipeline_features(
        palettes_a, 20042, split="test", cache_dir=tmp_path
    )
    assert again[3]["cache_hit"] is True
    assert np.array_equal(first[0], again[0])

    changed = benchmarks.extract_pipeline_features(
        palettes_b, 20042, split="test", cache_dir=tmp_path
    )
    assert changed[3]["cache_hit"] is False  # no reutiliza features de otras paletas
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_classify_splits_disjoint_with_populated_db(training_db):
    """DB cargada una vez y repartida: ninguna paleta de DB cae en dos splits."""
    _populate_classify_db(training_db, n=9)

    splits = benchmarks.classify_split_palettes(n_train=6, n_val=3, n_test=3)
    signatures = {
        name: benchmarks.palette_signatures(palettes)
        for name, palettes in splits.items()
    }
    assert {name: len(values) for name, values in signatures.items()} == {
        "train": 6,
        "val": 3,
        "test": 3,
    }
    assert signatures["train"].isdisjoint(signatures["val"])
    assert signatures["train"].isdisjoint(signatures["test"])
    assert signatures["val"].isdisjoint(signatures["test"])

    db_palettes = benchmarks.load_classify_palettes(None)
    db_signatures = benchmarks.palette_signatures(db_palettes)
    assert len(db_signatures) == 9
    # Las 9 paletas de DB aparecen exactamente una vez entre los tres splits.
    used = signatures["train"] | signatures["val"] | signatures["test"]
    assert db_signatures.issubset(used)
    assert sum(len(db_signatures & signatures[name]) for name in signatures) == 9


def test_eval_test_split_does_not_reuse_train_or_val_db_palettes(
    training_db, monkeypatch
):
    """`classify_eval_split` (pipeline test) usa el bucket de test, no train/val."""
    _populate_classify_db(training_db, n=9)

    captured: dict[str, list] = {}

    def _fake_extract(palettes, seed, *, split="train", cache_dir=None):
        captured["palettes"] = palettes
        return (
            np.zeros((len(palettes), benchmarks.FEATURE_DIM_V2), dtype=np.float32),
            np.zeros((len(palettes), benchmarks.FEATURE_DIM_V1), dtype=np.float32),
            np.zeros((len(palettes), 4), dtype=np.float32),
            {"cache_hit": False, "fallback_extractions": 0},
        )

    monkeypatch.setattr(benchmarks, "extract_pipeline_features", _fake_extract)
    _x54, _x40, _y, palettes, _stats = benchmarks.classify_eval_split(
        n=3, seed=42, test_seed=20042, cache_dir=None
    )
    expected = benchmarks.classify_split_palettes(
        n_train=0, n_val=0, n_test=3, seed=42
    )["test"]
    assert benchmarks.palette_signatures(palettes) == benchmarks.palette_signatures(
        expected
    )
    train_val = benchmarks.classify_split_palettes(
        n_train=3, n_val=3, n_test=0, seed=42
    )
    tv_signatures = benchmarks.palette_signatures(
        train_val["train"] + train_val["val"]
    )
    assert benchmarks.palette_signatures(palettes).isdisjoint(tv_signatures)


def test_extract_cache_key_includes_e_version_and_hash(monkeypatch):
    from pathlib import Path as _Path

    from app.services import model_registry

    def _install(name: str, sha: str):
        monkeypatch.setattr(
            model_registry,
            "load_manifest",
            lambda: {"extract": {"active": name}},
        )
        monkeypatch.setattr(
            model_registry,
            "resolve",
            lambda slot, manifest=None: _Path("models") / name,
        )
        monkeypatch.setattr(model_registry, "sha256_file", lambda path: sha)

    _install("extract-v1.keras", "a" * 64)
    tag_v1 = benchmarks.extract_cache_tag()
    path_v1 = benchmarks.extract_cache_filename("test", 80, 20042)
    assert tag_v1 == "extract-v1.keras-aaaaaaaa"
    assert path_v1.endswith("-e" + tag_v1 + ".npz")

    # Otra versión Y otro hash => otra clave de caché (invalida la anterior).
    _install("extract-v2.keras", "b" * 64)
    assert benchmarks.extract_cache_tag() == "extract-v2.keras-bbbbbbbb"
    assert benchmarks.extract_cache_filename("test", 80, 20042) != path_v1

    _install("extract-v1.keras", "c" * 64)
    assert benchmarks.extract_cache_filename("test", 80, 20042) != path_v1


def test_strict_fails_when_slot_is_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(run_eval, "tensorflow_version", lambda: "9.9.9")
    monkeypatch.setattr(
        run_eval,
        "evaluate_classify",
        lambda thresholds, has_tf: run_eval._skipped("classify", "modelo no cargable"),
    )
    exit_code = run_eval.main(
        ["--slot", "classify", "--strict", "--json", str(tmp_path / "r.json")]
    )
    assert exit_code != 0
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["slots"]["classify"]["status"] == "skipped"
    assert report["summary"]["exitCode"] == 1

    checked = run_eval.build_report(
        thresholds=_thresholds(),
        manifest=run_eval.load_json(run_eval.DEFAULT_MANIFEST),
        slots=["classify"],
        check_manifest_enabled=True,
        strict=True,
    )
    assert checked["checkManifest"]["ok"] is False
    assert any(
        item["slot"] == "classify" and item["metric"] == "status"
        for item in checked["checkManifest"]["drift"]
    )


def test_active_missing_fails_gate_and_reports_resolved_previous(monkeypatch):
    monkeypatch.setattr(run_eval, "tensorflow_version", lambda: "9.9.9")
    monkeypatch.setattr(
        run_eval,
        "evaluate_classify",
        lambda thresholds, has_tf: _passing_classify(),
    )
    manifest = run_eval.load_json(run_eval.DEFAULT_MANIFEST)
    manifest["classify"]["active"] = "classify-v99-missing.keras"
    manifest["classify"]["previous"] = "classify-v1.keras"

    report = run_eval.build_report(
        thresholds=_thresholds(),
        manifest=manifest,
        slots=["classify"],
        check_manifest_enabled=True,
        strict=True,
    )
    info = report["models"]["classify"]
    assert info["activeMissing"] is True
    assert info["resolved"] == "classify-v1.keras"
    assert info["resolvedPrevious"] is True
    assert report["summary"]["exitCode"] == 1
    assert report["checkManifest"]["ok"] is False
    assert any(
        item["metric"] == "model.active" and item["slot"] == "classify"
        for item in report["checkManifest"]["drift"]
    )
    markdown = run_eval.render_markdown(report)
    assert "classify-v1.keras" in markdown
    assert "FALTA active" in markdown


def test_baseline_json_matches_active_models():
    baseline = json.loads(
        (Path(__file__).resolve().parents[1] / "eval" / "baseline.json").read_text()
    )
    manifest = run_eval.load_json(run_eval.DEFAULT_MANIFEST)
    for slot in ("extract", "generate", "classify"):
        entry = baseline["models"][slot]
        assert entry["active"] == manifest[slot]["active"]
        path = run_eval.BACKEND_DIR / "models" / entry["active"]
        assert run_eval.sha256_file(path) == entry["sha256"]


def test_manifest_metadata_has_required_fields():
    manifest = run_eval.load_json(run_eval.DEFAULT_MANIFEST)
    assert manifest["schemaVersion"] == 2
    assert "updatedAt" in manifest
    for slot, expected_run in (("extract", 14), ("generate", 15), ("classify", 17)):
        active = manifest[slot]["active"]
        metadata = manifest[slot]["files"][active]
        assert metadata["trainingRunId"] == expected_run
        for key in ("sha256", "createdAt", "gitSha", "tfVersion", "kerasVersion", "metrics"):
            assert key in metadata
        assert len(metadata["sha256"]) == 64

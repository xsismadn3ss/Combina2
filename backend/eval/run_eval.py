"""CLI de evaluación E/G/C por el camino real de inferencia (issue #22 / B1).

Uso (desde ``backend/``)::

    python -m eval.run_eval
    python -m eval.run_eval --strict --check-manifest --json eval-report.json --md eval-report.md
    python -m eval.run_eval --slot classify

Slots:

- ``extract``: ``inference.extract_image_colors`` sobre 10 mosaicos fijos
  (seeds 5000..5009, 64 px, JPEG 80) comparado con el KMeans clásico.
  Gate: ΔE76 medio ≤ 5.0 y máximo ≤ 20.0.
- ``generate``: predicción cruda de G sobre 40 paletas (seed 99), ΔE00 real
  contra ground truth (emparejamiento posicional, el modelo genera los 5
  slots alineados). Gate: CIEDE2000 medio ≤ 14.5 y MSE ≤ 0.05.
- ``classify``: pipeline E→C sobre el split de test de 80 paletas (seed 20042,
  el mismo usado al entrenar ``classify-v2``). Gates: accuracy ≥ 0.90,
  macro-F1 ≥ 0.90 y recall por clase ≥ 0.85. B2 aplica la temperatura guardada
  en el manifiesto (ajustada en validación) y añade gates ``calibration/ece``
  y ``calibration/nll`` **solo si** ``calibration.calibrated`` es true. La
  T óptima de test se reporta como medición, nunca se ajusta/aplica desde test.
  Para G rechazada por AUC bajo se reporta ``not-applied`` sin fallar gates.

Con ``--strict`` devuelve exit 1 si falla algún gate, si un slot queda
``skipped``/``error`` o si el `active` de un slot falta o su sha256 no
coincide (aunque `resolve` caiga a `previous`). Con ``--check-manifest``
también se marca fallo si un slot no está ``ok`` o si el modelo activo no se
pudo verificar. Sin TensorFlow los slots se marcan como ``skipped``: exit 0
salvo en ``--strict``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = Path(__file__).resolve().parent / "thresholds.json"
DEFAULT_MANIFEST = BACKEND_DIR / "models" / "manifest.json"
MANIFEST_TOLERANCE = 0.10

HARMONY_LABELS = ["complementario", "triada", "analogo", "monocromatica"]

# Familia "buena reconstrucción" para el AUC informativo de G: una paleta se
# considera buena si su CIE76 medio (emparejamiento posicional) ≤ 14.0.
GOOD_RECONSTRUCTION_CIE76 = 14.0


# --- Utilidades ---------------------------------------------------------------


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def tensorflow_version() -> str | None:
    try:
        import tensorflow as tf  # type: ignore[import-not-found]

        return tf.__version__
    except Exception:
        return None


def environment_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _package_version("numpy"),
        "tensorflow": tensorflow_version(),
        "keras": _package_version("keras"),
        "scikit-learn": _package_version("scikit-learn"),
    }


def git_sha() -> str:
    from app.ml.benchmarks import git_sha as _git_sha

    return _git_sha()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def active_model_info(manifest: dict[str, Any]) -> dict[str, Any]:
    """Estado del `active` de cada slot y nombre realmente resuelto.

    ``resolved`` es el archivo que usa la inferencia (`previous` si el activo
    falta o su sha256 no coincide). Las marcas ``activeMissing`` /
    ``activeHashMismatch`` / ``resolvedPrevious`` evitan atribuir métricas del
    `previous` al `active` sin avisar.
    """
    from app.services import model_registry

    models_dir = BACKEND_DIR / "models"
    info: dict[str, Any] = {}
    for slot in ("extract", "generate", "classify"):
        entry = manifest.get(slot, {})
        if not isinstance(entry, dict):
            entry = {}
        active = entry.get("active")
        path = models_dir / active if active else None
        exists = bool(path is not None and path.exists())
        actual_sha = sha256_file(path) if exists else None
        files = entry.get("files") or {}
        metadata = files.get(active) if isinstance(files, dict) and active else None
        expected_sha = (
            metadata.get("sha256") if isinstance(metadata, dict) else None
        )
        hash_ok = (
            exists
            if not expected_sha
            else bool(actual_sha is not None and actual_sha == expected_sha)
        )
        try:
            resolved_path = model_registry.resolve(slot, manifest=manifest)
        except Exception:
            resolved_path = None
        resolved = resolved_path.name if resolved_path is not None else None
        active_missing = bool(active) and not exists
        active_mismatch = bool(active) and exists and not hash_ok
        info[slot] = {
            "active": active,
            "sha256": actual_sha,
            "exists": exists,
            "hashOk": hash_ok,
            "ok": bool(active) and exists and hash_ok,
            "activeMissing": active_missing,
            "activeHashMismatch": active_mismatch,
            "resolved": resolved,
            "resolvedPrevious": bool(
                resolved is not None and active is not None and resolved != active
            ),
        }
    return info


def _gate(
    metric: str, value: float, *, mode: str, threshold: float
) -> dict[str, Any]:
    passed = value <= threshold if mode == "max" else value >= threshold
    return {
        "metric": metric,
        "value": float(value),
        "mode": mode,
        "threshold": float(threshold),
        "passed": bool(passed),
    }


def _skipped(slot: str, reason: str) -> dict[str, Any]:
    return {"status": "skipped", "slot": slot, "reason": reason, "gates": []}


def _denormalize_palette_batch(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1, 5, 3)
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0] * 100.0
    out[..., 1] = arr[..., 1] * 128.0
    out[..., 2] = arr[..., 2] * 128.0
    return out


# --- Slots --------------------------------------------------------------------


def evaluate_extract(thresholds: dict[str, Any], *, has_tf: bool) -> dict[str, Any]:
    from app.ml import benchmarks, metrics

    cfg = thresholds["extract"]
    if not has_tf:
        return _skipped("extract", "TensorFlow no instalado")

    import app.services.inference as inference

    model = inference.load_keras_model("extract")
    if model is None:
        return _skipped("extract", "sin modelo E activo o no cargable")

    from app.ml.color_spaces import ColorConverter
    from app.services.classic import ClassicPaletteIdentifier

    classic = ClassicPaletteIdentifier()
    cases = benchmarks.extract_eval_cases(
        size=int(cfg.get("size", 64)),
        jpeg_quality=int(cfg.get("jpegQuality", 80)),
        seed0=int(cfg.get("seed", 5000)),
    )
    deltas: list[float] = []
    per_case: list[dict[str, Any]] = []
    fallbacks = 0
    for case in cases:
        reference = classic.identify_palette(case["image"])
        ref_lab = np.array([ColorConverter.hex_to_lab(c.value) for c in reference])
        result = inference.extract_image_colors(case["image"])
        fallbacks += int(bool(result.fallback))
        got_lab = np.array([ColorConverter.hex_to_lab(c.value) for c in result.colors])
        delta = metrics.palette_delta_e(got_lab, ref_lab, metric="cie76")
        deltas.append(delta)
        per_case.append(
            {
                "seed": case["seed"],
                "delta_e76": round(delta, 6),
                "fallback": bool(result.fallback),
                "n_colors": len(result.colors),
            }
        )

    mean_de = float(np.mean(deltas))
    max_de = float(np.max(deltas))
    gates = [
        _gate(
            "delta_e76_mean",
            mean_de,
            mode="max",
            threshold=float(cfg["delta_e76_mean_max"]),
        ),
        _gate(
            "delta_e76_max",
            max_de,
            mode="max",
            threshold=float(cfg["delta_e76_max_max"]),
        ),
    ]
    return {
        "status": "ok",
        "slot": "extract",
        "metrics": {
            "delta_e76_mean": mean_de,
            "delta_e76_max": max_de,
            "fallback_count": fallbacks,
            "n": len(cases),
        },
        "gates": gates,
        "cases": per_case,
    }


def evaluate_generate(thresholds: dict[str, Any], *, has_tf: bool) -> dict[str, Any]:
    from app.ml import benchmarks, metrics

    cfg = thresholds["generate"]
    if not has_tf:
        return _skipped("generate", "TensorFlow no instalado")

    import app.services.inference as inference

    model = inference.load_keras_model("generate")
    if model is None:
        return _skipped("generate", "sin modelo G activo o no cargable")

    (x_lab, x_mask, x_harm), (y_pal, _y_score) = benchmarks.generate_eval_dataset(
        n=int(cfg.get("n", 40)), seed=int(cfg.get("seed", 99))
    )
    pred_pal, pred_score = model.predict([x_lab, x_mask, x_harm], verbose=0)
    pred_pal = np.asarray(pred_pal, dtype=np.float64)
    true_lab = _denormalize_palette_batch(y_pal)
    pred_lab = _denormalize_palette_batch(pred_pal)

    per_color_cie76 = metrics.delta_e76(pred_lab, true_lab)
    per_color_cie00 = metrics.ciede2000(pred_lab, true_lab)
    per_palette_cie76 = per_color_cie76.mean(axis=1)
    per_palette_cie00 = per_color_cie00.mean(axis=1)

    ciede_mean = float(np.mean(per_palette_cie00))
    cie76_mean = float(np.mean(per_palette_cie76))
    mse = float(np.mean((np.asarray(y_pal, dtype=np.float64) - pred_pal) ** 2))
    scores = np.asarray(pred_score, dtype=np.float64).reshape(-1)
    good = per_palette_cie76 <= GOOD_RECONSTRUCTION_CIE76
    auc = metrics.roc_auc(scores, good)

    calibration_meta = inference.load_calibration("generate")
    applied = (
        calibration_meta.get("method") == "platt"
        and calibration_meta.get("calibrated") is True
    )
    if applied:
        from app.ml.calibration import apply_platt

        try:
            a = float(calibration_meta["a"])
            b = float(calibration_meta["b"])
            calibrated_scores = np.asarray(
                apply_platt(scores, a, b), dtype=np.float64
            )
            calibration = {
                "applied": True,
                "method": "platt",
                "params": {"a": a, "b": b},
                "auc_raw": float(auc),
                "auc_calibrated": float(metrics.roc_auc(calibrated_scores, good)),
                "reason": None,
            }
        except (KeyError, TypeError, ValueError):
            applied = False
    if not applied:
        calibration = {
            "applied": False,
            "method": calibration_meta.get("method", "none"),
            "params": None,
            "auc_raw": float(auc),
            "reason": calibration_meta.get("reason") or "not_calibrated",
            "requireAucMin": float(
                thresholds.get("calibration", {}).get("require_auc_min", 0.55)
            ),
        }

    gates = [
        _gate(
            "ciede2000_mean",
            ciede_mean,
            mode="max",
            threshold=float(cfg["ciede2000_mean_max"]),
        ),
        _gate("mse", mse, mode="max", threshold=float(cfg["mse_max"])),
    ]
    return {
        "status": "ok",
        "slot": "generate",
        "metrics": {
            "ciede2000_mean": ciede_mean,
            "cie76_mean": cie76_mean,
            "mse": mse,
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "auc_good_reconstruction": float(auc),
            "good_reconstruction_cie76_max": GOOD_RECONSTRUCTION_CIE76,
            "n": int(len(per_palette_cie00)),
        },
        "calibration": calibration,
        "gates": gates,
    }


def evaluate_classify(thresholds: dict[str, Any], *, has_tf: bool) -> dict[str, Any]:
    from app.ml import benchmarks, metrics

    cfg = thresholds["classify"]
    if not has_tf:
        return _skipped("classify", "TensorFlow no instalado")

    import app.services.inference as inference

    model = inference.load_keras_model("classify")
    if model is None:
        return _skipped("classify", "sin modelo C activo o no cargable")

    x54, x40, y, _palettes, stats = benchmarks.classify_eval_split(
        n=int(cfg.get("n", 80)),
        seed=int(cfg.get("splitSeed", 42)),
        test_seed=int(cfg.get("seed", 20042)),
    )
    input_dim = inference._classify_input_dim(model) or 54
    x = x54 if input_dim == 54 else x40

    probs = np.asarray(model.predict(x, verbose=0), dtype=np.float64)
    true = np.argmax(y, axis=1)
    pred = np.argmax(probs, axis=1)
    cm = metrics.confusion_matrix(true, pred, n_classes=len(HARMONY_LABELS))
    report = metrics.per_class_report(
        true, pred, labels=HARMONY_LABELS, n_classes=len(HARMONY_LABELS)
    )
    accuracy = float(np.mean(pred == true)) if len(true) else 0.0
    macro = metrics.macro_f1(
        true, pred, labels=HARMONY_LABELS, n_classes=len(HARMONY_LABELS)
    )
    recall_min = float(min(entry["recall"] for entry in report.values()))

    calibration_cfg = thresholds.get("calibration", {})
    t_min = float(calibration_cfg.get("t_min", 0.2))
    t_max = float(calibration_cfg.get("t_max", 5.0))
    ece_max = float(calibration_cfg.get("ece_max", 0.10))
    t_opt, nll_t = metrics.optimal_temperature(
        probs, true, t_min=t_min, t_max=t_max
    )
    ece_raw = metrics.ece(probs, true)
    nll_raw = metrics.nll(probs, true)
    ece_t = metrics.ece(metrics.apply_temperature(probs, t_opt), true)

    # Calibración APLICADA: parámetros del manifiesto (ajustados en validación
    # por scripts/calibrate.py), nunca reajustados sobre test.
    calibration_meta = inference.load_calibration("classify")
    applied = (
        calibration_meta.get("method") == "temperature"
        and calibration_meta.get("calibrated") is True
    )
    calibration_gates: list[dict[str, Any]] = []
    ece_after: float | None = None
    nll_after: float | None = None
    params: dict[str, Any] | None = None
    if applied:
        try:
            temperature = float(calibration_meta["temperature"])
            if not np.isfinite(temperature) or temperature <= 0.0:
                raise ValueError("temperatura inválida")
            probs_cal = metrics.apply_temperature(probs, temperature)
            ece_after = metrics.ece(probs_cal, true)
            nll_after = metrics.nll(probs_cal, true)
            params = {"temperature": temperature}
            calibration_gates = [
                _gate("calibration/ece", ece_after, mode="max", threshold=ece_max),
                _gate("calibration/nll", nll_after, mode="max", threshold=nll_raw),
            ]
        except (KeyError, TypeError, ValueError):
            applied = False
    calibration = {
        "informational": bool(calibration_cfg.get("informational", True)),
        "applied": bool(applied),
        "method": calibration_meta.get("method", "none"),
        "params": params,
        "reason": calibration_meta.get("reason"),
        "nll_raw": nll_raw,
        "ece_raw": ece_raw,
        "t_opt": t_opt,
        "nll_at_t_opt": nll_t,
        "ece_at_t_opt": ece_t,
        "ece_max": ece_max,
        "nll_before": nll_raw if applied else None,
        "ece_before": ece_raw if applied else None,
        "nll_after": nll_after,
        "ece_after": ece_after,
        "t_range": [t_min, t_max],
        "require_auc_min": float(calibration_cfg.get("require_auc_min", 0.55)),
        "gates": calibration_gates,
    }

    gates = [
        _gate(
            "accuracy",
            accuracy,
            mode="min",
            threshold=float(cfg["pipeline_accuracy_min"]),
        ),
        _gate(
            "macro_f1",
            macro,
            mode="min",
            threshold=float(cfg["pipeline_macro_f1_min"]),
        ),
    ]
    recall_floor = float(cfg["pipeline_per_class_recall_min"])
    for label in HARMONY_LABELS:
        gates.append(
            _gate(
                f"recall/{label}",
                float(report[label]["recall"]),
                mode="min",
                threshold=recall_floor,
            )
        )
    # Los gates de calibración solo existen si hay calibración APLICADA.
    gates.extend(calibration_gates)

    return {
        "status": "ok",
        "slot": "classify",
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro,
            "per_class_recall": {
                label: float(report[label]["recall"]) for label in HARMONY_LABELS
            },
            "recall_min": recall_min,
            "confusion": cm.tolist(),
            "nll": nll_raw,
            "ece": ece_raw,
            "t_opt": t_opt,
            "n": int(len(true)),
            "feature_dim": int(input_dim),
            "fallback_extractions": int(stats.get("fallback_extractions", 0)),
        },
        "calibration": calibration,
        "gates": gates,
    }


# --- Manifiesto ---------------------------------------------------------------


def _compare_metric(
    slot: str, metric: str, measured: float, expected: float, drift: list[dict]
) -> None:
    try:
        measured_f = float(measured)
        expected_f = float(expected)
    except (TypeError, ValueError):
        drift.append(
            {
                "slot": slot,
                "metric": metric,
                "expected": expected,
                "measured": measured,
                "reason": "no numérico",
            }
        )
        return
    if not np.isfinite(measured_f) or not np.isfinite(expected_f):
        return
    denom = abs(expected_f) if abs(expected_f) > 1e-9 else 1.0
    relative = abs(measured_f - expected_f) / denom
    if relative > MANIFEST_TOLERANCE:
        drift.append(
            {
                "slot": slot,
                "metric": metric,
                "expected": expected_f,
                "measured": measured_f,
                "relative_diff": float(relative),
                "tolerance": MANIFEST_TOLERANCE,
            }
        )


def check_manifest(
    report: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Compara métricas medidas vs metadatos del manifiesto (tolerancia 10%).

    Un slot que no está ``ok`` (``skipped``/``error``) cuenta como fallo: no
    hay métricas verificables. Un `active` ausente o con hash incorrecto
    también falla aunque ``resolve`` haya caído a ``previous``.
    """
    drift: list[dict[str, Any]] = []
    checked: list[str] = []
    for slot, model_info in (report.get("models") or {}).items():
        if model_info.get("ok"):
            continue
        reason = "active ausente" if model_info.get("activeMissing") else (
            "sha256 del active no coincide"
            if model_info.get("activeHashMismatch")
            else "sin modelo activo"
        )
        drift.append(
            {
                "slot": slot,
                "metric": "model.active",
                "expected": "verificado",
                "measured": model_info.get("resolved") or "none",
                "reason": reason,
                "active": model_info.get("active"),
            }
        )
    for slot, slot_report in report["slots"].items():
        if slot_report.get("status") != "ok":
            drift.append(
                {
                    "slot": slot,
                    "metric": "status",
                    "expected": "ok",
                    "measured": slot_report.get("status"),
                    "reason": slot_report.get("reason"),
                }
            )
            continue
        active = manifest.get(slot, {}).get("active")
        metadata = (
            manifest.get(slot, {}).get("files", {}).get(active, {}) if active else {}
        )
        expected = metadata.get("metrics") if isinstance(metadata, dict) else None
        if not isinstance(expected, dict):
            continue
        measured = slot_report.get("metrics", {})
        for key, expected_value in expected.items():
            if isinstance(expected_value, dict):
                measured_nested = measured.get(key)
                if not isinstance(measured_nested, dict):
                    continue
                for sub_key, sub_expected in expected_value.items():
                    if sub_key in measured_nested:
                        _compare_metric(
                            slot,
                            f"{key}.{sub_key}",
                            measured_nested[sub_key],
                            sub_expected,
                            drift,
                        )
                        checked.append(f"{slot}.{key}.{sub_key}")
            elif key in measured:
                _compare_metric(slot, key, measured[key], expected_value, drift)
                checked.append(f"{slot}.{key}")
    return {
        "enabled": True,
        "tolerance": MANIFEST_TOLERANCE,
        "checked": checked,
        "drift": drift,
        "ok": not drift,
    }


# --- Reporte ------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Combina2 — evaluación E/G/C")
    lines.append("")
    lines.append(f"- Generado: `{report['generatedAt']}`")
    lines.append(f"- Git: `{report['gitSha']}`")
    env = report["environment"]
    lines.append(
        "- Entorno: Python "
        f"`{env['python']}` · NumPy `{env['numpy']}` · TF `{env['tensorflow']}` "
        f"· Keras `{env['keras']}` · sklearn `{env['scikit-learn']}`"
    )
    lines.append("")
    lines.append("## Modelos activos")
    lines.append("")
    lines.append("| Slot | Archivo | sha256 | Resuelto | Estado |")
    lines.append("|---|---|---|---|---|")
    for slot, info in report["models"].items():
        sha = info.get("sha256") or "—"
        resolved = info.get("resolved") or "—"
        if info.get("ok"):
            state = "OK"
        elif info.get("activeMissing"):
            state = "**FALTA active**"
        elif info.get("activeHashMismatch"):
            state = "**HASH active**"
        else:
            state = "**SIN active**"
        if info.get("resolvedPrevious"):
            state += " (usa previous)"
        lines.append(
            f"| {slot} | `{info.get('active')}` | `{sha[:16]}…` | "
            f"`{resolved}` | {state} |"
        )
    lines.append("")

    for slot in ("extract", "generate", "classify"):
        data = report["slots"].get(slot)
        if data is None:
            continue
        lines.append(f"## {slot}")
        lines.append("")
        if data.get("status") != "ok":
            lines.append(f"**SKIPPED**: {data.get('reason', '')}")
            lines.append("")
            continue
        lines.append("| Métrica | Valor | Umbral | Estado |")
        lines.append("|---|---|---|---|")
        for gate in data.get("gates", []):
            comparator = "<=" if gate["mode"] == "max" else ">="
            state = "PASS" if gate["passed"] else "FAIL"
            lines.append(
                f"| `{gate['metric']}` | {_fmt(gate['value'])} | "
                f"{comparator} {_fmt(gate['threshold'])} | **{state}** |"
            )
        lines.append("")
        metrics = data.get("metrics", {})
        if slot == "extract":
            lines.append(
                f"- ΔE76 medio/máx: `{_fmt(metrics['delta_e76_mean'])}` / "
                f"`{_fmt(metrics['delta_e76_max'])}`; fallbacks: "
                f"`{metrics['fallback_count']}/{metrics['n']}`"
            )
        elif slot == "generate":
            lines.append(
                f"- CIEDE2000 medio `{_fmt(metrics['ciede2000_mean'])}` · "
                f"CIE76 medio `{_fmt(metrics['cie76_mean'])}` · MSE "
                f"`{_fmt(metrics['mse'], 6)}`"
            )
            lines.append(
                f"- Score: media `{_fmt(metrics['score_mean'])}`, std "
                f"`{_fmt(metrics['score_std'])}`, rango "
                f"[`{_fmt(metrics['score_min'])}`, `{_fmt(metrics['score_max'])}`]"
            )
            lines.append(
                "- AUC(score->buena reconstrucción, CIE76 <= "
                f"{_fmt(metrics['good_reconstruction_cie76_max'], 1)}) = "
                f"`{_fmt(metrics['auc_good_reconstruction'])}`"
            )
            cal = data.get("calibration", {})
            if cal.get("applied"):
                params = cal.get("params") or {}
                lines.append(
                    "- Calibración Platt APLICADA: a = "
                    f"`{_fmt(params.get('a'), 6)}`, b = `{_fmt(params.get('b'), 6)}`"
                )
            else:
                lines.append(
                    "- Calibración: **no aplicada** — motivo "
                    f"`{cal.get('reason') or 'not_calibrated'}` "
                    f"(AUC mínimo {_fmt(cal.get('requireAucMin', 0.55), 2)}; "
                    "rechazada sin fallar el gate)."
                )
        elif slot == "classify":
            lines.append(
                f"- accuracy `{_fmt(metrics['accuracy'])}` · macro-F1 "
                f"`{_fmt(metrics['macro_f1'])}` · NLL `{_fmt(metrics['nll'])}` "
                f"· ECE `{_fmt(metrics['ece'])}`"
            )
            recalls = ", ".join(
                f"{label} `{_fmt(value)}`"
                for label, value in metrics["per_class_recall"].items()
            )
            lines.append(f"- Recall por clase: {recalls}")
            lines.append("- Confusión (filas = verdad):")
            lines.append("")
            lines.append("```")
            for row in metrics["confusion"]:
                lines.append("[" + ", ".join(f"{int(v):2d}" for v in row) + "]")
            lines.append("```")
            cal = data.get("calibration", {})
            lines.append(
                "- Medición en test: "
                f"NLL cruda `{_fmt(cal.get('nll_raw'))}`, ECE cruda "
                f"`{_fmt(cal.get('ece_raw'))}`; T óptima (solo medición) "
                f"`{_fmt(cal.get('t_opt'), 2)}` -> NLL "
                f"`{_fmt(cal.get('nll_at_t_opt'))}`, ECE "
                f"`{_fmt(cal.get('ece_at_t_opt'))}`"
            )
            if cal.get("applied"):
                params = cal.get("params") or {}
                lines.append(
                    "- Calibración APLICADA (T ajustada en validación): "
                    f"T = `{_fmt(params.get('temperature'), 6)}`; "
                    f"ECE `{_fmt(cal.get('ece_before'))}` -> "
                    f"`{_fmt(cal.get('ece_after'))}`, NLL "
                    f"`{_fmt(cal.get('nll_before'))}` -> "
                    f"`{_fmt(cal.get('nll_after'))}` "
                    f"(gate ECE ≤ `{_fmt(cal.get('ece_max'), 2)}` y NLL no peor)."
                )
            else:
                lines.append(
                    "- Calibración: **no aplicada** — motivo "
                    f"`{cal.get('reason') or 'not_calibrated'}`."
                )
        lines.append("")

    check = report.get("checkManifest")
    if check:
        lines.append("## Manifiesto")
        lines.append("")
        if check.get("ok"):
            lines.append(
                f"OK: {len(check.get('checked', []))} métricas comparadas "
                f"(tolerancia {int(check['tolerance'] * 100)}%)."
            )
        else:
            lines.append(
                f"**DRIFT** en {len(check['drift'])} métricas "
                f"(tolerancia {int(check['tolerance'] * 100)}%):"
            )
            for item in check["drift"]:
                lines.append(
                    f"- `{item['slot']}.{item['metric']}`: manifiesto "
                    f"`{_fmt(item.get('expected'))}` vs medido "
                    f"`{_fmt(item.get('measured'))}`"
                )
        lines.append("")

    summary = report["summary"]
    lines.append("## Resumen")
    lines.append("")
    lines.append(
        f"- Gates: {summary['gatesPassed']}/{summary['gatesTotal']} PASS"
        + (
            f" · **{summary['gatesFailed']} FALLAN**"
            if summary["gatesFailed"]
            else ""
        )
    )
    lines.append(f"- Exit code: `{summary['exitCode']}`")
    lines.append("")
    return "\n".join(lines)


# --- CLI ----------------------------------------------------------------------


def build_report(
    *,
    thresholds: dict[str, Any],
    manifest: dict[str, Any],
    slots: list[str],
    check_manifest_enabled: bool,
    strict: bool,
) -> dict[str, Any]:
    has_tf = tensorflow_version() is not None
    evaluators = {
        "extract": evaluate_extract,
        "generate": evaluate_generate,
        "classify": evaluate_classify,
    }
    results: dict[str, Any] = {}
    for slot in slots:
        try:
            results[slot] = evaluators[slot](thresholds, has_tf=has_tf)
        except Exception as exc:  # un slot roto no oculta los demás
            results[slot] = {
                "status": "error",
                "slot": slot,
                "reason": f"{type(exc).__name__}: {exc}",
                "gates": [],
            }

    gates = [gate for result in results.values() for gate in result.get("gates", [])]
    gates_failed = sum(1 for gate in gates if not gate["passed"])

    models = active_model_info(manifest)
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "gitSha": git_sha(),
        "environment": environment_info(),
        "models": models,
        "thresholds": thresholds,
        "slots": results,
        "summary": {
            "strict": strict,
            "tensorflowAvailable": has_tf,
            "gatesTotal": len(gates),
            "gatesPassed": len(gates) - gates_failed,
            "gatesFailed": gates_failed,
            "exitCode": 0,
        },
    }

    check = None
    if check_manifest_enabled:
        check = check_manifest(report, manifest)
    report["checkManifest"] = check

    slot_not_ok = any(
        result.get("status") != "ok" for result in results.values()
    )
    bad_models = any(not info.get("ok") for info in models.values())
    blocked = (
        gates_failed > 0
        or any(result.get("status") == "error" for result in results.values())
        # En --strict un slot skipped (aunque TF esté instalado) es fallo.
        or (strict and slot_not_ok)
        # Un active ausente/hash incorrecto es fallo de gate en --strict.
        or (strict and bad_models)
        or (check is not None and not check["ok"])
        or (not has_tf and strict)
    )
    report["summary"]["exitCode"] = 1 if strict and blocked else 0
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 si falla un gate, hay error de slot o drift de manifiesto",
    )
    parser.add_argument(
        "--check-manifest",
        action="store_true",
        help="Compara métricas medidas vs metadatos del manifiesto (tolerancia 10%%)",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="eval-report.json",
        default=None,
        help="Escribe el reporte JSON (por defecto eval-report.json)",
    )
    parser.add_argument(
        "--md",
        nargs="?",
        const="eval-report.md",
        default=None,
        help="Escribe el reporte Markdown (por defecto eval-report.md)",
    )
    parser.add_argument(
        "--slot",
        action="append",
        choices=("extract", "generate", "classify"),
        help="Evalúa solo el slot indicado (repetible)",
    )
    parser.add_argument(
        "--thresholds",
        default=str(DEFAULT_THRESHOLDS),
        help="Ruta de thresholds.json",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Ruta de manifest.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Consolas Windows (cp1252) no soportan algunos símbolos del reporte.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = parse_args(argv)
    thresholds = load_json(Path(args.thresholds))
    manifest = load_json(Path(args.manifest))
    slots = args.slot or ["extract", "generate", "classify"]

    report = build_report(
        thresholds=thresholds,
        manifest=manifest,
        slots=slots,
        check_manifest_enabled=args.check_manifest,
        strict=args.strict,
    )

    markdown = render_markdown(report)
    print(markdown)
    if report["slots"] and not report["summary"]["tensorflowAvailable"]:
        print(
            "AVISO: TensorFlow no está instalado; slots ML omitidos "
            "(en --strict esto cuenta como fallo).",
            file=sys.stderr,
        )

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[eval] JSON escrito en {args.json}", file=sys.stderr)
    if args.md:
        Path(args.md).write_text(markdown + "\n", encoding="utf-8")
        print(f"[eval] Markdown escrito en {args.md}", file=sys.stderr)

    return int(report["summary"]["exitCode"])


if __name__ == "__main__":
    raise SystemExit(main())

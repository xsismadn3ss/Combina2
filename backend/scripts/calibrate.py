"""Calibración de las redes C y G (B2 del issue #22).

Uso (desde ``backend/``)::

    .venv-tf/Scripts/python.exe scripts/calibrate.py --slot all
    .venv-tf/Scripts/python.exe scripts/calibrate.py --slot classify --dry-run
    .venv-tf/Scripts/python.exe scripts/calibrate.py --slot generate --source auto

Comportamiento:

- ``classify``: ajusta la temperatura **solo con el split de validación**
  (120 paletas, seed 10042; nunca test), mide ECE/NLL antes/después y escribe
  ``calibration`` en el manifiesto para el ``.keras`` activo. Test se usa
  únicamente como referencia de reporte (la T no se ajusta ahí).
- ``generate``: intenta leer feedback real (``feedback`` con etiqueta) y, si
  hay menos de 50 filas, usa el proxy de reconstrucción de G (40 paletas seed
  99, buena = CIE76 medio ≤ 14). Si el AUC score→calidad es menor que
  ``calibration.require_auc_min`` (0.55), **rechaza** la calibración Platt:
  deja ``calibrated: false`` con motivo y no toca la inferencia. El score de
  ``generate-v1`` está anti-correlacionado (AUC 0.3516) y no se invierte.
- ``--dry-run``: imprime todo pero no escribe el manifiesto.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.calibration import (  # noqa: E402
    DEFAULT_MIN_AUC,
    apply_platt,
    apply_temperature,
    auc_score,
    ece,
    fit_temperature,
    nll,
    platt_fit,
    risk_coverage,
)

DEFAULT_MANIFEST = BACKEND_DIR / "models" / "manifest.json"
DEFAULT_THRESHOLDS = BACKEND_DIR / "eval" / "thresholds.json"
MIN_FEEDBACK_ROWS = 50
RECOMMENDED_MIN_CONFIDENCE = 0.70
GOOD_RECONSTRUCTION_CIE76 = 14.0
CLASSIFY_VAL_SEED_OFFSET = 10_000


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updatedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _round(value: float, digits: int = 6) -> float:
    return float(round(float(value), digits))


def _print_risk_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"  {title}")
    print("    umbral  cobertura  accuracy  n")
    for row in rows:
        accuracy = row.get("accuracy")
        acc_text = "—" if accuracy is None else f"{float(accuracy):.4f}"
        print(
            f"    {float(row['threshold']):>6.2f}  {float(row['coverage']):>9.4f}"
            f"  {acc_text:>8}  {int(row.get('n', 0)):>3}"
        )


def _recommended_threshold(
    rows: list[dict[str, Any]], *, min_coverage: float = 0.90
) -> float:
    """Mayor umbral que conserva la accuracy máxima con cobertura ≥ 90%.

    Es una política conservadora: filtra lo máximo posible sin perder acierto
    respecto de los umbrales bajos ni rechazar más de ~10% de las fotos.
    """
    scored = [
        row
        for row in rows
        if row.get("accuracy") is not None and float(row["coverage"]) >= min_coverage
    ]
    if not scored:
        return RECOMMENDED_MIN_CONFIDENCE
    best = max(float(row["accuracy"]) for row in scored)
    for row in reversed(scored):
        if float(row["accuracy"]) >= best - 1e-12:
            return float(row["threshold"])
    return RECOMMENDED_MIN_CONFIDENCE


# --- C ------------------------------------------------------------------------


def _classify_probs(model: Any, x54: np.ndarray, x40: np.ndarray) -> np.ndarray:
    import app.services.inference as inference

    input_dim = inference._classify_input_dim(model) or 54
    x = x54 if input_dim == 54 else x40
    return np.asarray(model.predict(x, verbose=0), dtype=np.float64)


def calibrate_classify(
    manifest: dict[str, Any],
    *,
    dry_run: bool,
    t_min: float,
    t_max: float,
    manifest_path: Path,
) -> dict[str, Any]:
    from app.ml import benchmarks
    import app.services.inference as inference

    model = inference.load_keras_model("classify")
    if model is None:
        raise RuntimeError("no hay red C activa o TensorFlow no está disponible")

    seed = benchmarks.CLASSIFY_SPLIT_SEED
    n_val = benchmarks.CLASSIFY_N_VAL
    # Misma partición disjunta que train/test: la DB se carga una vez y val
    # recibe su propio bucket (nunca paletas de train o test).
    val_palettes = benchmarks.classify_split_palettes(
        n_train=0, n_val=n_val, n_test=0, seed=seed
    )["val"]
    x54, x40, y_val, _stats = benchmarks.extract_pipeline_features(
        val_palettes, seed + CLASSIFY_VAL_SEED_OFFSET, split="val"
    )
    probs_val = _classify_probs(model, x54, x40)
    true_val = np.argmax(y_val, axis=1)

    temperature = fit_temperature(probs_val, true_val, t_min=t_min, t_max=t_max)
    probs_val_cal = apply_temperature(probs_val, temperature)
    val_before = {
        "ece": ece(probs_val, true_val),
        "nll": nll(probs_val, true_val),
        "accuracy": float(np.mean(probs_val.argmax(1) == true_val)),
        "n": int(len(true_val)),
    }
    val_after = {
        "ece": ece(probs_val_cal, true_val),
        "nll": nll(probs_val_cal, true_val),
        "accuracy": float(np.mean(probs_val_cal.argmax(1) == true_val)),
        "n": int(len(true_val)),
    }

    # Referencia de reporte (NO se ajusta con test).
    xt54, xt40, y_test, _pals, _stats_test = benchmarks.classify_eval_split(
        n=benchmarks.CLASSIFY_N_TEST,
        seed=seed,
        test_seed=seed + benchmarks.CLASSIFY_TEST_SEED_OFFSET,
    )
    probs_test = _classify_probs(model, xt54, xt40)
    true_test = np.argmax(y_test, axis=1)
    probs_test_cal = apply_temperature(probs_test, temperature)
    test_before = {
        "ece": ece(probs_test, true_test),
        "nll": nll(probs_test, true_test),
        "accuracy": float(np.mean(probs_test.argmax(1) == true_test)),
        "n": int(len(true_test)),
    }
    test_after = {
        "ece": ece(probs_test_cal, true_test),
        "nll": nll(probs_test_cal, true_test),
        "accuracy": float(np.mean(probs_test_cal.argmax(1) == true_test)),
        "n": int(len(true_test)),
    }
    coverage_rows = risk_coverage(
        probs_test_cal, true_test, [0.5, 0.6, 0.7, 0.8, 0.9]
    )
    recommended = _recommended_threshold(coverage_rows)

    print(f"[calibrate] C: T ajustada en validación = {temperature:.6f}")
    print(
        "  val  antes: ECE {ece:.4f}  NLL {nll:.4f}  acc {acc:.4f} (n={n})".format(
            ece=val_before["ece"],
            nll=val_before["nll"],
            acc=val_before["accuracy"],
            n=val_before["n"],
        )
    )
    print(
        "  val  después: ECE {ece:.4f}  NLL {nll:.4f}  acc {acc:.4f}".format(
            ece=val_after["ece"],
            nll=val_after["nll"],
            acc=val_after["accuracy"],
        )
    )
    print(
        "  test antes: ECE {ece:.4f}  NLL {nll:.4f}  acc {acc:.4f} (n={n})".format(
            ece=test_before["ece"],
            nll=test_before["nll"],
            acc=test_before["accuracy"],
            n=test_before["n"],
        )
    )
    print(
        "  test después: ECE {ece:.4f}  NLL {nll:.4f}  acc {acc:.4f}".format(
            ece=test_after["ece"],
            nll=test_after["nll"],
            acc=test_after["accuracy"],
        )
    )
    _print_risk_table("risk-coverage en test con la T de validación:", coverage_rows)
    print(
        "  umbral recomendado para IDENTIFY_MIN_CONFIDENCE: "
        f"{recommended:.2f} (máxima accuracy con máxima cobertura)."
    )

    calibration = {
        "method": "temperature",
        "calibrated": True,
        "temperature": _round(temperature),
        "fittedOn": "pipeline-val",
        "n": int(n_val),
        "eceBefore": _round(val_before["ece"]),
        "eceAfter": _round(val_after["ece"]),
        "nllBefore": _round(val_before["nll"]),
        "nllAfter": _round(val_after["nll"]),
        "range": [float(t_min), float(t_max)],
    }

    active = manifest.get("classify", {}).get("active")
    metadata = (
        manifest.get("classify", {}).get("files", {}).get(active) if active else None
    )
    if not isinstance(metadata, dict):
        raise RuntimeError("el manifiesto no tiene metadatos para la C activa")

    if dry_run:
        print("[calibrate] dry-run: no se escribe el manifiesto")
    else:
        metadata["calibration"] = calibration
        _write_manifest(manifest_path, manifest)
        print(f"[calibrate] manifiesto actualizado: classify/{active}")

    return {
        "slot": "classify",
        "active": active,
        "temperature": temperature,
        "validation": {"before": val_before, "after": val_after},
        "testReference": {"before": test_before, "after": test_after},
        "riskCoverage": coverage_rows,
        "recommendedMinConfidence": recommended,
        "calibration": calibration,
        "applied": not dry_run,
    }


# --- G ------------------------------------------------------------------------


def _denormalize_palette_batch(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1, 5, 3)
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0] * 100.0
    out[..., 1] = arr[..., 1] * 128.0
    out[..., 2] = arr[..., 2] * 128.0
    return out


def _reconstruction_samples() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from app.ml import benchmarks, metrics
    import app.services.inference as inference

    model = inference.load_keras_model("generate")
    if model is None:
        raise RuntimeError("no hay red G activa o TensorFlow no está disponible")
    (x_lab, x_mask, x_harm), (y_pal, _y_score) = benchmarks.generate_eval_dataset(
        n=benchmarks.GENERATE_EVAL_N, seed=benchmarks.GENERATE_EVAL_SEED
    )
    pred_pal, pred_score = model.predict([x_lab, x_mask, x_harm], verbose=0)
    pred_pal = np.asarray(pred_pal, dtype=np.float64)
    scores = np.asarray(pred_score, dtype=np.float64).reshape(-1)
    per_palette = metrics.delta_e76(
        _denormalize_palette_batch(pred_pal), _denormalize_palette_batch(y_pal)
    ).mean(axis=1)
    good = per_palette <= GOOD_RECONSTRUCTION_CIE76
    stats = {
        "n": int(len(scores)),
        "cie76_mean": float(per_palette.mean()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "good_rate": float(good.mean()),
    }
    return scores, good, stats


def _feedback_rows() -> list[Any]:
    from app.db.sqlite import get_connection

    try:
        conn = get_connection()
    except Exception:
        return []
    try:
        return conn.execute(
            "SELECT accepted, score, palette_json FROM feedback "
            "WHERE accepted IS NOT NULL ORDER BY id"
        ).fetchall()
    except Exception:
        return []
    finally:
        conn.close()


def _g_scores_for_feedback_rows(rows: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Scores para feedback real: usa ``score`` o infiere con G sobre la paleta."""
    import app.services.inference as inference
    from app.ml.color_spaces import ColorConverter

    scores: list[float] = []
    labels: list[bool] = []
    pending: list[int] = []
    parsed: list[list[str]] = []
    for row in rows:
        labels.append(bool(row["accepted"]))
        if row["score"] is not None:
            scores.append(float(row["score"]))
            parsed.append([])
        else:
            scores.append(float("nan"))
            pending.append(len(scores) - 1)
            try:
                palette = json.loads(row["palette_json"] or "{}")
                parsed.append(
                    [str(c.get("value", "")).upper() for c in palette.get("colors", [])]
                )
            except Exception:
                parsed.append([])

    if pending:
        model = inference.load_keras_model("generate")
        if model is None:
            raise RuntimeError("no hay red G activa para puntuar el feedback")
        x_lab = np.zeros((len(pending), 10, 3), dtype=np.float32)
        x_mask = np.zeros((len(pending), 10), dtype=np.float32)
        x_harm = np.zeros((len(pending), 4), dtype=np.float32)
        for k, idx in enumerate(pending):
            for j, hexv in enumerate(parsed[idx][:10]):
                try:
                    lab = ColorConverter.lab_normalize(ColorConverter.hex_to_lab(hexv))
                except Exception:
                    continue
                x_lab[k, j] = np.asarray(lab, dtype=np.float32)
                x_mask[k, j] = 1.0
        _pal, pred_score = model.predict([x_lab, x_mask, x_harm], verbose=0)
        flat = np.asarray(pred_score, dtype=np.float64).reshape(-1)
        for k, idx in enumerate(pending):
            scores[idx] = float(flat[k])
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)


def calibrate_generate(
    manifest: dict[str, Any],
    *,
    dry_run: bool,
    source: str,
    min_auc: float,
    manifest_path: Path,
) -> dict[str, Any]:
    rows = _feedback_rows() if source in ("auto", "feedback") else []
    used_source = "feedback"
    stats: dict[str, Any] = {}
    if len(rows) >= MIN_FEEDBACK_ROWS:
        scores, good = _g_scores_for_feedback_rows(rows)
        stats = {"n": int(len(scores)), "rows": int(len(rows))}
    else:
        if source == "feedback":
            raise RuntimeError(
                f"feedback insuficiente: {len(rows)} filas < {MIN_FEEDBACK_ROWS}"
            )
        used_source = "reconstruction"
        if source == "auto" and rows:
            print(
                f"[calibrate] G: solo {len(rows)} filas de feedback "
                f"(<{MIN_FEEDBACK_ROWS}); se usa el proxy de reconstrucción."
            )
        scores, good, stats = _reconstruction_samples()

    auc = auc_score(scores, good)
    usable = bool(np.isfinite(auc) and auc >= min_auc)
    print(f"[calibrate] G: fuente={used_source} n={stats.get('n')} AUC={auc:.4f} "
          f"(mínimo {min_auc:.2f})")
    if "score_mean" in stats:
        print(
            "  score: media {mean:.4f}  std {std:.4f}  buena_reconstrucción "
            "{rate:.2%}".format(
                mean=stats["score_mean"],
                std=stats["score_std"],
                rate=stats["good_rate"],
            )
        )

    coverage_rows = risk_coverage(
        np.column_stack([1.0 - scores, scores]), good.astype(int), [0.80, 0.85, 0.90, 0.92, 0.94, 0.96]
    )
    # risk_coverage asume confianza = max(prob); el score de G es la confianza.
    _print_risk_table("risk-coverage del score de G (umbral = score mínimo):", coverage_rows)

    if not usable:
        reason = "score_auc_below_threshold"
        calibration: dict[str, Any] = {
            "method": "platt",
            "calibrated": False,
            "reason": reason,
            "auc": _round(auc),
            "minAuc": float(min_auc),
            "a": None,
            "b": None,
        }
        print(
            "  RECHAZADO: el AUC no supera el mínimo; no se aplica Platt ni se "
            "invierte el score (cambiaría el ranking sin base)."
        )
        print(
            "  umbral recomendado: no filtrar por score (el score no discrimina "
            "calidad en v1)."
        )
    else:
        a, b = platt_fit(scores, good)
        before_nll = float(
            -np.mean(
                good * np.log(np.clip(scores, 1e-12, 1.0))
                + (~good) * np.log(np.clip(1.0 - scores, 1e-12, 1.0))
            )
        )
        calibrated = apply_platt(scores, a, b)
        after_nll = float(
            -np.mean(
                good * np.log(np.clip(calibrated, 1e-12, 1.0))
                + (~good) * np.log(np.clip(1.0 - calibrated, 1e-12, 1.0))
            )
        )
        calibration = {
            "method": "platt",
            "calibrated": True,
            "reason": None,
            "auc": _round(auc),
            "minAuc": float(min_auc),
            "a": _round(a),
            "b": _round(b),
            "n": int(len(scores)),
        }
        print(f"  Platt: a={a:.6f} b={b:.6f}")
        print(f"  log-loss antes {before_nll:.4f} -> después {after_nll:.4f}")

    active = manifest.get("generate", {}).get("active")
    metadata = (
        manifest.get("generate", {}).get("files", {}).get(active) if active else None
    )
    if not isinstance(metadata, dict):
        raise RuntimeError("el manifiesto no tiene metadatos para la G activa")

    if dry_run:
        print("[calibrate] dry-run: no se escribe el manifiesto")
    else:
        metadata["calibration"] = calibration
        _write_manifest(manifest_path, manifest)
        print(f"[calibrate] manifiesto actualizado: generate/{active}")

    return {
        "slot": "generate",
        "active": active,
        "source": used_source,
        "auc": auc,
        "minimumAuc": min_auc,
        "accepted": usable,
        "stats": stats,
        "riskCoverage": coverage_rows,
        "calibration": calibration,
        "applied": (not dry_run) and usable,
    }


# --- CLI ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slot", choices=("classify", "generate", "all"), default="all"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mide e imprime pero no escribe el manifiesto",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "feedback", "reconstruction"),
        default="auto",
        help="Fuente para G (auto: feedback si hay >=50 filas, si no proxy)",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--json", default=None, help="Opcional: escribe el resumen JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    manifest = _load_json(manifest_path)
    thresholds = _load_json(Path(args.thresholds))
    calibration_cfg = thresholds.get("calibration", {})
    t_min = float(calibration_cfg.get("t_min", 0.2))
    t_max = float(calibration_cfg.get("t_max", 5.0))
    min_auc = float(calibration_cfg.get("require_auc_min", DEFAULT_MIN_AUC))

    result: dict[str, Any] = {"dryRun": bool(args.dry_run), "slots": {}}
    try:
        if args.slot in ("classify", "all"):
            result["slots"]["classify"] = calibrate_classify(
                manifest,
                dry_run=args.dry_run,
                t_min=t_min,
                t_max=t_max,
                manifest_path=manifest_path,
            )
        if args.slot in ("generate", "all"):
            result["slots"]["generate"] = calibrate_generate(
                manifest,
                dry_run=args.dry_run,
                source=args.source,
                min_auc=min_auc,
                manifest_path=manifest_path,
            )
    except RuntimeError as exc:
        print(f"[calibrate] error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        Path(args.json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[calibrate] JSON escrito en {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

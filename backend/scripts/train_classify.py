"""Entrena la red C clasificadora (Fase 4 / issue #22).

Modos:
- `pipeline` (por defecto, `classify-v2`): paletas etiquetadas con
  `app.ml.benchmarks.build_classify_palette_list`, convertidas a mosaico
  sintético con pesos Dirichlet,
  orden barajado, tamaño/JPEG variables; se extraen los colores con la red E
  activa (`inference.extract_image_colors`, camino real de producción) y se
  construyen las features de 54 dims (`palette_features`). Este modo es
  pipeline-consistente: entrena con la misma representación que usa la
  inferencia.
- `synthetic` (legacy, `classify-v1`): 40 dims (10*Lab +10 pesos) generadas
  directamente desde los HEX, sin pasar por E.

Arquitectura: MLP input_dim→64→32 ReLU → softmax 4
Loss: categorical_crossentropy + label smoothing 0.05
Datos: paletas de la DB de entrenamiento (PostgreSQL) con harmony_label +
fallback sintético.

Uso (solo .venv-tf):
    backend/.venv-tf/bin/python backend/scripts/train_classify.py
    backend/.venv-tf/bin/python backend/scripts/train_classify.py --mode synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.db.session import training_session
from app.ml.benchmarks import (
    DEFAULT_CACHE_DIR,
    FEATURE_DIM_V1,
    FEATURE_DIM_V2,
    HARMONY_LABELS,
    build_classify_palette_list as build_palette_list,
    classify_split_palettes,
    extract_pipeline_features,
    palette_to_legacy_dataset as build_dataset,
)
from app.ml.metrics import (
    confusion_matrix,
    macro_f1,
    nll,
    per_class_report,
)
from app.repositories import training as training_repo
from app.services import model_registry


def classification_metrics(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Métricas completas sin sklearn: acc, macro-F1, recall/clase, confusión."""
    probs = np.asarray(model.predict(X, verbose=0), dtype=np.float64)
    pred = np.argmax(probs, axis=1)
    true = np.argmax(y, axis=1)
    n = len(true)
    cm = confusion_matrix(true, pred, n_classes=len(HARMONY_LABELS))
    report = per_class_report(
        true, pred, labels=HARMONY_LABELS, n_classes=len(HARMONY_LABELS)
    )
    return {
        "accuracy": float(np.mean(pred == true)) if n else 0.0,
        "macro_f1": macro_f1(
            true, pred, labels=HARMONY_LABELS, n_classes=len(HARMONY_LABELS)
        )
        if n
        else 0.0,
        "recall_per_class": {
            label: float(report[label]["recall"]) for label in HARMONY_LABELS
        },
        "confusion": cm.tolist(),
        "nll": nll(probs, true) if n else 0.0,
        "n": int(n),
    }


def train_pipeline(
    epochs: int = 60,
    n_train: int = 600,
    n_val: int = 120,
    n_test: int = 80,
    batch: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
):
    import tensorflow as tf

    from app.ml.classify_model import build_classify_model

    splits = classify_split_palettes(
        n_train=n_train, n_val=n_val, n_test=n_test, seed=seed
    )
    pals_tr, pals_va, pals_te = splits["train"], splits["val"], splits["test"]
    X_tr, _, y_tr, stats_tr = extract_pipeline_features(
        pals_tr, seed, split="train", cache_dir=cache_dir
    )
    X_va, _, y_va, stats_va = extract_pipeline_features(
        pals_va, seed + 10_000, split="val", cache_dir=cache_dir
    )
    X_te, _, y_te, stats_te = extract_pipeline_features(
        pals_te, seed + 20_000, split="test", cache_dir=cache_dir
    )

    print(
        f"[train_classify] TF {tf.__version__} modo=pipeline dim={FEATURE_DIM_V2} "
        f"n_train={n_train} n_val={n_val} n_test={n_test} "
        f"fallbacks(E)={stats_tr['fallback_extractions']}/"
        f"{stats_va['fallback_extractions']}/{stats_te['fallback_extractions']}",
        flush=True,
    )

    model = build_classify_model(input_dim=FEATURE_DIM_V2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch,
        verbose=2,
        callbacks=[early],
    )
    epochs_run = len(history.history.get("loss", []))

    train_metrics = classification_metrics(model, X_tr, y_tr)
    val_metrics = classification_metrics(model, X_va, y_va)
    test_metrics = classification_metrics(model, X_te, y_te)
    print(
        f"[train_classify] train acc={train_metrics['accuracy']:.4f} "
        f"val acc={val_metrics['accuracy']:.4f} "
        f"test acc={test_metrics['accuracy']:.4f} "
        f"epochs={epochs_run}",
        flush=True,
    )
    metrics = {
        "val_accuracy": val_metrics["accuracy"],
        "val_loss": float(
            np.min(history.history["val_loss"]) if history.history.get("val_loss") else 0.0
        ),
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_recall_per_class": test_metrics["recall_per_class"],
        "test_confusion": test_metrics["confusion"],
        "test_nll": test_metrics["nll"],
        "epochs_run": epochs_run,
        "feature_dim": FEATURE_DIM_V2,
        "pipeline_consistent": True,
        "fallback_extractions": {
            "train": stats_tr["fallback_extractions"],
            "val": stats_va["fallback_extractions"],
            "test": stats_te["fallback_extractions"],
        },
    }
    return model, metrics


def train_and_evaluate(epochs=20, n_train=800, n_val=120, batch=32, lr=1e-3, seed=42):
    import tensorflow as tf
    from app.ml.classify_model import build_classify_model

    pals_tr = build_palette_list(n_train, seed)
    pals_va = build_palette_list(n_val, seed + 10_000)
    X_tr, y_tr = build_dataset(pals_tr, seed)
    X_va, y_va = build_dataset(pals_va, seed + 10_000)

    print(f"[train_classify] TF {tf.__version__} n_train={n_train} n_val={n_val}", flush=True)

    model = build_classify_model(input_dim=FEATURE_DIM_V1)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs, batch_size=batch, verbose=2)

    loss, acc = model.evaluate(X_va, y_va, verbose=0)
    print(f"[train_classify] val acc={acc:.4f} loss={loss:.4f}", flush=True)

    # confusion matrix (optional)
    preds = model.predict(X_va, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(y_va, axis=1)
    # compute per class accuracy
    from collections import Counter

    correct = int(np.sum(pred_labels == true_labels))
    total = len(true_labels)
    print(f"[train_classify] correct {correct}/{total}", flush=True)

    return model, {"val_accuracy": float(acc), "val_loss": float(loss), "history": history.history}


def save_and_register(
    model, out_path: Path, params: dict, metrics: dict, *, force: bool = False
) -> dict:
    tmp_path = out_path.with_name(f".tmp-{out_path.name}")
    model.save(str(tmp_path))
    import tensorflow as tf

    try:
        reloaded = tf.keras.models.load_model(str(tmp_path))
        assert reloaded.output_shape[-1] == 4

        tf_version, keras_version = model_registry.framework_versions()
        with training_session() as session:
            run = training_repo.create_training_run(
                session,
                model="classify",
                keras_path=f"backend/models/{out_path.name}",
                git_sha=model_registry.git_sha(),
                params_json=params,
                metrics_json=metrics,
                tf_version=tf_version,
                keras_version=keras_version,
            )
            run_id = int(run.id or 0)

        try:
            registered = model_registry.register_version(
                "classify",
                tmp_path,
                metrics=metrics,
                params=params,
                training_run_id=run_id,
                force=force,
                filename=out_path.name,
            )
        except Exception:
            with training_session() as session:
                training_repo.delete_training_run(session, run_id)
            raise
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return {
        "keras_path": registered["keras_path"],
        "filename": registered["filename"],
        "training_run_id": run_id,
    }


def main(argv: list[str] | None = None) -> dict:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pipeline", "synthetic"), default="pipeline")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default=None,
                        help="Versión nueva: entero o vN (ej. 3 o v3 -> classify-v3.keras)")
    parser.add_argument("--out", type=str, default=None,
                        help="Ruta alternativa (prevalece sobre --version); se registra su nombre real")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Sobrescribe la versión destino y registra aunque acc<threshold")
    args = parser.parse_args(argv)

    out_path = model_registry.target_path(
        "classify", version=args.version, out=args.out
    )
    if out_path.exists() and not args.force:
        raise SystemExit(
            f"[train_classify] ERROR: {out_path} ya existe; usa --version N libre "
            "o --force para sobrescribir (nunca se pisa una versión por accidente)"
        )

    if args.mode == "pipeline":
        epochs = args.epochs if args.epochs is not None else 60
        n_train = args.n_train if args.n_train is not None else 600
        n_val = args.n_val if args.n_val is not None else 120
        n_test = args.n_test if args.n_test is not None else 80
        cache_dir = None if args.no_cache else Path(args.cache_dir)
        model, eval_metrics = train_pipeline(
            epochs=epochs,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            batch=args.batch,
            lr=args.lr,
            seed=args.seed,
            cache_dir=cache_dir,
        )
        params = {
            "mode": "pipeline",
            "feature_dim": FEATURE_DIM_V2,
            "epochs": epochs,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "batch": args.batch,
            "lr": args.lr,
            "seed": args.seed,
            "label_smoothing": 0.05,
            "early_stopping_patience": 8,
            "mosaic": {
                "size": [96, 160],
                "jpeg_quality": [60, 95],
                "min_weight": 0.08,
                "shuffle_order": True,
            },
            "extractor": model_registry.active_version("extract") or "extract-v1.keras",
        }
        metrics = dict(eval_metrics)
        threshold = 0.70
    else:
        epochs = args.epochs if args.epochs is not None else 20
        n_train = args.n_train if args.n_train is not None else 800
        n_val = args.n_val if args.n_val is not None else 120
        model, eval_metrics = train_and_evaluate(
            epochs=epochs,
            n_train=n_train,
            n_val=n_val,
            batch=args.batch,
            lr=args.lr,
            seed=args.seed,
        )
        params = {
            "mode": "synthetic",
            "feature_dim": FEATURE_DIM_V1,
            "epochs": epochs,
            "n_train": n_train,
            "n_val": n_val,
            "batch": args.batch,
            "lr": args.lr,
            "seed": args.seed,
        }
        metrics = {
            "val_accuracy": eval_metrics["val_accuracy"],
            "val_loss": eval_metrics["val_loss"],
        }
        threshold = 0.70

    ok = metrics["val_accuracy"] >= threshold
    saved = None
    if ok or args.force:
        saved = save_and_register(
            model, out_path, params, metrics, force=args.force
        )
        print(
            f"[train_classify] guardado {saved['keras_path']} "
            f"run={saved['training_run_id']} acc={metrics['val_accuracy']:.4f}",
            flush=True,
        )
    else:
        print(
            f"[train_classify] acc {metrics['val_accuracy']:.4f} <{threshold} "
            "sin --force no registra",
            flush=True,
        )
    result = {"params": params, "metrics": metrics, "ok": ok, "saved": saved}
    print("[train_classify] " + json.dumps(result, sort_keys=True))
    if not ok and not args.force:
        raise SystemExit(2)
    return result


if __name__ == "__main__":
    main()

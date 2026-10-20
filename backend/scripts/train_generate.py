"""Entrena la red G generadora (Fase 4, 100% offline si fuentes públicas fallan).

Pipeline:
1. Paletas: lee palette_colors de training.db; si no alcanzan, fallback sintético
   HSL vía ClassicPaletteGenerator + ruido Lab (idéntico a make_synthetic).
2. Por paleta ground truth (5 colores) genera una muestra de entrenamiento:
   entrada = (10,3) Lab normalizado + mask (10,) + harmony one-hot (4,) con
   1-3 colores base visibles; salida = paleta completa (5*3=15) + rating sigmoide.
3. Entrena app.ml.generate_model.build_generate_model con loss
   MSE-Lab + mu*CIEDE2000 aproximado (la loss NO cambia; sigue siendo la
   aproximación euclidiana diferenciable).
4. Guarda backend/models/<slot>-v<N>.keras (siguiente versión libre o
   `--version`/`--out`), activa el slot `generate` en `manifest.json` vía
   `model_registry.register_version` e inserta fila en `training_runs` con
   versiones TF/Keras. Eval: CIEDE2000 **real** (Sharma) y CIE76
   vs ground truth, en ``app.ml.metrics``.

Uso (solo .venv-tf):
    backend/.venv-tf/bin/python backend/scripts/train_generate.py --epochs 20 --n-train 800
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.db.sqlite import get_connection
from app.ml.benchmarks import (
    build_generate_palette_list as build_palette_list,
    palette_to_generate_dataset as palette_to_dataset,
)
from app.ml.metrics import palette_delta_e
from app.services import model_registry


def _denormalize_palette_batch(values) -> np.ndarray:
    """(N,15) Lab normalizado → (N,5,3) Lab crudo."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1, 5, 3)
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0] * 100.0
    out[..., 1] = arr[..., 1] * 128.0
    out[..., 2] = arr[..., 2] * 128.0
    return out


def mean_palette_delta_e(y_true_pal, y_pred_pal, *, metric: str = "ciede2000") -> float:
    """ΔE medio por paleta con emparejamiento greedy (CIEDE2000 real).

    ``metric="cie76"`` calcula la euclidiana en Lab (métrica histórica que
    antes se etiquetaba erróneamente como CIEDE2000).
    """
    pred = _denormalize_palette_batch(y_pred_pal)
    true = _denormalize_palette_batch(y_true_pal)
    values = [
        palette_delta_e(p, t, metric=metric) for p, t in zip(pred, true)
    ]
    return float(np.mean(values)) if values else 0.0


def train_and_evaluate(epochs=20, n_train=800, n_val=120, batch=32, lr=1e-3, seed=42, mu=0.1):
    import tensorflow as tf
    from app.ml.generate_model import build_generate_model, palette_loss_with_ciede

    pals_tr = build_palette_list(n_train, seed)
    pals_va = build_palette_list(n_val, seed + 10_000)

    (X_lab_tr, X_mask_tr, X_harm_tr), (Y_pal_tr, Y_score_tr) = palette_to_dataset(pals_tr, seed)
    (X_lab_va, X_mask_va, X_harm_va), (Y_pal_va, Y_score_va) = palette_to_dataset(pals_va, seed + 10_000)

    print(f"[train_generate] TF {tf.__version__} n_train={n_train} n_val={n_val} mu={mu}", flush=True)

    model = build_generate_model()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss={"palette": palette_loss_with_ciede(mu), "score": "binary_crossentropy"},
        loss_weights={"palette": 1.0, "score": 0.5},
        metrics={"score": ["accuracy"]},
    )

    # fit
    history = model.fit(
        {"palette_lab": X_lab_tr, "mask": X_mask_tr, "harmony": X_harm_tr},
        {"palette": Y_pal_tr, "score": Y_score_tr},
        validation_data=(
            {"palette_lab": X_lab_va, "mask": X_mask_va, "harmony": X_harm_va},
            {"palette": Y_pal_va, "score": Y_score_va},
        ),
        epochs=epochs,
        batch_size=batch,
        verbose=2,
    )

    # eval CIEDE2000 real + CIE76 (histórico) + MSE en espacio normalizado
    pred_pal, _pred_score = model.predict(
        {"palette_lab": X_lab_va, "mask": X_mask_va, "harmony": X_harm_va}, verbose=0
    )
    ciede2000 = mean_palette_delta_e(Y_pal_va, pred_pal, metric="ciede2000")
    cie76 = mean_palette_delta_e(Y_pal_va, pred_pal, metric="cie76")
    print(
        f"[train_generate] val CIEDE2000={ciede2000:.3f} CIE76={cie76:.3f}",
        flush=True,
    )
    mse = float(np.mean((Y_pal_va - pred_pal) ** 2))
    print(f"[train_generate] val MSE={mse:.5f}", flush=True)

    return model, {
        "val_ciede2000": ciede2000,
        "val_cie76": cie76,
        "val_mse": mse,
        "history": history.history,
    }


def save_and_register(
    model, out_path: Path, params: dict, metrics: dict, *, force: bool = False
) -> dict:
    tmp_path = out_path.with_name(f".tmp-{out_path.name}")
    model.save(str(tmp_path))
    # verify reload
    import tensorflow as tf
    from app.ml.generate_model import GENERATE_CUSTOM_OBJECTS

    try:
        reloaded = tf.keras.models.load_model(
            str(tmp_path), custom_objects=GENERATE_CUSTOM_OBJECTS
        )
        assert len(reloaded.outputs) == 2

        tf_version, keras_version = model_registry.framework_versions()
        conn = get_connection()
        try:
            cur = conn.execute(
                "INSERT INTO training_runs(model, git_sha, params_json, metrics_json,"
                " keras_path, tf_version, keras_version, calibration_json)"
                " VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "generate",
                    model_registry.git_sha(),
                    json.dumps(params, sort_keys=True),
                    json.dumps(metrics, sort_keys=True),
                    f"backend/models/{out_path.name}",
                    tf_version,
                    keras_version,
                    None,
                ),
            )
            conn.commit()
            run_id = int(cur.lastrowid or 0)
        finally:
            conn.close()

        try:
            registered = model_registry.register_version(
                "generate",
                tmp_path,
                metrics=metrics,
                params=params,
                training_run_id=run_id,
                force=force,
                filename=out_path.name,
            )
        except Exception:
            conn = get_connection()
            try:
                conn.execute("DELETE FROM training_runs WHERE id = ?", (run_id,))
                conn.commit()
            finally:
                conn.close()
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-val", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mu", type=float, default=0.1, help="peso CIEDE")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default=None,
                        help="Versión nueva: entero o vN (ej. 2 o v2 -> generate-v2.keras)")
    parser.add_argument("--out", type=str, default=None,
                        help="Ruta alternativa (prevalece sobre --version); se registra su nombre real")
    parser.add_argument("--force", action="store_true",
                        help="Sobrescribe la versión destino y registra aunque CIEDE>=30")
    args = parser.parse_args(argv)

    out_path = model_registry.target_path(
        "generate", version=args.version, out=args.out
    )
    if out_path.exists() and not args.force:
        raise SystemExit(
            f"[train_generate] ERROR: {out_path} ya existe; usa --version N libre "
            "o --force para sobrescribir (nunca se pisa una versión por accidente)"
        )

    model, eval_metrics = train_and_evaluate(
        epochs=args.epochs, n_train=args.n_train, n_val=args.n_val, batch=args.batch, lr=args.lr, seed=args.seed, mu=args.mu
    )
    params = {
        "epochs": args.epochs,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "batch": args.batch,
        "lr": args.lr,
        "mu": args.mu,
        "seed": args.seed,
    }
    metrics = {
        "val_ciede2000": eval_metrics["val_ciede2000"],
        "val_cie76": eval_metrics["val_cie76"],
        "val_mse": eval_metrics["val_mse"],
    }
    ok = metrics["val_ciede2000"] < 30.0
    saved = None
    if ok or args.force:
        saved = save_and_register(
            model, out_path, params, metrics, force=args.force
        )
        print(
            f"[train_generate] guardado {saved['keras_path']} run={saved['training_run_id']} "
            f"CIEDE2000={metrics['val_ciede2000']:.3f}",
            flush=True,
        )
    else:
        print(
            f"[train_generate] CIEDE2000 {metrics['val_ciede2000']:.3f} >=30 "
            "sin --force no registra",
            flush=True,
        )
    result = {"params": params, "metrics": metrics, "ok": ok, "saved": saved}
    print("[train_generate] " + json.dumps(result, sort_keys=True))
    if not ok and not args.force:
        raise SystemExit(2)
    return result


if __name__ == "__main__":
    main()

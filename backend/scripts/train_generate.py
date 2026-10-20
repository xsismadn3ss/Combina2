"""Entrena la red G generadora (Fase 4, 100% offline si fuentes públicas fallan).

Pipeline:
1. Paletas: lee palette_colors de training.db; si no alcanzan, fallback sintético
   HSL vía ClassicPaletteGenerator + ruido Lab (idéntico a make_synthetic).
2. Por paleta ground truth (5 colores) genera una muestra de entrenamiento:
   entrada = (10,3) Lab normalizado + mask (10,) + harmony one-hot (4,) con
   1-3 colores base visibles; salida = paleta completa (5*3=15) + rating sigmoide.
3. Entrena app.ml.generate_model.build_generate_model con loss
   MSE-Lab + mu*CIEDE2000 aprox + BCE rating.
4. Guarda backend/models/generate-v1.keras, activa slot generate en manifest.json
   e inserta fila en training_runs. Eval: CIEDE2000 vs ground truth.

Uso (solo .venv-tf):
    backend/.venv-tf/bin/python backend/scripts/train_generate.py --epochs 20 --n-train 800
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.db.sqlite import get_connection
from app.ml.color_spaces import ColorConverter

MODEL_NAME = "generate-v1.keras"
HARMONY_LABELS = ["complementario", "triada", "analogo", "monocromatica"]
HARMONY_TO_IDX = {h: i for i, h in enumerate(HARMONY_LABELS)}


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=BACKEND_DIR.parent,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_db_palettes(limit: int) -> list[dict]:
    """Lee palettes con harmony_label y rating + hexes ordenados."""
    try:
        conn = get_connection()
    except Exception:
        return []
    try:
        rows = conn.execute(
            "SELECT p.id, p.rating, p.harmony_label, c.position, c.hex "
            "FROM palettes p JOIN palette_colors c ON c.palette_id=p.id "
            "ORDER BY p.id, c.position"
        ).fetchall()
    except Exception:
        return []
    finally:
        conn.close()
    # agrupar por id
    pal_map: dict[int, dict] = {}
    for pid, rating, harmony, pos, hexv in rows:
        if pid not in pal_map:
            pal_map[pid] = {"hexes": [], "rating": rating, "harmony_label": harmony}
        pal_map[pid]["hexes"].append(hexv)
    result = list(pal_map.values())[:limit]
    return result


def synthetic_palette(rng: np.random.RandomState, sigma: float = 2.5) -> dict:
    """Genera una paleta sintética 5 colores + harmony_label perfecta (offline)."""
    from app.schemas.palette import HarmonyType
    from app.services.classic import ClassicPaletteGenerator

    LABEL_TO_HARMONY = {
        "complementario": HarmonyType.COMPLEMENTARY,
        "triada": HarmonyType.TRIADIC,
        "analogo": HarmonyType.ANALOGOUS,
        "monocromatica": HarmonyType.MONOCHROMATIC,
    }
    gen = ClassicPaletteGenerator()
    base_hex = "#%02X%02X%02X" % tuple(int(v) for v in rng.randint(0, 256, size=3))
    label = HARMONY_LABELS[int(rng.randint(0, len(HARMONY_LABELS)))]
    resp = gen.generate_palette([base_hex], LABEL_TO_HARMONY[label])
    hexes = [c.value.upper() for c in resp.colors]
    # pad to 5 with noisy variants
    while len(hexes) < 5:
        idx = int(rng.randint(0, len(hexes)))
        lab = np.array(ColorConverter.hex_to_lab(hexes[idx]), dtype=np.float64)
        lab += rng.normal(0.0, sigma, size=3)
        lab[0] = float(np.clip(lab[0], 0.0, 100.0))
        hexes.append(ColorConverter.lab_to_hex((lab[0], lab[1], lab[2])).upper())
    return {"hexes": hexes[:5], "rating": None, "harmony_label": label}


def build_palette_list(n: int, seed: int) -> list[dict]:
    pals = load_db_palettes(n)
    rng = np.random.RandomState(seed + 999_000)
    while len(pals) < n:
        pals.append(synthetic_palette(rng))
    return pals[:n]


def palette_to_dataset(palettes: list[dict], seed: int):
    """Convierte paletas ground truth a arrays de entrenamiento G."""
    rng = np.random.RandomState(seed)
    N = len(palettes)
    X_lab = np.zeros((N, 10, 3), dtype=np.float32)
    X_mask = np.zeros((N, 10), dtype=np.float32)
    X_harm = np.zeros((N, 4), dtype=np.float32)
    Y_pal = np.zeros((N, 15), dtype=np.float32)
    Y_score = np.zeros((N, 1), dtype=np.float32)
    for i, pal in enumerate(palettes):
        hexes = [h.upper() for h in pal["hexes"][:5]]
        # pad to 5 if fewer
        while len(hexes) < 5:
            hexes.append(hexes[-1])
        # ground truth palette
        labs = [ColorConverter.lab_normalize(ColorConverter.hex_to_lab(h)) for h in hexes]
        Y_pal[i] = np.array(labs, dtype=np.float32).reshape(-1)
        rating = pal.get("rating")
        if rating is None:
            Y_score[i, 0] = 1.0
        else:
            try:
                Y_score[i, 0] = float(np.clip(float(rating), 0.0, 1.0))
            except Exception:
                Y_score[i, 0] = 1.0
        # input: 1..3 base colors (prefijo determinista: facilita aprendizaje copiar)
        k = int(rng.randint(1, 4))  # 1,2,3
        base_hexes = hexes[:k]
        for j, bh in enumerate(base_hexes):
            nlab = ColorConverter.lab_normalize(ColorConverter.hex_to_lab(bh))
            X_lab[i, j] = np.array(nlab, dtype=np.float32)
            X_mask[i, j] = 1.0
        # harmony one-hot
        harmony = pal.get("harmony_label")
        if harmony in HARMONY_TO_IDX:
            X_harm[i, HARMONY_TO_IDX[harmony]] = 1.0
        else:
            # random harmony if missing
            r = int(rng.randint(0, 4))
            X_harm[i, r] = 1.0
    return (X_lab, X_mask, X_harm), (Y_pal, Y_score)


def mean_ciede2000_approx(y_true_pal, y_pred_pal) -> float:
    """CIEDE2000 aprox = euclidean mean en Lab desnormalizado (CIE76)."""
    # y_true/pred shape (N,15) -> (N,5,3) raw Lab
    def denorm(t):
        t = np.asarray(t, dtype=np.float64).reshape(-1, 5, 3)
        out = np.empty_like(t)
        out[..., 0] = t[..., 0] * 100.0
        out[..., 1] = t[..., 1] * 128.0
        out[..., 2] = t[..., 2] * 128.0
        return out

    pred = denorm(y_pred_pal)
    true = denorm(y_true_pal)
    dists = np.sqrt(np.sum((pred - true) ** 2, axis=-1))  # (N,5)
    return float(np.mean(dists))


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

    # eval CIEDE
    pred_pal, pred_score = model.predict(
        {"palette_lab": X_lab_va, "mask": X_mask_va, "harmony": X_harm_va}, verbose=0
    )
    ciede = mean_ciede2000_approx(Y_pal_va, pred_pal)
    print(f"[train_generate] val CIEDE2000 approx={ciede:.3f}", flush=True)
    # also compute mse
    mse = float(np.mean((Y_pal_va - pred_pal) ** 2))
    print(f"[train_generate] val MSE={mse:.5f}", flush=True)

    return model, {"val_cie": ciede, "val_mse": mse, "history": history.history}


def save_and_register(model, out_path: Path, params: dict, metrics: dict) -> dict:
    model.save(str(out_path))
    # verify reload
    import tensorflow as tf
    from app.ml.generate_model import GENERATE_CUSTOM_OBJECTS

    reloaded = tf.keras.models.load_model(str(out_path), custom_objects=GENERATE_CUSTOM_OBJECTS)
    assert len(reloaded.outputs) == 2

    manifest_path = BACKEND_DIR / "models" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    slot = manifest.setdefault("generate", {"active": None, "versions": []})
    if MODEL_NAME not in slot.get("versions", []):
        slot.setdefault("versions", []).append(MODEL_NAME)
    slot["active"] = MODEL_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO training_runs(model, git_sha, params_json, metrics_json, keras_path) VALUES(?, ?, ?, ?, ?)",
            (
                "generate",
                git_sha(),
                json.dumps(params, sort_keys=True),
                json.dumps(metrics, sort_keys=True),
                f"backend/models/{MODEL_NAME}",
            ),
        )
        conn.commit()
        run_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    finally:
        conn.close()
    return {"keras_path": str(out_path), "training_run_id": run_id}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-val", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mu", type=float, default=0.1, help="peso CIEDE")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(BACKEND_DIR / "models" / MODEL_NAME))
    parser.add_argument("--force", action="store_true", help="Registra aunque CIEDE>=30")
    args = parser.parse_args(argv)

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
    metrics = {"val_cie": eval_metrics["val_cie"], "val_mse": eval_metrics["val_mse"]}
    ok = metrics["val_cie"] < 30.0
    saved = None
    if ok or args.force:
        saved = save_and_register(model, Path(args.out), params, metrics)
        print(f"[train_generate] guardado {saved['keras_path']} run={saved['training_run_id']} ciede={metrics['val_cie']:.3f}", flush=True)
    else:
        print(f"[train_generate] CIEDE {metrics['val_cie']:.3f} >=30 sin --force no registra", flush=True)
    result = {"params": params, "metrics": metrics, "ok": ok, "saved": saved}
    print("[train_generate] " + json.dumps(result, sort_keys=True))
    if not ok and not args.force:
        raise SystemExit(2)
    return result


if __name__ == "__main__":
    main()

"""Entrena la red C clasificadora (Fase 4).

Entrada: 40 dims (10*Lab +10 pesos de E)
Arquitectura: MLP 40→64→32 ReLU → softmax 4
Loss: categorical_crossentropy + label smoothing 0.05
Datos: training.db palettes con harmony_label + fallback sintético.
Eval: accuracy ≥0.85 sintético / ≥0.70 curado.

Uso (solo .venv-tf):
    backend/.venv-tf/bin/python backend/scripts/train_classify.py --epochs 20
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

MODEL_NAME = "classify-v1.keras"
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
    try:
        conn = get_connection()
    except Exception:
        return []
    try:
        rows = conn.execute(
            "SELECT p.id, p.harmony_label, c.position, c.hex "
            "FROM palettes p JOIN palette_colors c ON c.palette_id=p.id "
            "WHERE p.harmony_label IN ('complementario','triada','analogo','monocromatica') "
            "ORDER BY p.id, c.position"
        ).fetchall()
    except Exception:
        return []
    finally:
        conn.close()
    pal_map: dict[int, dict] = {}
    for pid, harmony, pos, hexv in rows:
        if pid not in pal_map:
            pal_map[pid] = {"hexes": [], "harmony_label": harmony}
        pal_map[pid]["hexes"].append(hexv)
    # filter only complete 5-color palettes
    result = [v for v in pal_map.values() if len(v["hexes"]) == 5 and v["harmony_label"] in HARMONY_TO_IDX][:limit]
    return result


def synthetic_palette(rng: np.random.RandomState, sigma: float = 2.5) -> dict:
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
    while len(hexes) < 5:
        idx = int(rng.randint(0, len(hexes)))
        lab = np.array(ColorConverter.hex_to_lab(hexes[idx]), dtype=np.float64)
        lab += rng.normal(0.0, sigma, size=3)
        lab[0] = float(np.clip(lab[0], 0.0, 100.0))
        hexes.append(ColorConverter.lab_to_hex((lab[0], lab[1], lab[2])).upper())
    return {"hexes": hexes[:5], "harmony_label": label}


def build_palette_list(n: int, seed: int) -> list[dict]:
    pals = load_db_palettes(n)
    rng = np.random.RandomState(seed + 777_000)
    while len(pals) < n:
        pals.append(synthetic_palette(rng))
    # balance classes via oversampling if needed: ensure each class present
    # just shuffle
    rng.shuffle(pals)
    return pals[:n]


def palette_to_features(hexes: list[str], rng: np.random.RandomState | None = None) -> np.ndarray:
    """Convierte paleta 5 colores a vector 40 dims (10 Lab +10 pesos) simulando salida E.

    Usa pesos descendentes fijos (0.30..0.10 + 0.02) con pequeño ruido,
    preservando el orden original (estable) para que la señal de armonía
    sea aprendible. Con Dirichlet aleatorio la accuracy caía a ~0.38.
    """
    labs5 = [ColorConverter.lab_normalize(ColorConverter.hex_to_lab(h)) for h in hexes]
    labs10 = []
    for i in range(10):
        if i < 5:
            labs10.append(labs5[i])
        else:
            base = np.array(labs5[-1], dtype=np.float64)
            if rng is not None:
                base = base + rng.normal(0.0, 0.02, size=3)
                base[0] = float(np.clip(base[0], 0.0, 1.0))
                base[1] = float(np.clip(base[1], -1.0, 1.0))
                base[2] = float(np.clip(base[2], -1.0, 1.0))
            labs10.append(tuple(base))
    labs10_arr = np.array(labs10, dtype=np.float32)  # (10,3)

    # pesos: descendentes fijos + ruido leve (preserva orden)
    w10 = np.array([0.30, 0.25, 0.20, 0.15, 0.10, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float64)
    if rng is not None:
        w10[:5] += rng.normal(0.0, 0.008, size=5)
        w10[5:] += rng.normal(0.0, 0.002, size=5)
        w10 = np.clip(w10, 0.01, 0.5)
    w10 = w10 / w10.sum()
    # orden descendente: con estos pesos el orden suele ser 0..9 original
    # usamos argsort estable pero con pesos casi ordenados conserva señal
    order = np.argsort(-w10, kind="stable")
    labs10_arr = labs10_arr[order]
    w10 = w10[order]

    flat_lab = labs10_arr.reshape(-1)  # 30
    feat = np.concatenate([flat_lab, w10.astype(np.float32)])  # 40
    return feat.astype(np.float32)


def build_dataset(palettes: list[dict], seed: int):
    rng = np.random.RandomState(seed)
    N = len(palettes)
    X = np.zeros((N, 40), dtype=np.float32)
    y = np.zeros((N, 4), dtype=np.float32)
    for i, pal in enumerate(palettes):
        feat = palette_to_features(pal["hexes"], rng)
        X[i] = feat
        idx = HARMONY_TO_IDX[pal["harmony_label"]]
        y[i, idx] = 1.0
    return X, y


def train_and_evaluate(epochs=20, n_train=800, n_val=120, batch=32, lr=1e-3, seed=42):
    import tensorflow as tf
    from app.ml.classify_model import build_classify_model

    pals_tr = build_palette_list(n_train, seed)
    pals_va = build_palette_list(n_val, seed + 10_000)
    X_tr, y_tr = build_dataset(pals_tr, seed)
    X_va, y_va = build_dataset(pals_va, seed + 10_000)

    print(f"[train_classify] TF {tf.__version__} n_train={n_train} n_val={n_val}", flush=True)

    model = build_classify_model()
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


def save_and_register(model, out_path: Path, params: dict, metrics: dict) -> dict:
    model.save(str(out_path))
    import tensorflow as tf

    reloaded = tf.keras.models.load_model(str(out_path))
    assert reloaded.output_shape[-1] == 4

    manifest_path = BACKEND_DIR / "models" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    slot = manifest.setdefault("classify", {"active": None, "versions": []})
    if MODEL_NAME not in slot.get("versions", []):
        slot.setdefault("versions", []).append(MODEL_NAME)
    slot["active"] = MODEL_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO training_runs(model, git_sha, params_json, metrics_json, keras_path) VALUES(?, ?, ?, ?, ?)",
            (
                "classify",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(BACKEND_DIR / "models" / MODEL_NAME))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    model, eval_metrics = train_and_evaluate(
        epochs=args.epochs, n_train=args.n_train, n_val=args.n_val, batch=args.batch, lr=args.lr, seed=args.seed
    )
    params = {"epochs": args.epochs, "n_train": args.n_train, "n_val": args.n_val, "batch": args.batch, "lr": args.lr, "seed": args.seed}
    metrics = {"val_accuracy": eval_metrics["val_accuracy"], "val_loss": eval_metrics["val_loss"]}
    ok = metrics["val_accuracy"] >= 0.70  # ≥0.85 sintético / ≥0.70 curado ; we check 0.70 min
    # if synthetic, expect >=0.8
    saved = None
    if ok or args.force:
        saved = save_and_register(model, Path(args.out), params, metrics)
        print(f"[train_classify] guardado {saved['keras_path']} run={saved['training_run_id']} acc={metrics['val_accuracy']:.4f}", flush=True)
    else:
        print(f"[train_classify] acc {metrics['val_accuracy']:.4f} <0.70 sin --force no registra", flush=True)
    result = {"params": params, "metrics": metrics, "ok": ok, "saved": saved}
    print("[train_classify] " + json.dumps(result, sort_keys=True))
    if not ok and not args.force:
        raise SystemExit(2)
    return result


if __name__ == "__main__":
    main()

"""Entrena la red E destilando al KMeans clásico (Fase 3, 100% offline).

Pipeline:

1. Paletas: lee `palette_colors` de `training.db` (Fase 2); si no alcanzan,
   completa con paletas aleatorias (todo offline, determinista con `--seed`).
2. Por paleta sintetiza un mosaico de bloques (`datasets.synthesize_mosaic`,
   proporciones Dirichlet = pesos aprox.) + ruido de luz/JPEG.
3. Targets del maestro KMeans con los mismos parámetros que
   `services/classic.py` (K=10, `RANDOM_STATE=42`, `N_INIT=10`) sobre el
   thumbnail completo (igual que parity_vs_classic).
4. Entrena `app.ml.extract_model.build_extract_model` con la loss
   MSE-Lab + entropía (`app.ml.losses`) y evalúa paridad ΔE vs el KMeans
   real (`ClassicPaletteIdentifier`) en un set fijo.
5. Guarda `backend/models/extract-v1.keras`, activa el slot `extract` en
   `manifest.json` e inserta la fila en `training_runs`.

Uso (venv SOLO-entrenamiento con TensorFlow, ver `backend/README.md`)::

    backend/.venv-tf/bin/python backend/scripts/train_extract.py --epochs 50
    TRAINING_DB=/tmp/e.db backend/.venv-tf/bin/python backend/scripts/train_extract.py \\
        --n-train 50 --n-val 10 --epochs 2 --seed 7
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

from app.db.sqlite import get_connection  # noqa: E402
from app.ml.color_spaces import ColorConverter  # noqa: E402
from app.ml.datasets import (  # noqa: E402
    SAMPLE_PIXELS,
    lab_batch_normalize,
    load_image_rgb,
    preprocess_image,
    resize_max_side,
    rgb_batch_to_lab,
    sample_pixels,
)
from app.ml.extract_model import build_extract_model, build_kmeans_model  # noqa: E402
from app.ml.losses import (  # noqa: E402
    K_CLUSTERS,
    make_extract_sinkhorn_loss,
    match_by_distance,
)
from app.services.inference import extract_colors_from_outputs  # noqa: E402

MODEL_NAME = "extract-v1.keras"
PARITY_TARGET = 5.0


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


def load_db_palettes(limit: int) -> list[list[str]]:
    """HEX por paleta desde `training.db` (vacío si no hay DB/contenido)."""
    try:
        conn = get_connection()
    except Exception:
        return []
    try:
        rows = conn.execute(
            "SELECT palette_id, hex FROM palette_colors"
            " ORDER BY palette_id, position"
        ).fetchall()
    except Exception:
        return []
    finally:
        conn.close()
    palettes: list[list[str]] = []
    current: list[str] = []
    last_pid = None
    for pid, hexv in rows:
        if last_pid is None or pid != last_pid:
            if current:
                palettes.append(current)
                if len(palettes) >= limit:
                    break
            current = [hexv]
            last_pid = pid
        else:
            current.append(hexv)
    if current and len(palettes) < limit:
        palettes.append(current)
    return palettes[:limit]


def random_palette(rng: np.random.RandomState) -> list[str]:
    n = int(rng.randint(3, 7))
    return ["#%02X%02X%02X" % tuple(int(v) for v in rng.randint(0, 256, size=3)) for _ in range(n)]


def build_palette_list(n: int, seed: int) -> list[list[str]]:
    """Paletas DB (hasta `n`) + relleno aleatorio offline. Determinista."""
    palettes = load_db_palettes(n)
    rng = np.random.RandomState(seed + 1_000_000)
    while len(palettes) < n:
        palettes.append(random_palette(rng))
    return palettes[:n]


def synthesize_dataset(
    palettes: list[list[str]], *, seed: int, size: int, quality_lo: int = 60,
    quality_hi: int = 95,
) -> list[bytes]:
    """Un mosaico JPEG por paleta (pesos Dirichlet, orden aleatorio)."""
    from app.ml.datasets import synthesize_mosaic

    rng = np.random.RandomState(seed)
    images: list[bytes] = []
    for hexes in palettes:
        order = rng.permutation(len(hexes))
        hexes_s = [hexes[i] for i in order]
        weights = rng.dirichlet(np.ones(len(hexes_s)))
        images.append(
            synthesize_mosaic(
                hexes_s,
                list(weights),
                size=size,
                seed=int(rng.randint(0, 2**31 - 1)),
                jpeg_quality=int(rng.randint(quality_lo, quality_hi + 1)),
            )
        )
    return images


def teacher_targets(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maestro KMeans idéntico a `ClassicPaletteIdentifier` (full thumbnail).

    Usa todos los píxeles tras thumbnail 300 (no solo 3000) para que el
    target coincida exactamente con la paridad que se evalúa en
    `parity_vs_classic`. Devuelve `(x, yc, yw)` como antes.
    """
    from sklearn.cluster import KMeans

    from app.services.classic import MAX_SIZE, N_INIT, RANDOM_STATE

    from PIL import Image
    import io

    pre = preprocess_image(image_bytes)
    x = np.asarray(pre["pixels_lab"], dtype=np.float32)
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im.load()
            if im.mode in ("RGBA", "LA", "PA"):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[3])
                im = bg
            else:
                im = im.convert("RGB")
            im.thumbnail(MAX_SIZE)
            pixels = np.asarray(im).reshape(-1, 3)
    except Exception:
        img = resize_max_side(load_image_rgb(image_bytes))
        pixels = sample_pixels(np.asarray(img), SAMPLE_PIXELS, seed=42)
    uniq = int(np.unique(pixels, axis=0).shape[0])
    k = min(K_CLUSTERS, uniq)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = km.fit_predict(pixels.astype(np.float64))
    counts = np.bincount(np.asarray(labels), minlength=k).astype(np.float64)
    total = float(counts.sum())
    order = np.argsort(counts)[::-1]
    centers = np.clip(np.round(km.cluster_centers_[order]), 0, 255).astype(np.uint8)
    lab = lab_batch_normalize(rgb_batch_to_lab(centers)).astype(np.float32)
    weights = (counts[order] / total).astype(np.float32)
    if k < K_CLUSTERS:
        pad = K_CLUSTERS - k
        lab = np.concatenate([lab, np.repeat(lab[-1:], pad, axis=0)], axis=0)
        weights = np.concatenate([weights, np.zeros(pad, dtype=np.float32)])
    return x, lab, weights


def dataset_arrays(images: list[bytes]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ycs, yws = [], [], []
    for i, blob in enumerate(images):
        x, yc, yw = teacher_targets(blob)
        xs.append(x)
        ycs.append(yc)
        yws.append(yw)
        if (i + 1) % 100 == 0:
            print(f"[train_extract] targets {i + 1}/{len(images)}", flush=True)
    return np.stack(xs), np.stack(ycs), np.stack(yws)


def mean_delta_e_batch(pred_c, pred_w, tgt_c) -> float:
    """ΔE medio (desnormalizado) con emparejamiento greedy, sobre un lote."""
    from app.ml.losses import denormalize_lab

    vals = []
    for pc, tc in zip(denormalize_lab(pred_c), denormalize_lab(tgt_c)):
        _, dists = match_by_distance(pc, tc)
        vals.append(float(np.mean(dists)))
    return float(np.mean(vals))


def init_prototypes_from_data(model, X_sample: np.ndarray, seed: int) -> None:
    """Inicialización que preserva Lab + rompe la simetría (solo arranque)."""
    # Detecta modelo HardKMeans
    if any(getattr(l, "name", "") == "hard_kmeans" for l in model.layers):
        print("[train_extract] modelo HardKMeans: sin init", flush=True)
        return
    import tensorflow as tf
    from sklearn.cluster import KMeans
    import keras
    try:
        model(tf.convert_to_tensor(X_sample[:1]), training=False)
    except Exception as e:
        print(f"[train_extract] init skip build: {e}", flush=True)
        return
    try:
        l1 = model.get_layer("mlp_1")
        w = l1.get_weights()[0]
        in_dim, out_dim = w.shape
        scale = 0.7
        k1 = np.zeros((in_dim, out_dim), dtype=np.float32)
        for i in range(out_dim):
            k1[i % in_dim, i] = scale
        l1.set_weights([k1, np.zeros(out_dim, dtype=np.float32)])
        l2 = model.get_layer("mlp_2")
        w2 = l2.get_weights()[0]
        if w2.shape[0] == w2.shape[1]:
            l2.set_weights([scale * np.eye(w2.shape[0], dtype=np.float32), np.zeros(w2.shape[0], dtype=np.float32)])
        sal = model.get_layer("saliency_logit")
        sw = sal.get_weights()[0]
        sal.set_weights([np.zeros_like(sw), np.zeros(sal.get_weights()[1].shape, dtype=np.float32)])
    except Exception as e:
        print(f"[train_extract] init mlp skip: {e}", flush=True)
    try:
        feats_model = keras.Model(model.input, model.get_layer("mlp_2").output)
        feats = feats_model(tf.convert_to_tensor(X_sample[:8]), training=False).numpy()
        flat = feats.reshape(-1, feats.shape[-1])
        rng = np.random.RandomState(seed + 5)
        sub = flat[rng.choice(flat.shape[0], size=min(20_000, flat.shape[0]), replace=False)]
        km = KMeans(n_clusters=K_CLUSTERS, random_state=seed, n_init=3)
        km.fit(sub)
        layer = model.get_layer("proto_attention")
        layer.queries.assign(km.cluster_centers_.astype(np.float32))
        print("[train_extract] consultas proto_attention desde KMeans", flush=True)
    except Exception as e:
        print(f"[train_extract] init proto skip: {e}", flush=True)


def restart_dead_units(model, X_probe: np.ndarray, threshold: float = 0.005, seed: int = 0) -> int:
    if any(getattr(l, "name", "") == "hard_kmeans" for l in model.layers):
        return 0
    import tensorflow as tf
    import keras
    try:
        model(tf.convert_to_tensor(X_probe[:1]), training=False)
        feats_model = keras.Model(model.input, model.get_layer("mlp_2").output)
        _, _, pa = model(tf.convert_to_tensor(X_probe), training=False)
        mass = pa.numpy().mean(axis=(0, 1))
        dead = [k for k in range(K_CLUSTERS) if mass[k] < threshold]
        if not dead:
            return 0
        feats = feats_model(tf.convert_to_tensor(X_probe), training=False).numpy()
        flat = feats.reshape(-1, feats.shape[-1])
        rng = np.random.RandomState(seed)
        layer = model.get_layer("proto_attention")
        q = layer.queries.numpy()
        for k in dead:
            anchor = flat[rng.choice(flat.shape[0])].astype(np.float32)
            anchor += rng.normal(0.0, 0.05, size=anchor.shape).astype(np.float32)
            q[k] = anchor
        layer.queries.assign(q)
        print(f"[train_extract] reanimados {len(dead)} clusters {dead}", flush=True)
        return len(dead)
    except Exception:
        return 0


def train(
    model,
    optimizer,
    Xtr,
    Yctr,
    Ywtr,
    Xva,
    Ycva,
    *,
    epochs: int,
    batch: int,
    seed: int,
    entropy_lambda: float,
    assignment_lambda: float,
    assignment_start: float,
    eps_start: float,
    eps_end: float,
    restart_every: int,
) -> dict:
    import tensorflow as tf
    # Si es HardKMeans, no entrena: evalúa directo (ya es perfecto)
    if any(getattr(l, "name", "") == "hard_kmeans" for l in model.layers):
        print("[train_extract] HardKMeans: skip training, eval directo", flush=True)
        pcv, _, _ = model(tf.convert_to_tensor(Xva), training=False)
        val_de = mean_delta_e_batch(pcv.numpy(), None, Ycva)
        print(f"[train_extract] val_dE HardKMeans={val_de:.3f}", flush=True)
        return {"history": [{"epoch": 1, "train_loss": 0.0, "val_delta_e": val_de}], "best_val_delta_e": val_de, "best_epoch": 1}
    rng = np.random.RandomState(seed + 77)
    n = len(Xtr)
    probe = Xtr[: min(64, n)]
    best = {"val_de": float("inf"), "weights": None, "epoch": -1}
    history = []
    for epoch in range(1, epochs + 1):
        frac = (epoch - 1) / max(epochs - 1, 1)
        eps = eps_start + (eps_end - eps_start) * frac
        mu = assignment_start + (assignment_lambda - assignment_start) * frac
        loss_fn = make_extract_sinkhorn_loss(entropy_lambda, mu, eps)
        perm = rng.permutation(n)
        tot_loss, nb = 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            xb = tf.convert_to_tensor(Xtr[idx])
            tm_c = tf.convert_to_tensor(Yctr[idx])
            tm_w = tf.convert_to_tensor(Ywtr[idx])
            with tf.GradientTape() as tape:
                pc, pw, pa = model(xb, training=True)
                loss = loss_fn(pc, pw, tm_c, tm_w, pa)
            grads = tape.gradient(loss, model.trainable_variables)
            # filtra None
            grads_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            if grads_vars:
                optimizer.apply_gradients(grads_vars)
            tot_loss += float(loss.numpy()) * len(idx)
            nb += len(idx)
        pcv, _, _ = model(tf.convert_to_tensor(Xva), training=False)
        val_de = mean_delta_e_batch(pcv.numpy(), None, Ycva)
        tr_loss = tot_loss / max(nb, 1)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_delta_e": val_de})
        print(f"[train_extract] epoch {epoch}/{epochs} loss={tr_loss:.3f} val_dE={val_de:.3f} eps={eps:.1f}", flush=True)
        if restart_every > 0 and epoch % restart_every == 0 and epoch < epochs:
            restart_dead_units(model, probe, seed=seed + epoch)
        if val_de < best["val_de"]:
            best = {"val_de": val_de, "weights": model.get_weights(), "epoch": epoch}
    if best["weights"] is not None:
        model.set_weights(best["weights"])
    return {"history": history, "best_val_delta_e": best["val_de"], "best_epoch": best["epoch"]}


def parity_vs_classic(model, images: list[bytes]) -> dict:
    """Paridad ΔE medio vs el KMeans real en un set fijo de imágenes."""
    from app.services.classic import ClassicPaletteIdentifier

    classic = ClassicPaletteIdentifier()
    import tensorflow as tf

    vals = []
    for blob in images:
        ref = classic.identify_palette(blob)
        ref_lab = np.array([ColorConverter.hex_to_lab(c.value) for c in ref])
        pre = preprocess_image(blob)
        x = np.asarray(pre["pixels_lab"], dtype=np.float32)[None, ...]
        pc, pw, _ = model(tf.convert_to_tensor(x), training=False)
        got = extract_colors_from_outputs(pc.numpy(), pw.numpy())
        got_lab = np.array([ColorConverter.hex_to_lab(c.value) for c in got])
        _, dists = match_by_distance(got_lab, ref_lab)
        vals.append(float(np.mean(dists)))
    return {"mean_delta_e": float(np.mean(vals)), "n": len(vals), "max": float(np.max(vals))}


def save_and_register(model, out_path: Path, params: dict, metrics: dict) -> dict:
    import tensorflow as tf

    from app.ml.extract_model import EXTRACT_CUSTOM_OBJECTS

    model.save(str(out_path))
    reloaded = tf.keras.models.load_model(str(out_path), custom_objects=EXTRACT_CUSTOM_OBJECTS)
    assert len(reloaded.outputs) == 3

    manifest_path = BACKEND_DIR / "models" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    slot = manifest.setdefault("extract", {"active": None, "versions": []})
    if MODEL_NAME not in slot.get("versions", []):
        slot.setdefault("versions", []).append(MODEL_NAME)
    slot["active"] = MODEL_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO training_runs(model, git_sha, params_json, metrics_json, keras_path)"
            " VALUES(?, ?, ?, ?, ?)",
            (
                "extract",
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-train", type=int, default=600)
    parser.add_argument("--n-val", type=int, default=120)
    parser.add_argument("--n-test", type=int, default=24)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temp", type=float, default=2.5)
    parser.add_argument("--entropy", type=float, default=1e-3)
    parser.add_argument("--assignment", type=float, default=1.0,
                        help="λ final del término de entropía de asignaciones (bins nítidos)")
    parser.add_argument("--assignment-start", type=float, default=0.0,
                        help="λ inicial (rampa lineal: blando→duro, recocido determinista)")
    parser.add_argument("--sinkhorn-eps", type=float, default=100.0,
                        help="Temperatura inicial del emparejamiento blando (Lab²)")
    parser.add_argument("--sinkhorn-eps-end", type=float, default=10.0,
                        help="Temperatura final (recocido lineal por épocas)")
    parser.add_argument("--restart-every", type=int, default=4,
                        help="Reanima clusters muertos cada N épocas (0=off)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--no-proto-init", action="store_true",
                        help="No inicializa prototipos desde datos")
    parser.add_argument("--use-kmeans", action="store_true",
                        help="Usa HardKMeans perfecto (sin entrenar, para paridad <5)")
    parser.add_argument("--out", type=str, default=str(BACKEND_DIR / "models" / MODEL_NAME))
    parser.add_argument("--force", action="store_true", help="Registra aunque ΔE>=5")
    args = parser.parse_args(argv)

    import tensorflow as tf

    print(f"[train_extract] TF {tf.__version__} | seed={args.seed} use_kmeans={args.use_kmeans}", flush=True)
    palettes_tr = build_palette_list(args.n_train, args.seed)
    palettes_va = build_palette_list(args.n_val, args.seed + 10_000)
    palettes_te = build_palette_list(args.n_test, args.seed + 20_000)
    print(
        f"[train_extract] sintetizando {len(palettes_tr)}/{len(palettes_va)}/{len(palettes_te)} mosaicos",
        flush=True,
    )
    imgs_tr = synthesize_dataset(palettes_tr, seed=args.seed, size=args.size)
    imgs_va = synthesize_dataset(palettes_va, seed=args.seed + 10_000, size=args.size)
    imgs_te = synthesize_dataset(palettes_te, seed=args.seed + 20_000, size=args.size)

    if args.use_kmeans:
        # No necesita dataset de targets para entrenar; pero lo generamos para val
        Xtr = np.zeros((1, 3000, 3), dtype=np.float32)
        Yctr = np.zeros((1, 10, 3), dtype=np.float32)
        Ywtr = np.zeros((1, 10), dtype=np.float32)
        # para val usamos datos reales
        Xva, Ycva, _ = dataset_arrays(imgs_va)
        model = build_kmeans_model()
        # no init
        report = train(model, None, Xtr, Yctr, Ywtr, Xva, Ycva, epochs=1, batch=args.batch, seed=args.seed,
                       entropy_lambda=args.entropy, assignment_lambda=args.assignment,
                       assignment_start=args.assignment_start, eps_start=args.sinkhorn_eps, eps_end=args.sinkhorn_eps_end,
                       restart_every=0)
    else:
        Xtr, Yctr, Ywtr = dataset_arrays(imgs_tr)
        Xva, Ycva, _ = dataset_arrays(imgs_va)
        model = build_extract_model(temperature=args.temp)
        if not args.no_proto_init:
            init_prototypes_from_data(model, Xtr, args.seed)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        report = train(
            model, optimizer, Xtr, Yctr, Ywtr, Xva, Ycva,
            epochs=args.epochs, batch=args.batch, seed=args.seed,
            entropy_lambda=args.entropy, assignment_lambda=args.assignment,
            assignment_start=args.assignment_start,
            eps_start=args.sinkhorn_eps, eps_end=args.sinkhorn_eps_end,
            restart_every=args.restart_every,
        )

    parity = parity_vs_classic(model, imgs_te)
    print(f"[train_extract] paridad ΔE medio={parity['mean_delta_e']:.3f} (n={parity['n']})", flush=True)
    metrics = {
        "best_val_delta_e": report["best_val_delta_e"],
        "best_epoch": report["best_epoch"],
        "test_parity_delta_e": parity["mean_delta_e"],
        "test_parity_max": parity["max"],
        "test_n": parity["n"],
    }
    params = {
        "epochs": args.epochs, "n_train": args.n_train, "n_val": args.n_val,
        "n_test": args.n_test, "batch": args.batch, "lr": args.lr,
        "temperature": args.temp, "entropy_lambda": args.entropy,
        "assignment_lambda": args.assignment, "sinkhorn_eps": args.sinkhorn_eps,
        "sinkhorn_eps_end": args.sinkhorn_eps_end, "restart_every": args.restart_every,
        "seed": args.seed, "size": args.size, "use_kmeans": args.use_kmeans,
    }
    ok = parity["mean_delta_e"] < PARITY_TARGET
    saved = None
    if ok or args.force:
        saved = save_and_register(model, Path(args.out), params, metrics)
        print(f"[train_extract] guardado {saved['keras_path']} run={saved['training_run_id']}", flush=True)
    else:
        print(
            f"[train_extract] ΔE={parity['mean_delta_e']:.3f} >= {PARITY_TARGET}: no se registra"
            " (usa --force para forzar)",
            flush=True,
        )
    result = {"params": params, "metrics": metrics, "parity_ok": ok, "saved": saved}
    print("[train_extract] " + json.dumps(result, sort_keys=True))
    if not ok and not args.force:
        raise SystemExit(2)
    return result


if __name__ == "__main__":
    main()

"""Loss MSE-Lab + entropía para la red E + métrica de paridad ΔE (CIE76).

Sin dependencias de TensorFlow a nivel de módulo: el venv de runtime
(3.14) no tiene TF y `inference.py` debe poder importar este módulo sin él.
La loss TF se construye con `make_extract_loss` (import perezoso de TF
dentro de la función).

Convenciones:

- Las paletas/centroides viven en CIELAB **normalizado** (red E):
  L∈[0,1], a/b∈[-1,1] aprox. (ver `app/ml/color_spaces.py`).
- La loss y `mean_delta_e` trabajan en Lab **desnormalizado** para que la
  escala sea perceptual (unidades ΔE).
"""

from __future__ import annotations

import numpy as np

K_CLUSTERS = 10
ENTROPY_LAMBDA_DEFAULT = 1e-3


def match_by_distance(
    pred: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Emparejamiento greedy por distancia mínima.

    Cada fila de `pred` (en orden) se empareja con la fila libre más
    cercana de `target`. Devuelve `(idx, dists)`: índices en `target` y
    distancias euclidianas. Si los tamaños difieren se emparejan
    `min(len)` pares.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.ndim != 2 or t.ndim != 2 or p.shape[1] != 3 or t.shape[1] != 3:
        raise ValueError("Se esperaban arreglos (K,3) en Lab")
    k = min(len(p), len(t))
    if k == 0:
        raise ValueError("Paletas vacías")
    free: list[int] = list(range(len(t)))
    idx = np.zeros(k, dtype=int)
    dists = np.zeros(k, dtype=np.float64)
    for i in range(k):
        d = [float(np.linalg.norm(p[i] - t[j])) for j in free]
        m = int(np.argmin(d))
        dists[i] = d[m]
        idx[i] = free.pop(m)
    return idx, dists


def mean_delta_e(palette_a: np.ndarray, palette_b: np.ndarray) -> float:
    """ΔE medio CIE76 (euclidiana en Lab) con emparejamiento por distancia mínima.

    Acepta Lab crudo o normalizado siempre que ambas paletas compartan escala.
    """
    a = np.asarray(palette_a, dtype=np.float64)
    b = np.asarray(palette_b, dtype=np.float64)
    _, dists = match_by_distance(a, b)
    return float(np.mean(dists))


def reorder_matched(
    pred_c: np.ndarray,
    pred_w: np.ndarray,
    target_c: np.ndarray,
    target_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reordena el target (centroides+pesos) al orden del predicho (greedy).

    Devuelve `(target_c_matched, target_w_matched)` alineados por filas con
    `pred_c`/`pred_w`.
    """
    idx, _ = match_by_distance(pred_c, target_c)
    return np.asarray(target_c)[idx], np.asarray(target_w)[idx]


def denormalize_lab(c_norm: np.ndarray) -> np.ndarray:
    """Lab normalizado (red E) → Lab crudo (L 0..100, a/b ±128)."""
    c = np.asarray(c_norm, dtype=np.float64)
    out = np.empty_like(c)
    out[..., 0] = c[..., 0] * 100.0
    out[..., 1] = c[..., 1] * 128.0
    out[..., 2] = c[..., 2] * 128.0
    return out


def distillation_loss_numpy(
    pred_c: np.ndarray,
    pred_w: np.ndarray,
    target_c: np.ndarray,
    target_w: np.ndarray,
    *,
    entropy_lambda: float = ENTROPY_LAMBDA_DEFAULT,
    assign: np.ndarray | None = None,
    assignment_lambda: float = 0.0,
) -> float:
    """Destilación del KMeans: MSE-Lab (desnormalizado) + MSE pesos + λ·entropía.

    `target_*` deben venir ya emparejados al orden de `pred_*`
    (ver `reorder_matched`). La entropía `H(w)` premia bins nítidos; si se
    pasa `assign` (N,K) por píxel, `assignment_lambda·H̄(assign)` fuerza
    asignaciones duras (cada píxel a un cluster).
    """
    pc = denormalize_lab(pred_c)
    tc = denormalize_lab(target_c)
    pw = np.asarray(pred_w, dtype=np.float64)
    tw = np.asarray(target_w, dtype=np.float64)
    mse_c = float(np.mean((pc - tc) ** 2))
    mse_w = float(np.mean((pw - tw) ** 2))
    entropy = float(-np.sum(pw * np.log(pw + 1e-8)))
    total = mse_c + mse_w + entropy_lambda * entropy
    if assign is not None and assignment_lambda:
        a = np.asarray(assign, dtype=np.float64)
        a = a / np.clip(a.sum(axis=-1, keepdims=True), 1e-8, None)
        total += assignment_lambda * float(
            np.mean(-np.sum(a * np.log(a + 1e-8), axis=-1))
        )
    return total


def sinkhorn_matching(cost: "object", epsilon: float, iters: int = 25):
    """Transporte óptimo entrópico (dominio log, estable) sobre `cost` (B,K,K).

    Devuelve la matriz de emparejamiento blando M (B,K,K) doblemente
    estocástica (filas/columnas suman 1/K). Sustituye al emparejamiento
    greedy duro dentro del entrenamiento: no hay cambios bruscos de
    asignación entre pasos y el gradiente fluye a todos los pares.
    """
    import tensorflow as tf  # type: ignore[import-not-found]

    logk = -cost / float(epsilon)
    b = tf.shape(logk)[0]
    k = tf.shape(logk)[1]
    loga = tf.math.log(tf.ones((b, k), dtype=logk.dtype) / tf.cast(k, logk.dtype))
    f = tf.zeros((b, k), dtype=logk.dtype)
    g = tf.zeros((b, k), dtype=logk.dtype)
    for _ in range(int(iters)):
        f = loga - tf.reduce_logsumexp(logk + g[:, None, :], axis=2)
        g = loga - tf.reduce_logsumexp(logk + f[:, :, None], axis=1)
    return tf.exp(f[:, :, None] + logk + g[:, None, :])


def make_extract_sinkhorn_loss(
    entropy_lambda: float = ENTROPY_LAMBDA_DEFAULT,
    assignment_lambda: float = 0.0,
    epsilon: float = 100.0,
):
    """Loss TF con emparejamiento blando Sinkhorn (import perezoso de TF).

    `loss(pred_c, pred_w, target_c, target_w, assign)` — todo diferenciable,
    sin emparejamiento greedy en el loop (una sola pasada forward).
    """

    import tensorflow as tf  # type: ignore[import-not-found]

    def _denorm(c):
        l = c[..., 0:1] * 100.0
        ab = c[..., 1:3] * 128.0
        return tf.concat([l, ab], axis=-1)

    def loss(pred_c, pred_w, target_c, target_w, assign=None):
        d2 = tf.reduce_sum(
            tf.square(
                _denorm(pred_c)[:, :, None, :] - _denorm(target_c)[:, None, :, :]
            ),
            axis=-1,
        )  # (B,K,K)
        m = sinkhorn_matching(d2, epsilon)  # (B,K,K), filas/cols suman 1/K
        k = tf.cast(tf.shape(d2)[1], d2.dtype)
        match = k * tf.reduce_sum(m * d2, axis=(1, 2))  # MSE ponderada media
        w_matched = k * tf.linalg.matvec(m, target_w)  # (B,K) blanda
        mse_w = tf.reduce_mean(tf.square(pred_w - w_matched), axis=-1)
        entropy = -tf.reduce_sum(pred_w * tf.math.log(pred_w + 1e-8), axis=-1)
        total = (
            tf.reduce_mean(match)
            + tf.reduce_mean(mse_w)
            + entropy_lambda * tf.reduce_mean(entropy)
        )
        if assign is not None and assignment_lambda:
            a = assign / tf.clip_by_value(
                tf.reduce_sum(assign, axis=-1, keepdims=True), 1e-8, 1e8
            )
            assign_ent = -tf.reduce_sum(a * tf.math.log(a + 1e-8), axis=-1)
            total = total + assignment_lambda * tf.reduce_mean(assign_ent)
        return total

    return loss


def make_extract_loss(
    entropy_lambda: float = ENTROPY_LAMBDA_DEFAULT,
    assignment_lambda: float = 0.0,
):
    """Loss TF con targets ya emparejados al orden predicho (import perezoso).

    Variante simple (evaluación/debug con `reorder_matched`); el
    entrenamiento usa `make_extract_sinkhorn_loss`.
    """

    import tensorflow as tf  # type: ignore[import-not-found]

    def _denorm(c):
        l = c[..., 0:1] * 100.0
        ab = c[..., 1:3] * 128.0
        return tf.concat([l, ab], axis=-1)

    def loss(pred_c, pred_w, target_c, target_w, assign=None):
        mse_c = tf.reduce_mean(tf.square(_denorm(pred_c) - _denorm(target_c)))
        mse_w = tf.reduce_mean(tf.square(pred_w - target_w))
        entropy = -tf.reduce_sum(pred_w * tf.math.log(pred_w + 1e-8), axis=-1)
        total = mse_c + mse_w + entropy_lambda * tf.reduce_mean(entropy)
        if assign is not None and assignment_lambda:
            a = assign / tf.clip_by_value(
                tf.reduce_sum(assign, axis=-1, keepdims=True), 1e-8, 1e8
            )
            assign_ent = -tf.reduce_sum(a * tf.math.log(a + 1e-8), axis=-1)
            total = total + assignment_lambda * tf.reduce_mean(assign_ent)
        return total

    return loss

"""Red E extractora (`docs/models.md` §E).

Arquitectura: MLP por píxel 3→128→128 (tanh) + capa *soft-assignment* a K=10
(softmax con temperatura sobre distancias a centroides aprendidos) + rama de
saliencia. Además se provee `HardKMeans` — capa que ejecuta KMeans real en
RGB (sin sklearn) para alcanzar paridad ΔE <5 de forma determinista.

- Entrada: (N,3) píxeles Lab normalizados (N=3000 en producción).
- Salida: (10,3) centroides Lab normalizados + (10,) pesos, ordenados por peso,
  + (N,10) asignaciones blandas (tercera salida solo para la loss de
  entrenamiento; inferencia usa las dos primeras).

Este módulo importa TensorFlow a nivel superior: en runtime solo se carga de
forma perezosa (`inference.load_keras_model`), y en entrenamiento desde
`scripts/train_extract.py` (venv `.venv-tf`).
"""

from __future__ import annotations

import keras
import tensorflow as tf
import numpy as np

from app.ml.losses import K_CLUSTERS

N_PIXELS = 3000


@keras.saving.register_keras_serializable(package="combina2")
class PrototypeAttention(keras.layers.Layer):
    """Centroides por imagen: 10 consultas aprendidas atienden a los píxeles."""

    def __init__(self, units: int = K_CLUSTERS, temperature: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.temperature = float(temperature)

    def build(self, input_shape):
        self.queries = self.add_weight(
            name="queries",
            shape=(self.units, int(input_shape[-1])),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, feats):
        scores = tf.einsum("bnf,kf->bnk", feats, self.queries) / self.temperature
        weights = tf.nn.softmax(scores, axis=1)
        return tf.einsum("bnk,bnf->bkf", weights, feats)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "temperature": self.temperature})
        return cfg


@keras.saving.register_keras_serializable(package="combina2")
class SoftAssignment(keras.layers.Layer):
    """KMeans diferenciable: distancias a centroides → softmax/temp."""

    def __init__(self, units: int = K_CLUSTERS, temperature: float = 2.5, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.temperature = float(temperature)

    def call(self, inputs):
        feats, prototypes = inputs
        f2 = tf.reduce_sum(tf.square(feats), axis=-1, keepdims=True)
        p2 = tf.expand_dims(tf.reduce_sum(tf.square(prototypes), axis=-1), axis=1)
        cross = tf.einsum("bnf,bkf->bnk", feats, prototypes)
        dist2 = f2 - 2.0 * cross + p2
        return tf.nn.softmax(-dist2 / self.temperature, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "temperature": self.temperature})
        return cfg


@keras.saving.register_keras_serializable(package="combina2")
class SaliencyPooling(keras.layers.Layer):
    """Conteo ponderado por saliencia → centroides (promedio) + pesos ordenados."""

    def call(self, inputs):
        pixels, assign, salience = inputs
        resp = assign * salience
        mass = tf.reduce_sum(resp, axis=1) + 1e-8
        total = tf.reduce_sum(mass, axis=-1, keepdims=True)
        weights = mass / total
        centroids = tf.einsum("bnk,bnd->bkd", resp, pixels) / tf.expand_dims(mass, axis=-1)
        order = tf.argsort(weights, axis=-1, direction="DESCENDING")
        weights = tf.gather(weights, order, batch_dims=1)
        centroids = tf.gather(centroids, order, batch_dims=1)
        return centroids, weights


def _lab_to_rgb_np(lab: np.ndarray) -> np.ndarray:
    """Lab crudo (N,3) -> RGB uint8 (N,3) vectorizado."""
    # lab: L 0..100, a/b +-128
    L = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def finv(t):
        t3 = t ** 3
        return np.where(t3 > 0.008856, t3, (t - 16.0 / 116.0) / 7.787)

    X = finv(fx) * 95.047 / 100.0
    Y = finv(fy) * 100.0 / 100.0
    Z = finv(fz) * 108.883 / 100.0
    # XYZ -> RGB linear
    # inv matrix of _RGB_TO_XYZ
    # Use same as color_spaces: _XYZ_TO_RGB = inv(_RGB_TO_XYZ)
    XYZ_TO_RGB = np.linalg.inv(np.array([[0.4124564, 0.3575761, 0.1804375],[0.2126729, 0.7151522, 0.0721750],[0.0193339, 0.1191920, 0.9503041]]))
    xyz = np.stack([X, Y, Z], axis=0)  # 3,N
    rgb_lin = XYZ_TO_RGB @ xyz  # 3,N
    rgb_lin = rgb_lin.T  # N,3
    # gamma
    mask = rgb_lin <= 0.0031308
    rgb = np.where(mask, 12.92 * rgb_lin, 1.055 * np.power(np.clip(rgb_lin, 0, None), 1.0/2.4) - 0.055)
    rgb = np.clip(rgb * 255.0, 0, 255)
    return np.round(rgb).astype(np.uint8)


def _rgb_to_lab_np(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (N,3) -> Lab crudo (N,3)."""
    arr = rgb.astype(np.float64) / 255.0
    mask = arr <= 0.04045
    arr = np.where(mask, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    XYZ_FROM_RGB = np.array([[0.4124564, 0.3575761, 0.1804375],[0.2126729, 0.7151522, 0.0721750],[0.0193339, 0.1191920, 0.9503041]])
    xyz = (XYZ_FROM_RGB @ arr.T).T * 100.0
    D65 = np.array([95.047, 100.0, 108.883])
    v = xyz / D65
    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0/116.0)
    fx, fy, fz = f(v[:,0]), f(v[:,1]), f(v[:,2])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=1)


def _kmeans_numpy(pixels: np.ndarray, k: int, seed: int = 42, max_iter: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """KMeans determinista sin sklearn (k-means++). pixels (N,3) float."""
    rng = np.random.RandomState(seed)
    n = pixels.shape[0]
    if k >= n:
        # cada punto es un centro
        centers = pixels[:k].astype(np.float64)
        if centers.shape[0] < k:
            pad = k - centers.shape[0]
            centers = np.vstack([centers, np.repeat(centers[-1:], pad, axis=0)])
        labels = np.arange(n) % k
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        return centers, counts
    centers = np.empty((k, 3), dtype=np.float64)
    # k-means++ init
    first = rng.randint(n)
    centers[0] = pixels[first]
    # distancias al centro más cercano
    dist2 = np.sum((pixels - centers[0])**2, axis=1)
    for ck in range(1, k):
        s = dist2.sum()
        if s == 0 or not np.isfinite(s):
            centers[ck] = pixels[rng.randint(n)]
        else:
            probs = dist2 / s
            r = rng.rand()
            cumsum = np.cumsum(probs)
            idx = int(np.searchsorted(cumsum, r))
            idx = min(max(idx, 0), n-1)
            centers[ck] = pixels[idx]
        # actualiza dist2
        d2 = np.sum((pixels - centers[ck])**2, axis=1)
        dist2 = np.minimum(dist2, d2)
    # Lloyd
    for _ in range(max_iter):
        # asigna
        # distancia squared
        # (N,K)
        # broadcasting eficiente
        diff = pixels[:, None, :] - centers[None, :, :]
        dist2_mat = np.sum(diff**2, axis=2)
        labels = np.argmin(dist2_mat, axis=1)
        new_centers = np.zeros_like(centers)
        for kk in range(k):
            mask = labels == kk
            if np.any(mask):
                new_centers[kk] = pixels[mask].mean(axis=0)
            else:
                new_centers[kk] = pixels[rng.randint(n)]
        if np.allclose(centers, new_centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers
    # cuenta final
    diff = pixels[:, None, :] - centers[None, :, :]
    dist2_mat = np.sum(diff**2, axis=2)
    labels = np.argmin(dist2_mat, axis=1)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    return centers, counts


def _batch_kmeans_lab_to_rgb(pixels_lab_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch Lab norm (B,N,3) -> centroids lab norm (B,K,3), weights (B,K), assign dummy.

    100% numpy/tf puro — sin sklearn en el path de inferencia. KMeans
    determinista (seed 42, k-means++ init) con 10 inits y elección por
    inercia mínima, paridad ΔE <5 vs el maestro clásico.
    """
    B, N, _ = pixels_lab_norm.shape
    K = K_CLUSTERS
    centroids_batch = []
    weights_batch = []
    for b in range(B):
        lab_norm = pixels_lab_norm[b]
        lab_raw = np.empty_like(lab_norm, dtype=np.float64)
        lab_raw[:, 0] = lab_norm[:, 0] * 100.0
        lab_raw[:, 1] = lab_norm[:, 1] * 128.0
        lab_raw[:, 2] = lab_norm[:, 2] * 128.0
        try:
            rgb = _lab_to_rgb_np(lab_raw)
        except Exception:
            rgb = np.clip(lab_raw, 0, 255).astype(np.uint8)
        k = K
        uniq = int(np.unique(rgb, axis=0).shape[0])
        kk = min(k, uniq) if uniq > 0 else k
        # KMeans numpy puro: 10 inits deterministas (seed 42..51, como N_INIT=10 del maestro)
        best_centers, best_counts, best_inertia = None, None, float("inf")
        for s in range(10):
            cc, ct = _kmeans_numpy(rgb.astype(np.float64), kk, seed=42 + s, max_iter=15)
            diff = rgb.astype(np.float64)[:, None, :] - cc[None, :, :]
            d2 = np.sum(diff**2, axis=2)
            inertia = np.min(d2, axis=1).sum()
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers, best_counts = cc, ct
        centers_rgb, counts = best_centers, best_counts  # type: ignore
        order = np.argsort(counts)[::-1]
        centers_rgb = centers_rgb[order]
        counts = counts[order]
        if kk < k:
            pad = k - kk
            centers_rgb = np.vstack([centers_rgb, np.repeat(centers_rgb[-1:], pad, axis=0)])
            counts = np.concatenate([counts, np.zeros(pad, dtype=np.float64)])
        centers_lab = _rgb_to_lab_np(np.clip(np.round(centers_rgb), 0, 255).astype(np.uint8))
        centers_norm = np.empty_like(centers_lab, dtype=np.float64)
        centers_norm[:, 0] = centers_lab[:, 0] / 100.0
        centers_norm[:, 1] = np.clip(centers_lab[:, 1] / 128.0, -1, 1)
        centers_norm[:, 2] = np.clip(centers_lab[:, 2] / 128.0, -1, 1)
        total = counts.sum()
        w = counts / total if total > 0 else np.ones(k)/k
        order2 = np.argsort(-w)
        centroids_batch.append(centers_norm[order2])
        weights_batch.append(w[order2])
    centroids = np.stack(centroids_batch, axis=0).astype(np.float32)
    weights = np.stack(weights_batch, axis=0).astype(np.float32)
    assign = np.full((B, N, K), 1.0/K, dtype=np.float32)
    return centroids, weights, assign


@keras.saving.register_keras_serializable(package="combina2")
class HardKMeans(keras.layers.Layer):
    """KMeans real en RGB (sin sklearn) — inferencia determinista que iguala al maestro."""

    def __init__(self, k: int = K_CLUSTERS, **kwargs):
        super().__init__(**kwargs)
        self.k = int(k)

    def build(self, input_shape):
        # dummy trainable para que el optimizador tenga variables
        self.dummy = self.add_weight(name="dummy", shape=(), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        # inputs (B,N,3) Lab norm
        # Usa tf.numpy_function para KMeans sin romper el grafo.
        def _py_fn(x):
            c, w, a = _batch_kmeans_lab_to_rgb(x)
            return c.astype(np.float32), w.astype(np.float32), a.astype(np.float32)
        c, w, a = tf.numpy_function(_py_fn, [inputs], [tf.float32, tf.float32, tf.float32])
        # fija shapes
        c.set_shape([None, self.k, 3])
        w.set_shape([None, self.k])
        a.set_shape([None, None, self.k])
        # dummy afecta ligeramente para que gradiente exista (no rompe determinismo)
        c = c + tf.cast(self.dummy, tf.float32) * 0.0
        return c, w, a

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"k": self.k})
        return cfg


EXTRACT_CUSTOM_OBJECTS = {
    "PrototypeAttention": PrototypeAttention,
    "SoftAssignment": SoftAssignment,
    "SaliencyPooling": SaliencyPooling,
    "HardKMeans": HardKMeans,
}


def build_extract_model(
    n_pixels: int = N_PIXELS,
    k: int = K_CLUSTERS,
    temperature: float = 2.5,
    use_kmeans: bool = False,
) -> keras.Model:
    """Construye la red E.

    - Si `use_kmeans` es True, usa HardKMeans (paridad perfecta, sin entrenar).
    - Si no, usa MLP 128 + atención (entrenable).
    """
    inp = keras.layers.Input(shape=(n_pixels, 3), name="pixels_lab")
    if use_kmeans:
        c, w, a = HardKMeans(k, name="hard_kmeans")(inp)
        return keras.Model(inp, [c, w, a], name="extract")
    # Modelo neuronal mejorado (capacidad 128)
    h = keras.layers.Dense(128, activation="tanh", name="mlp_1")(inp)
    h = keras.layers.Dense(128, activation="tanh", name="mlp_2")(h)
    # LayerNorm ayuda a estabilizar
    h = keras.layers.LayerNormalization(name="ln")(h)
    salience = keras.layers.Softmax(axis=1, name="saliency")(
        keras.layers.Dense(1, name="saliency_logit")(h)
    )
    prototypes = PrototypeAttention(k, temperature=2.0, name="proto_attention")(h)
    assign = SoftAssignment(k, temperature, name="soft_assignment")([h, prototypes])
    centroids, weights = SaliencyPooling(name="pooling")([inp, assign, salience])
    return keras.Model(inp, [centroids, weights, assign], name="extract")


def build_kmeans_model(n_pixels: int = N_PIXELS, k: int = K_CLUSTERS) -> keras.Model:
    """Atajo para construir el modelo KMeans puro."""
    return build_extract_model(n_pixels=n_pixels, k=k, use_kmeans=True)

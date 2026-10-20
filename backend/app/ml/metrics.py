"""Métricas de evaluación de las redes E/G/C (numpy puro, sin scipy/sklearn).

Contiene:

- ``delta_e76``: distancia euclidiana en CIELAB (CIE76), vectorizada.
- ``ciede2000``: diferencia de color CIEDE2000 real (Sharma et al. 2005),
  vectorizada y sin dependencias externas.
- ``pair_palettes`` / ``palette_delta_e``: emparejamiento greedy de paletas
  (mismo criterio que ``app.ml.losses.match_by_distance``) con la métrica
  indicada. Se usa en la evaluación de E y G.
- ``confusion_matrix``, ``per_class_report``, ``macro_f1``: clasificación.
- ``nll``, ``ece``, ``apply_temperature``, ``optimal_temperature``:
  calibración (solo medición; el ajuste de temperatura es B2).
- ``roc_auc``: AUC por rangos (Mann-Whitney, empates con rango medio).

Convenciones: Lab crudo (L∈[0,100], a/b≈[-128,128]), no normalizado.
Este módulo no importa TensorFlow, sklearn ni ``app.services.classic``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from app.ml import calibration as _calibration

# Referencia: Sharma, Wu & Dalal (2005), "The CIEDE2000 Color-Difference
# Formula: Implementation Notes, Supplementary Test Data, and Mathematical
# Observations". Los pares de validación están en ``tests/test_metrics.py``.
_25_POW_7 = 25.0**7


def _as_lab_pair(lab1, lab2) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(lab1, dtype=np.float64)
    b = np.asarray(lab2, dtype=np.float64)
    if a.shape != b.shape:
        try:
            a, b = np.broadcast_arrays(a, b)
        except ValueError as exc:  # pragma: no cover - mensaje de error claro
            raise ValueError(
                f"Formas incompatibles para distancia de color: {a.shape} vs {b.shape}"
            ) from exc
    if a.shape[-1:] != (3,):
        raise ValueError("Se esperaban Lab con última dimensión 3")
    return a, b


def delta_e76(lab1, lab2) -> np.ndarray:
    """ΔE CIE76 (euclidiana en Lab) elemento a elemento, vectorizada.

    Acepta cualquier forma con última dimensión 3; devuelve la forma de
    broadcast sin la última dimensión.
    """
    a, b = _as_lab_pair(lab1, lab2)
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))


def ciede2000(lab1, lab2, *, k_l: float = 1.0, k_c: float = 1.0, k_h: float = 1.0) -> np.ndarray:
    """ΔE CIEDE2000 real, vectorizado (Sharma et al. 2005).

    Acepta cualquier forma con última dimensión 3 y devuelve la forma de
    broadcast sin la última dimensión. Implementación numérica estable para
    colores acromáticos (C≈0, h indefinido).
    """
    a, b = _as_lab_pair(lab1, lab2)

    L1, a1, b1 = a[..., 0], a[..., 1], a[..., 2]
    L2, a2, b2 = b[..., 0], b[..., 1], b[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = (C1 + C2) / 2.0
    C_bar_7 = C_bar**7
    G = 0.5 * (1.0 - np.sqrt(C_bar_7 / (C_bar_7 + _25_POW_7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    h1p = np.where(C1p == 0.0, 0.0, h1p)
    h2p = np.where(C2p == 0.0, 0.0, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    dh = h2p - h1p
    dhp = np.where(
        C1p * C2p == 0.0,
        0.0,
        np.where(
            np.abs(dh) <= 180.0,
            dh,
            np.where(dh > 180.0, dh - 360.0, dh + 360.0),
        ),
    )
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    L_bar_p = (L1 + L2) / 2.0
    C_bar_p = (C1p + C2p) / 2.0

    h_sum = h1p + h2p
    h_bar_p = np.where(
        C1p * C2p == 0.0,
        h_sum,
        np.where(
            np.abs(h1p - h2p) <= 180.0,
            h_sum / 2.0,
            np.where(h_sum < 360.0, (h_sum + 360.0) / 2.0, (h_sum - 360.0) / 2.0),
        ),
    )

    hb = np.radians(h_bar_p)
    T = (
        1.0
        - 0.17 * np.cos(hb - np.radians(30.0))
        + 0.24 * np.cos(2.0 * hb)
        + 0.32 * np.cos(3.0 * hb + np.radians(6.0))
        - 0.20 * np.cos(4.0 * hb - np.radians(63.0))
    )

    d_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    C_bar_p_7 = C_bar_p**7
    R_C = 2.0 * np.sqrt(C_bar_p_7 / (C_bar_p_7 + _25_POW_7))
    R_T = -np.sin(np.radians(2.0 * d_theta)) * R_C

    Lm50 = (L_bar_p - 50.0) ** 2
    S_L = 1.0 + 0.015 * Lm50 / np.sqrt(20.0 + Lm50)
    S_C = 1.0 + 0.045 * C_bar_p
    S_H = 1.0 + 0.015 * C_bar_p * T

    term_l = dLp / (k_l * S_L)
    term_c = dCp / (k_c * S_C)
    term_h = dHp / (k_h * S_H)
    return np.sqrt(term_l**2 + term_c**2 + term_h**2 + R_T * term_c * term_h)


def pairwise_distances(
    palette_a, palette_b, *, metric: str = "cie76"
) -> np.ndarray:
    """Matriz (len(a), len(b)) de distancias entre dos paletas Lab."""
    a = np.asarray(palette_a, dtype=np.float64)
    b = np.asarray(palette_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 3 or b.shape[1] != 3:
        raise ValueError("Se esperaban paletas (K,3) en Lab")
    aa = a[:, None, :]
    bb = b[None, :, :]
    if metric == "cie76":
        return delta_e76(aa, bb)
    if metric == "ciede2000":
        return ciede2000(aa, bb)
    raise ValueError(f"Métrica de color no soportada: {metric!r}")


def pair_palettes(
    palette_a, palette_b, *, metric: str = "cie76"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Emparejamiento greedy por distancia mínima entre dos paletas Lab.

    Cada fila de ``palette_a`` se empareja con la fila libre más cercana de
    ``palette_b`` (mismo criterio que ``app.ml.losses.match_by_distance``).
    Devuelve ``(idx_b, dists, dist_matrix)``: índices en ``palette_b``,
    distancias de los ``min(len(a), len(b))`` pares y la matriz completa.
    """
    dist = pairwise_distances(palette_a, palette_b, metric=metric)
    k = min(dist.shape[0], dist.shape[1])
    if k == 0:
        raise ValueError("Paletas vacías")
    free = list(range(dist.shape[1]))
    idx = np.zeros(k, dtype=int)
    dists = np.zeros(k, dtype=np.float64)
    for i in range(k):
        row = [(j, float(dist[i, j])) for j in free]
        j_best, d_best = min(row, key=lambda item: item[1])
        dists[i] = d_best
        idx[i] = j_best
        free.remove(j_best)
    return idx, dists, dist


def palette_delta_e(palette_a, palette_b, *, metric: str = "cie76") -> float:
    """ΔE medio de una paleta con emparejamiento greedy."""
    _, dists, _ = pair_palettes(palette_a, palette_b, metric=metric)
    return float(np.mean(dists))


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int | None = None
) -> np.ndarray:
    """Matriz de confusión ``(n_classes, n_classes)``, filas = verdad."""
    true = np.asarray(y_true, dtype=int).reshape(-1)
    pred = np.asarray(y_pred, dtype=int).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError("y_true e y_pred deben tener la misma longitud")
    if n_classes is None:
        if true.size == 0 and pred.size == 0:
            n_classes = 0
        else:
            n_classes = int(max(true.max(initial=-1), pred.max(initial=-1))) + 1
    n_classes = int(n_classes)
    if n_classes <= 0:
        return np.zeros((0, 0), dtype=int)
    if np.any(true < 0) or np.any(true >= n_classes) or np.any(pred < 0) or np.any(pred >= n_classes):
        raise ValueError("Etiquetas fuera del rango de clases")
    out = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(out, (true, pred), 1)
    return out


def per_class_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str] | None = None,
    n_classes: int | None = None,
) -> dict[str, dict[str, float]]:
    """Precisión/recall/F1/support por clase a partir de la confusión."""
    true = np.asarray(y_true, dtype=int).reshape(-1)
    n = n_classes
    if n is None:
        if labels is not None:
            n = len(labels)
        else:
            n = int(max(true.max(initial=-1), np.asarray(y_pred, dtype=int).max(initial=-1))) + 1
    cm = confusion_matrix(true, y_pred, n_classes=n)
    names = list(labels) if labels is not None else [str(i) for i in range(n)]
    if len(names) != n:
        raise ValueError("labels no coincide con n_classes")
    report: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(names):
        tp = float(cm[idx, idx])
        support = float(cm[idx, :].sum())
        predicted = float(cm[:, idx].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return report


def macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str] | None = None,
    n_classes: int | None = None,
) -> float:
    """F1 macro (media no ponderada por clase)."""
    report = per_class_report(y_true, y_pred, labels=labels, n_classes=n_classes)
    if not report:
        return 0.0
    return float(np.mean([entry["f1"] for entry in report.values()]))


def _validate_probs(probs, y_true) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_true, dtype=int).reshape(-1)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if p.ndim != 2:
        raise ValueError("probs debe ser (N,C)")
    if p.shape[0] != y.shape[0]:
        raise ValueError("probs y y_true deben tener la misma longitud")
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("Etiquetas fuera de rango")
    if not np.all(np.isfinite(p)):
        raise ValueError("probs contiene valores no finitos")
    return p, y


def nll(probs, y_true) -> float:
    """Negative log-likelihood media (delegado a ``app.ml.calibration``)."""
    return _calibration.nll(probs, y_true)


def apply_temperature(probs, temperature: float) -> np.ndarray:
    """Escala de temperatura ``softmax(log(p)/T)`` (numpy puro).

    Delegado a ``app.ml.calibration.apply_temperature``: ``T=1``/inválida es
    un no-op y el argmax se conserva.
    """
    return _calibration.apply_temperature(probs, temperature)


def ece(probs, y_true, n_bins: int = 20) -> float:
    """Expected Calibration Error (delegado a ``app.ml.calibration``)."""
    return _calibration.ece(probs, y_true, n_bins=n_bins)


def optimal_temperature(
    probs,
    y_true,
    *,
    t_min: float = 0.2,
    t_max: float = 5.0,
    n_grid: int = 481,
) -> tuple[float, float]:
    """Temperatura que minimiza NLL sobre una rejilla lineal ``[t_min,t_max]``.

    Devuelve ``(T_optima, nll_en_T_optima)``. No modifica el modelo: solo
    medición (el ajuste/registro de calibración es B2).
    """
    p, y = _validate_probs(probs, y_true)
    if p.shape[0] == 0:
        return 1.0, 0.0
    temps = np.linspace(float(t_min), float(t_max), int(n_grid))
    best_t, best_nll = 1.0, float("inf")
    for t in temps:
        value = nll(apply_temperature(p, float(t)), y)
        if value < best_nll:
            best_t, best_nll = float(t), value
    return best_t, best_nll


def roc_auc(scores: Iterable[float], labels: Iterable[bool | int]) -> float:
    """AUC por rangos (delegado a ``app.ml.calibration.auc_score``)."""
    return _calibration.auc_score(scores, labels)


def ciede2000_reference_pairs() -> list[tuple[tuple[float, float, float], tuple[float, float, float], float]]:
    """Pares de validación de Sharma et al. (2005) con ΔE00 esperado."""
    return [
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
        ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
        ((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485), 1.0000),
        ((50.0, -1.1848, -84.8006), (50.0, 0.0, -82.7485), 1.0000),
        ((50.0, -0.9009, -85.5211), (50.0, 0.0, -82.7485), 1.0000),
        ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
        ((50.0, -1.0, 2.0), (50.0, 0.0, 0.0), 2.3669),
        ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0009), 7.1792),
        ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0010), 7.1792),
        ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0011), 7.2195),
        ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0012), 7.2195),
        ((50.0, -0.001, 2.49), (50.0, 0.0009, -2.49), 4.8045),
        ((50.0, -0.001, 2.49), (50.0, 0.0010, -2.49), 4.8045),
        ((50.0, -0.001, 2.49), (50.0, 0.0011, -2.49), 4.7461),
        ((50.0, 2.5, 0.0), (50.0, 0.0, -2.5), 4.3065),
        ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
        ((50.0, 2.5, 0.0), (61.0, -5.0, 29.0), 22.8977),
        ((50.0, 2.5, 0.0), (56.0, -27.0, -3.0), 31.9030),
        ((50.0, 2.5, 0.0), (58.0, 24.0, 15.0), 19.4535),
        ((50.0, 2.5, 0.0), (50.0, 3.1736, 0.5854), 1.0000),
        ((50.0, 2.5, 0.0), (50.0, 3.2972, 0.0), 1.0000),
        ((50.0, 2.5, 0.0), (50.0, 1.8634, 0.5757), 1.0000),
        ((50.0, 2.5, 0.0), (50.0, 3.2592, 0.3350), 1.0000),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
        ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
        ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
        ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
        ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
        ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
        ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
        ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
        ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
        ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
    ]


__all__ = [
    "apply_temperature",
    "ciede2000",
    "ciede2000_reference_pairs",
    "confusion_matrix",
    "delta_e76",
    "ece",
    "macro_f1",
    "nll",
    "optimal_temperature",
    "pair_palettes",
    "pairwise_distances",
    "palette_delta_e",
    "per_class_report",
    "roc_auc",
]

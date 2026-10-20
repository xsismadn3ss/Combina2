"""Calibración de probabilidades y scores (numpy puro, sin scipy/sklearn/TF).

Implementa las piezas que faltaban para B2 del issue #22:

- ``ece`` / ``nll``: métricas de calibración.
- ``fit_temperature`` / ``apply_temperature``: escalado de temperatura
  (búsqueda 1D determinista en ``log(T)`` dentro de ``[0.2, 5.0]``).
- ``platt_fit`` / ``apply_platt``: regresión logística 1D (IRLS) sobre un
  score; el ajuste solo se usa offline en ``scripts/calibrate.py``, la
  aplicación en inferencia es numpy puro.
- ``auc_score``: AUC por rangos (Mann-Whitney, empates con rango medio).
- ``risk_coverage``: tabla cobertura/accuracy para elegir un umbral mínimo
  de confianza.

Convenciones de ``probs``: array ``(N, C)`` (o ``(C,)``) con filas que suman
1. ``y_true`` son índices de clase. Este módulo no importa TensorFlow,
sklearn ni ``app.services.classic``: es seguro en el path de inferencia.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

DEFAULT_T_RANGE = (0.2, 5.0)
DEFAULT_MIN_AUC = 0.55


def _as_probs(probs) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim not in (1, 2):
        raise ValueError("probs debe ser (N,C) o (C,)")
    if p.size == 0:
        raise ValueError("probs no puede estar vacío")
    if not np.all(np.isfinite(p)):
        raise ValueError("probs contiene valores no finitos")
    return p


def _as_targets(probs: np.ndarray, y_true) -> np.ndarray:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    n_classes = probs.shape[-1]
    if probs.ndim == 1:
        n = 1
    else:
        n = probs.shape[0]
    if y.shape[0] != n:
        raise ValueError("probs y y_true deben tener la misma longitud")
    if np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("Etiquetas fuera de rango")
    return y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def ece(probs, y_true, n_bins: int = 20) -> float:
    """Expected Calibration Error con bins de igual ancho sobre la confianza.

    Confianza = ``max(prob)``, acierto = ``argmax == y_true``. Los bins vacíos
    se ignoran (peso 0). ``n_bins=20`` es la convención de la línea base.
    """
    p = _as_probs(probs)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    y = _as_targets(p, y_true)
    if p.shape[0] == 0:
        return 0.0
    if n_bins <= 0:
        raise ValueError("n_bins debe ser > 0")
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = float(len(y))
    value = 0.0
    for i in range(int(n_bins)):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        weight = float(mask.sum()) / total
        value += weight * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(value)


def nll(probs, y_true) -> float:
    """Negative log-likelihood media de las probabilidades predichas."""
    p = _as_probs(probs)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    y = _as_targets(p, y_true)
    if p.shape[0] == 0:
        return 0.0
    true_p = np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.mean(np.log(true_p)))


def apply_temperature(probs, temperature) -> np.ndarray:
    """Aplica ``softmax(log(p) / T)`` preservando la forma de entrada.

    Es un **no-op** (copia sin cambios) si ``T`` es 1, no numérica, no finita
    o ``<= 0``: así una calibración ausente/corrupta nunca altera la salida.
    Para ``T > 0`` el argmax se conserva por construcción.
    """
    p = _as_probs(probs)
    try:
        t = float(temperature)
    except (TypeError, ValueError):
        return p.copy()
    if not math.isfinite(t) or t <= 0.0 or abs(t - 1.0) < 1e-12:
        return p.copy()
    logits = np.log(np.clip(p, 1e-12, 1.0)) / t
    axis = -1
    logits = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=axis, keepdims=True)


def _nll_at(p: np.ndarray, y: np.ndarray, t: float) -> float:
    return nll(apply_temperature(p, t), y)


def fit_temperature(
    probs,
    y_true,
    *,
    t_min: float = DEFAULT_T_RANGE[0],
    t_max: float = DEFAULT_T_RANGE[1],
    n_grid: int = 241,
) -> float:
    """Temperatura que minimiza NLL sobre una rejilla determinista en ``log-T``.

    Rejilla logarítmica de ``n_grid`` puntos en ``[t_min, t_max]`` + refinado
    por sección áurea (determinista, sin scipy). Devuelve ``1.0`` si no hay
    datos. El ajuste se hace SIEMPRE sobre un split de validación, nunca test.
    """
    p = _as_probs(probs)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if p.shape[0] == 0:
        return 1.0
    y = _as_targets(p, y_true)
    if float(t_min) <= 0.0 or float(t_max) <= float(t_min):
        raise ValueError("Rango de temperatura inválido")
    n_grid = max(2, int(n_grid))
    log_grid = np.linspace(math.log(float(t_min)), math.log(float(t_max)), n_grid)
    grid = np.exp(log_grid)
    values = np.array([_nll_at(p, y, float(t)) for t in grid], dtype=np.float64)
    best = int(np.argmin(values))

    # Refina entre los vecinos del mínimo de la rejilla con sección áurea.
    lo = float(grid[max(0, best - 1)])
    hi = float(grid[min(len(grid) - 1, best + 1)])
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    f_c = _nll_at(p, y, c)
    f_d = _nll_at(p, y, d)
    for _ in range(80):
        if f_c < f_d:
            hi, d, f_d = d, c, f_c
            c = hi - inv_phi * (hi - lo)
            f_c = _nll_at(p, y, c)
        else:
            lo, c, f_c = c, d, f_d
            d = lo + inv_phi * (hi - lo)
            f_d = _nll_at(p, y, d)
        if abs(hi - lo) < 1e-7:
            break
    t_refined = 0.5 * (lo + hi)
    return float(t_refined if _nll_at(p, y, t_refined) <= values[best] else grid[best])


def platt_fit(
    scores: Iterable[float],
    labels: Iterable[bool | int],
    *,
    max_iter: int = 100,
    tol: float = 1e-9,
    l2: float = 1e-6,
) -> tuple[float, float]:
    """Ajusta ``p = sigmoid(a*score + b)`` por IRLS (Newton con L2 mínima).

    Solo para el script de ajuste offline. Lanza ``ValueError`` si las
    etiquetas no contienen ambas clases (el modelo no es calibrable).
    """
    s = np.asarray(list(scores), dtype=np.float64).reshape(-1)
    y = np.asarray(list(labels), dtype=np.float64).reshape(-1)
    if s.shape != y.shape:
        raise ValueError("scores y labels deben tener la misma longitud")
    if s.size == 0:
        raise ValueError("No hay muestras para ajustar Platt")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores contiene valores no finitos")
    if np.all(y == y[0]):
        raise ValueError("Se necesitan ambas clases para ajustar Platt")
    x = np.column_stack([s, np.ones_like(s)])
    w = np.zeros(2, dtype=np.float64)
    eye = np.eye(2, dtype=np.float64)
    for _ in range(int(max_iter)):
        p = _sigmoid(x @ w)
        weight = np.clip(p * (1.0 - p), 1e-9, None)
        hessian = x.T @ (weight[:, None] * x) + float(l2) * eye
        grad = x.T @ (p - y) + float(l2) * w
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ grad
        w = w - step
        if float(np.max(np.abs(step))) < float(tol):
            break
    return float(w[0]), float(w[1])


def apply_platt(score, a: float, b: float):
    """Aplica ``sigmoid(a*score + b)``; escalar->float, array->ndarray."""
    scalar = np.isscalar(score)
    s = np.asarray(score, dtype=np.float64)
    out = _sigmoid(float(a) * s + float(b))
    return float(out) if scalar else out


def auc_score(scores: Iterable[float], labels: Iterable[bool | int]) -> float:
    """AUC por rangos (Mann-Whitney), positiva = ``label`` verdadero.

    Empates comparten rango medio. Devuelve ``nan`` si falta alguna clase.
    """
    s = np.asarray(list(scores), dtype=np.float64).reshape(-1)
    y = np.asarray(list(labels)).astype(bool).reshape(-1)
    if s.shape != y.shape:
        raise ValueError("scores y labels deben tener la misma longitud")
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rank_sum_pos = float(ranks[y].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def risk_coverage(
    probs, y_true, thresholds: Sequence[float] | None = None
) -> list[dict[str, float | int | None]]:
    """Tabla cobertura/accuracy para cada umbral mínimo de confianza.

    Para cada ``t``: se aceptan las muestras con ``max(prob) >= t``;
    ``coverage = aceptadas / N`` y ``accuracy`` es el acierto entre las
    aceptadas (``None`` si no se acepta ninguna; ``risk = 1 - accuracy``).
    """
    p = _as_probs(probs)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    y = _as_targets(p, y_true)
    if thresholds is None:
        thresholds = (0.5, 0.6, 0.7, 0.8, 0.9)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    total = float(len(y))
    rows: list[dict[str, float | int | None]] = []
    for t in thresholds:
        threshold = float(t)
        mask = conf >= threshold
        n_selected = int(mask.sum())
        coverage = float(n_selected) / total if total else 0.0
        if n_selected:
            accuracy: float | None = float(correct[mask].mean())
            risk: float | None = 1.0 - accuracy
        else:
            accuracy = None
            risk = None
        rows.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "accuracy": accuracy,
                "risk": risk,
                "n": n_selected,
            }
        )
    return rows


def decide_platt(
    auc: float, *, min_auc: float = DEFAULT_MIN_AUC
) -> dict[str, object]:
    """Decide si un score es calibrable con Platt a partir de su AUC.

    Requiere ``auc >= min_auc`` (0.55). Un AUC menor (o ``nan``) implica que
    el score no ordena mejor que el azar: se rechaza la calibración y se
    documenta el motivo en vez de forzar un mapeo sin base.
    """
    usable = bool(np.isfinite(auc) and float(auc) >= float(min_auc))
    return {
        "usable": usable,
        "auc": float(auc),
        "minAuc": float(min_auc),
        "reason": None if usable else "score_auc_below_threshold",
    }


__all__ = [
    "DEFAULT_MIN_AUC",
    "DEFAULT_T_RANGE",
    "apply_platt",
    "apply_temperature",
    "auc_score",
    "decide_platt",
    "ece",
    "fit_temperature",
    "nll",
    "platt_fit",
    "risk_coverage",
]

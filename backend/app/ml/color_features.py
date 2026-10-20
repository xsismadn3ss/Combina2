"""Features de paleta para la red C (clasificador de armonía).

Dos representaciones conviven aquí:

- `palette_features` (54 dims, red `classify-v2`): representación de **conjunto**
  de colores, no de posición. Los colores se ordenan por tono `h` ascendente,
  de modo que el vector es invariante al orden de entrada (p.ej. al orden por
  peso con el que la red E devuelve `Color`s). Incluye un resumen de la
  estructura de tonos que hace explícita la armonía:

  - Bloque de colores (10 filas x 5 valores):
    `[sin(h), cos(h), chroma/128, l/100, weight]`
    (h, s, l en HSL: h∈[0,360), s/l∈[0,100]; chroma = hypot(a, b) de CIELAB).
    Máximo 10 colores, se descartan pesos < 0.01 y se renormalizan a suma 1.
    El padding repite la última fila con peso 0.
  - Resumen de tonos (4 dims) calculado con los colores cromáticos
    (`chroma > 8`): `[spread/180, n_clusters/4, max_gap/360, second_gap/360]`
    con `hues` ordenados y `gaps = diff(hues + [hues[0] + 360])`,
    `spread = 360 - max(gaps)` y `n_clusters = 1 + #{gaps > 25°}`.
    Con menos de 2 colores cromáticos: `[0.0, 0.25, 0.0, 1.0]`.

  Total: 10*5 + 4 = 54 dims.

- `palette_features_legacy` (40 dims, red `classify-v1`): representación
  histórica 10*Lab normalizado + 10 pesos, ordenada por peso. Se conserva
  **sin cambios de comportamiento** para permitir rollback a v1.

Módulo numpy puro: no importa TensorFlow ni sklearn.
"""

from __future__ import annotations

import math

import numpy as np

from app.ml.color_spaces import ColorConverter
from app.schemas.palette import Color

MAX_COLORS = 10
FEATURE_DIM_V2 = 54
FEATURE_DIM_V1 = 40
ROW_WIDTH = 5
MIN_WEIGHT = 0.01
CHROMATIC_MIN = 8.0
CLUSTER_GAP_DEG = 25.0
# Guard de baja información: croma máximo bajo *y* poca variedad de tono.
LOW_INFO_CHROMA_MAX = 12.0
LOW_INFO_HUE_SPREAD_MAX = 20.0


def _color_weight(color: Color, n_colors: int) -> float:
    return float(color.weight) if color.weight is not None else 1.0 / max(n_colors, 1)


def palette_colorfulness(colors: list[Color] | list[str]) -> float:
    """Croma máximo (``hypot(a, b)`` CIELAB) de los colores principales.

    Numpy/color_spaces puro. Un valor bajo (< ~12) indica una imagen
    prácticamente gris o monocroma: poca información para clasificar armonía.
    Acepta ``Color`` o HEX para facilitar tests y usos puntuales.
    """
    best = 0.0
    for color in colors or []:
        hexv = color.value if isinstance(color, Color) else str(color)
        try:
            _l, a, b = ColorConverter.hex_to_lab(hexv)
        except (ValueError, TypeError):
            continue
        best = max(best, float(math.hypot(float(a), float(b))))
    return best


def palette_hue_stats(colors: list[Color] | list[str]) -> tuple[float, int, int]:
    """``(spread_deg, n_clusters, n_colors)`` de los tonos de la paleta.

    Se calcula sobre **todos** los colores (no solo los cromáticos): un dúo de
    tonos apagados con hues distintos tiene variedad de tono aunque su croma
    sea bajo. ``spread = 360 - max_gap`` con los hues ordenados (círculo
    completo); ``n_clusters = 1 + #{gaps > 25°}``.
    """
    hues: list[float] = []
    for color in colors or []:
        hexv = color.value if isinstance(color, Color) else str(color)
        try:
            hue, _s, _l = ColorConverter.hex_to_hsl(hexv)
        except (ValueError, TypeError):
            continue
        hues.append(float(hue) % 360.0)
    if not hues:
        return (0.0, 0, 0)
    if len(hues) == 1:
        return (0.0, 1, 1)
    ordered = sorted(hues)
    gaps = np.diff(np.asarray(ordered + [ordered[0] + 360.0], dtype=np.float64))
    max_gap = float(np.max(gaps))
    spread = 360.0 - max_gap
    n_clusters = 1 + int(np.count_nonzero(gaps > CLUSTER_GAP_DEG))
    return (spread, n_clusters, len(ordered))


def palette_is_low_information(
    colors: list[Color] | list[str],
    *,
    max_chroma: float = LOW_INFO_CHROMA_MAX,
    max_hue_spread: float = LOW_INFO_HUE_SPREAD_MAX,
) -> bool:
    """True si la paleta no aporta señal de armonía (gris/monocroma).

    Dispara solo si el croma máximo es bajo (``< 12``) **y** además falta
    variedad de tono: spread ``< 20°`` o ``≤ 1`` cluster de tono. Así un dǧo
    apagado con hues diferentes (p. ej. ``#E6DCD2``/``#D2E1EB``) no pide
    reintento aunque su croma sea bajo.
    """
    if palette_colorfulness(colors) >= max_chroma:
        return False
    spread, n_clusters, n_colors = palette_hue_stats(colors)
    if n_colors <= 1:
        return True
    return spread < max_hue_spread or n_clusters <= 1


def _prepare_colors(
    colors: list[Color],
) -> list[tuple[float, float, float, float]]:
    """Filtra/normaliza y ordena por tono: devuelve (h, chroma, l, weight)."""
    if not colors:
        raise ValueError("La paleta no tiene colores")
    n_colors = len(colors)
    entries: list[tuple[float, float, float, float]] = []
    for color in colors[:MAX_COLORS]:
        h, _s, l = ColorConverter.hex_to_hsl(color.value)
        lab = ColorConverter.hex_to_lab(color.value)
        chroma = math.hypot(lab[1], lab[2])
        entries.append((float(h % 360.0), float(chroma), float(l), _color_weight(color, n_colors)))

    kept = [entry for entry in entries if entry[3] >= MIN_WEIGHT]
    if not kept:
        kept = [max(entries, key=lambda entry: entry[3])]
    total = float(sum(entry[3] for entry in kept))
    if not math.isfinite(total) or total <= 0.0:
        kept = [(entry[0], entry[1], entry[2], 1.0) for entry in kept]
        total = float(len(kept))
    normalized = [(h, c, l, w / total) for h, c, l, w in kept]
    normalized.sort(key=lambda entry: entry[0])
    return normalized


def _hue_summary(entries: list[tuple[float, float, float, float]]) -> list[float]:
    hues = sorted(entry[0] for entry in entries if entry[1] > CHROMATIC_MIN)
    if len(hues) < 2:
        return [0.0, 0.25, 0.0, 1.0]
    extended = hues + [hues[0] + 360.0]
    gaps = np.diff(np.asarray(extended, dtype=np.float64))
    max_gap = float(np.max(gaps))
    second_gap = float(np.sort(gaps)[-2])
    spread = 360.0 - max_gap
    n_clusters = 1 + int(np.count_nonzero(gaps > CLUSTER_GAP_DEG))
    return [
        spread / 180.0,
        n_clusters / 4.0,
        max_gap / 360.0,
        second_gap / 360.0,
    ]


def palette_features(colors: list[Color], input_dim: int = FEATURE_DIM_V2) -> np.ndarray:
    """Features de paleta de 54 dims (red C v2). Ver docstring del módulo.

    `input_dim` se acepta por compatibilidad con la detección de dimensión del
    modelo: 54 usa la representación nueva y 40 delega en la legacy.
    """
    if input_dim == FEATURE_DIM_V1:
        return palette_features_legacy(colors)
    if input_dim != FEATURE_DIM_V2:
        raise ValueError(f"Dimensión de features no soportada: {input_dim}")

    entries = _prepare_colors(colors)

    rows = np.zeros((MAX_COLORS, ROW_WIDTH), dtype=np.float32)
    for index, (h, chroma, l, weight) in enumerate(entries):
        rad = math.radians(h)
        rows[index] = (
            math.sin(rad),
            math.cos(rad),
            chroma / 128.0,
            l / 100.0,
            weight,
        )
    if len(entries) < MAX_COLORS:
        rows[len(entries) :] = rows[len(entries) - 1]
        rows[len(entries) :, ROW_WIDTH - 1] = 0.0

    summary = np.asarray(_hue_summary(entries), dtype=np.float32)
    return np.concatenate([rows.reshape(-1), summary]).astype(np.float32)


def palette_features_legacy(colors: list[Color]) -> np.ndarray:
    """Features históricas de 40 dims (red C v1): 10*Lab normalizado + 10 pesos.

    Réplica exacta de la lógica que vivía en
    `services.inference._classify_with_model`: orden de entrada preservado,
    padding repitiendo el último Lab con peso 0, pesos recortados a >= 0 y
    renormalizados. Lanza `ValueError` si no hay colores o el HEX es inválido
    (el llamador lo traduce a fallo de inferencia).
    """
    if not colors:
        raise ValueError("La paleta no tiene colores")
    labs: list[tuple[float, float, float]] = []
    weights: list[float] = []
    for color in colors[:MAX_COLORS]:
        lab_raw = ColorConverter.hex_to_lab(color.value)
        labs.append(ColorConverter.lab_normalize(lab_raw))
        weights.append(_color_weight(color, len(colors)))
    while len(labs) < MAX_COLORS:
        labs.append(labs[-1] if labs else (0.5, 0.0, 0.0))
        weights.append(0.0)
    w_arr = np.array(weights[:MAX_COLORS], dtype=np.float64)
    w_arr = np.clip(w_arr, 0.0, None)
    total = float(w_arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Pesos de paleta inválidos")
    w_arr = w_arr / total
    labs_arr = np.array(labs[:MAX_COLORS], dtype=np.float32).reshape(-1)
    return np.concatenate([labs_arr, w_arr.astype(np.float32)]).astype(np.float32)

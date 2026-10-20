"""Tests de `app.ml.color_features`: contrato de features 54 dims (issue #22).

Numpy puro: estos tests corren igual en `.venv` (3.14) y `.venv-tf` (3.12).
"""
import numpy as np
import pytest

from app.ml.color_features import (
    FEATURE_DIM_V1,
    FEATURE_DIM_V2,
    palette_colorfulness,
    palette_features,
    palette_features_legacy,
    palette_hue_stats,
    palette_is_low_information,
)
from app.schemas.palette import Color

ROLE = "predominant"


def _colors(pairs):
    return [Color(value=hexv, role=ROLE, weight=w) for hexv, w in pairs]


# Paleta de 5 tonos bien separados con pesos en orden descendente (como E).
PALETTE = _colors(
    [
        ("#FF0000", 0.30),  # rojo, h=0
        ("#00FF00", 0.25),  # verde, h=120
        ("#0000FF", 0.20),  # azul, h=240
        ("#FFFF00", 0.15),  # amarillo, h=60
        ("#FF00FF", 0.10),  # magenta, h=300
    ]
)


def test_palette_colorfulness_detects_grayscale_and_color():
    assert palette_colorfulness(_colors([("#808080", 0.5), ("#F0F0F0", 0.5)])) < 12.0
    assert palette_colorfulness(_colors([("#FF0000", 0.5), ("#00FF00", 0.5)])) >= 12.0
    # Acepta HEX directos y una lista vacía no revienta.
    assert palette_colorfulness(["#000000", "#FFFFFF"]) < 12.0
    assert palette_colorfulness([]) == 0.0


def test_low_information_requires_low_chroma_and_low_hue_variety():
    gray = _colors([("#808080", 0.5), ("#B0B0B0", 0.5)])
    dim_duo = _colors([("#E6DCD2", 0.5), ("#D2E1EB", 0.5)])
    colorful = _colors([("#FF0000", 0.5), ("#00FF00", 0.5)])

    assert palette_colorfulness(dim_duo) < 12.0  # ambos tonos son apagados...
    spread, clusters, _n = palette_hue_stats(dim_duo)
    assert spread >= 20.0 and clusters >= 2  # ...pero con hues distintos
    assert palette_is_low_information(dim_duo) is False
    assert palette_is_low_information(gray) is True
    assert palette_is_low_information(colorful) is False
    # Un solo color apagado no tiene variedad de tono: baja información.
    assert palette_is_low_information(_colors([("#E6DCD2", 1.0)])) is True


def test_palette_features_shape_and_dtype():
    feat = palette_features(PALETTE)
    assert feat.shape == (FEATURE_DIM_V2,)
    assert feat.dtype == np.float32
    assert np.all(np.isfinite(feat))


def test_palette_features_permutation_invariant():
    baseline = palette_features(PALETTE)
    for seed in range(5):
        rng = np.random.RandomState(seed)
        shuffled = [PALETTE[i] for i in rng.permutation(len(PALETTE))]
        assert np.allclose(palette_features(shuffled), baseline, atol=1e-6)


def test_palette_features_hue_rows_are_sorted():
    feat = palette_features(PALETTE)[:50].reshape(10, 5)
    hues = []
    for row in feat[: len(PALETTE)]:
        hues.append(np.degrees(np.arctan2(row[0], row[1])) % 360.0)
    assert hues == sorted(hues)


def test_palette_features_padding_repeats_last_row_with_zero_weight():
    feat = palette_features(PALETTE)[:50].reshape(10, 5)
    last = feat[len(PALETTE) - 1]
    for row in feat[len(PALETTE) :]:
        assert np.allclose(row[:4], last[:4])
        assert row[4] == 0.0


def test_palette_features_single_color_padding_and_summary():
    feat = palette_features(_colors([("#FF0000", 1.0)]))[:50].reshape(10, 5)
    assert np.allclose(feat[0, :4], feat[9, :4])
    assert np.allclose(feat[1:, 4], 0.0)
    assert feat[0, 4] == pytest.approx(1.0)
    summary = palette_features(_colors([("#FF0000", 1.0)]))[-4:]
    # un único color cromático => resumen acromático
    assert np.allclose(summary, [0.0, 0.25, 0.0, 1.0])


def test_palette_features_achromatic_palette_summary():
    feat = palette_features(_colors([("#808080", 0.6), ("#FFFFFF", 0.4)]))
    assert np.allclose(feat[-4:], [0.0, 0.25, 0.0, 1.0])
    rows = feat[:10].reshape(2, 5)
    # chroma ~0 en grises
    assert np.allclose(rows[:, 2], 0.0, atol=1e-3)


def test_palette_features_weight_filter_and_normalization():
    colors = _colors(
        [
            ("#FF0000", 0.6),
            ("#00FF00", 0.39),
            ("#0000FF", 0.009),  # < 0.01: se descarta
            ("#FFFF00", 0.001),  # < 0.01: se descarta
        ]
    )
    feat = palette_features(colors)
    rows = feat[:50].reshape(10, 5)
    weights = rows[:, 4]
    assert weights.sum() == pytest.approx(1.0, abs=1e-5)
    assert np.count_nonzero(weights[:2]) == 2
    # filas 2..9 son padding con peso 0
    assert np.allclose(weights[2:], 0.0)


def test_palette_features_summary_ranges_and_spread():
    feat = palette_features(PALETTE)
    rows = feat[:50].reshape(10, 5)
    assert np.all(rows[:, 0] >= -1.0) and np.all(rows[:, 0] <= 1.0)  # sin(h)
    assert np.all(rows[:, 1] >= -1.0) and np.all(rows[:, 1] <= 1.0)  # cos(h)
    assert np.all(rows[:, 3] >= 0.0) and np.all(rows[:, 3] <= 1.0)  # l/100
    assert np.all(rows[:, 4] >= 0.0) and np.all(rows[:, 4] <= 1.0)  # weight
    summary = feat[-4:]
    assert 0.0 <= summary[0] <= 2.0  # spread/180
    assert summary[1] >= 0.25  # n_clusters/4
    assert 0.0 <= summary[2] <= 1.0  # max_gap/360
    assert 0.0 <= summary[3] <= 1.0  # second_gap/360
    # 5 tonos separados ~60-120° => spread amplio
    assert summary[0] > 1.0


def test_palette_features_dim_40_delegates_to_legacy():
    assert np.array_equal(
        palette_features(PALETTE, input_dim=FEATURE_DIM_V1),
        palette_features_legacy(PALETTE),
    )
    with pytest.raises(ValueError):
        palette_features(PALETTE, input_dim=12)


def test_palette_features_empty_raises():
    with pytest.raises(ValueError):
        palette_features([])
    with pytest.raises(ValueError):
        palette_features_legacy([])


def test_legacy_features_shape_position_dependent_and_padding():
    feat = palette_features_legacy(PALETTE)
    assert feat.shape == (FEATURE_DIM_V1,)
    assert feat.dtype == np.float32
    assert feat[30:].sum() == pytest.approx(1.0, abs=1e-5)
    assert np.all(feat[35:] == 0.0)  # padding con peso 0
    reordered = list(reversed(PALETTE))
    assert not np.allclose(
        palette_features_legacy(reordered), feat, atol=1e-6
    ), "la representación legacy sí depende del orden (rollback v1)"


def test_legacy_features_invalid_hex_raises():
    with pytest.raises(ValueError):
        palette_features_legacy(_colors([("#ZZZZZZ", 1.0)]))

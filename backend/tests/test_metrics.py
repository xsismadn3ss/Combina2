"""Tests de `app.ml.metrics` (issue #22): numpy puro, ambos venvs.

Cubre CIEDE2000 contra Sharma et al. (2005), ΔE76, emparejamiento de paletas,
confusión/F1 y calibración (ECE/NLL/temperatura/AUC) con casos a mano.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.ml.metrics import (
    apply_temperature,
    ciede2000,
    ciede2000_reference_pairs,
    confusion_matrix,
    delta_e76,
    ece,
    macro_f1,
    nll,
    optimal_temperature,
    pair_palettes,
    palette_delta_e,
    per_class_report,
    roc_auc,
)


# --- CIEDE2000 (Sharma) -------------------------------------------------------


@pytest.mark.parametrize("lab1,lab2,expected", ciede2000_reference_pairs())
def test_ciede2000_sharma_reference_pairs(lab1, lab2, expected):
    value = float(ciede2000(np.array(lab1), np.array(lab2)))
    assert value == pytest.approx(expected, abs=1e-4)


def test_ciede2000_known_pair_from_issue():
    # Par documentado en el issue/baseline.
    value = float(
        ciede2000(np.array([50.0, 2.6772, -79.7751]), np.array([50.0, 0.0, -82.7485]))
    )
    assert value == pytest.approx(2.0425, abs=1e-4)


def test_ciede2000_vectorized_matches_scalar_and_symmetric():
    labs_a = np.array([[50.0, 2.6772, -79.7751], [60.0, -34.0, 36.0], [50.0, 0.0, 0.0]])
    labs_b = np.array([[50.0, 0.0, -82.7485], [60.0, -34.0, 39.0], [50.0, -1.0, 2.0]])
    batched = ciede2000(labs_a, labs_b)
    assert batched.shape == (3,)
    for i in range(3):
        assert batched[i] == pytest.approx(
            float(ciede2000(labs_a[i], labs_b[i])), abs=1e-12
        )
    assert np.allclose(ciede2000(labs_a, labs_b), ciede2000(labs_b, labs_a))
    assert np.allclose(ciede2000(labs_b, labs_b), 0.0, atol=1e-12)


# --- ΔE76 / emparejamiento de paletas ----------------------------------------


def test_delta_e76_known_values():
    assert float(delta_e76(np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0]))) == pytest.approx(100.0)
    assert float(delta_e76(np.array([50.0, 0.0, 0.0]), np.array([50.0, 3.0, 4.0]))) == pytest.approx(5.0)
    assert float(delta_e76(np.array([50.0, 10.0, -20.0]), np.array([50.0, 10.0, -20.0]))) == 0.0


def test_pair_palettes_is_order_invariant_and_greedy():
    a = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    b = a[::-1].copy()
    idx, dists, matrix = pair_palettes(a, b, metric="cie76")
    assert matrix.shape == (2, 2)
    assert np.allclose(dists, 0.0)
    assert sorted(idx.tolist()) == [0, 1]
    assert palette_delta_e(a, b, metric="cie76") == pytest.approx(0.0)


def test_pair_palettes_uses_requested_metric():
    a = np.array([[50.0, 2.5, 0.0]])
    b = np.array([[50.0, 0.0, -2.5]])
    _, dists76, _ = pair_palettes(a, b, metric="cie76")
    _, dists00, _ = pair_palettes(a, b, metric="ciede2000")
    assert dists76[0] == pytest.approx(3.5355, abs=1e-3)
    assert dists00[0] == pytest.approx(4.3065, abs=1e-4)


def test_metrics_reject_bad_shapes():
    with pytest.raises(ValueError):
        delta_e76(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        pair_palettes(np.zeros((2, 2)), np.zeros((2, 3)))


# --- Clasificación ------------------------------------------------------------


def test_confusion_matrix_hand_calculated():
    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 1, 1, 1, 2]
    cm = confusion_matrix(y_true, y_pred, n_classes=3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [0, 0, 1]]


def test_per_class_report_and_macro_f1_hand_calculated():
    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 1, 1, 1, 2]
    labels = ["a", "b", "c"]
    report = per_class_report(y_true, y_pred, labels=labels, n_classes=3)
    assert report["a"]["precision"] == pytest.approx(1.0)
    assert report["a"]["recall"] == pytest.approx(0.5)
    assert report["a"]["f1"] == pytest.approx(2.0 / 3.0)
    assert report["b"]["precision"] == pytest.approx(2.0 / 3.0)
    assert report["b"]["recall"] == pytest.approx(1.0)
    assert report["b"]["f1"] == pytest.approx(0.8)
    assert report["c"]["f1"] == pytest.approx(1.0)
    assert macro_f1(y_true, y_pred, labels=labels, n_classes=3) == pytest.approx(
        (2.0 / 3.0 + 0.8 + 1.0) / 3.0
    )


def test_confusion_matrix_rejects_out_of_range():
    with pytest.raises(ValueError):
        confusion_matrix([0, 3], [0, 0], n_classes=3)


# --- Calibración --------------------------------------------------------------


def test_nll_hand_calculated():
    probs = np.array([[0.5, 0.5], [0.9, 0.1]])
    assert nll(probs, [0, 1]) == pytest.approx((math.log(2.0) + math.log(10.0)) / 2.0)


def test_ece_hand_calculated():
    # conf 0.9 acierta (gap 0.1) y conf 0.8 falla (gap 0.8), 50% cada uno.
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert ece(probs, [0, 0], n_bins=10) == pytest.approx(0.45)


def test_ece_perfect_calibration_is_zero():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert ece(probs, [0, 1], n_bins=10) == pytest.approx(0.0)


def test_apply_temperature_identity_and_sharpening():
    probs = np.array([[0.6, 0.4]])
    assert np.allclose(apply_temperature(probs, 1.0), probs)
    sharp = apply_temperature(probs, 0.5)
    assert sharp.max() > probs.max()
    assert sharp.sum() == pytest.approx(1.0)


def test_optimal_temperature_improves_underconfident_probs():
    probs = np.array([[0.6, 0.4], [0.4, 0.6]])
    y = [0, 1]
    t_opt, nll_opt = optimal_temperature(probs, y, t_min=0.2, t_max=5.0, n_grid=481)
    assert t_opt < 1.0
    assert nll_opt < nll(probs, y)
    assert ece(apply_temperature(probs, t_opt), y) <= ece(probs, y) + 1e-9


def test_roc_auc_perfect_and_inverted():
    scores = [0.1, 0.2, 0.3, 0.4]
    assert roc_auc(scores, [0, 0, 1, 1]) == pytest.approx(1.0)
    assert roc_auc(scores, [1, 1, 0, 0]) == pytest.approx(0.0)
    assert roc_auc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    assert math.isnan(roc_auc([0.1, 0.2], [1, 1]))

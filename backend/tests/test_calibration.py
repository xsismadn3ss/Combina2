"""Tests de `app.ml.calibration` (B2 del issue #22), numpy puro, ambos venvs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.ml.calibration import (
    apply_platt,
    apply_temperature,
    auc_score,
    decide_platt,
    ece,
    fit_temperature,
    nll,
    platt_fit,
    risk_coverage,
)
from scripts import calibrate


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


def test_fit_temperature_recovers_synthetic_temperature():
    rng = np.random.RandomState(7)
    logits = rng.normal(0.0, 3.0, size=(2000, 4))
    true_t = 2.0
    # Datos generados con T=1; el "modelo" entrega probabilidades con logits
    # multiplicados por T (sobreconfiadas), y aplicar T=2 las recobra.
    p_data = np.exp(logits - logits.max(axis=1, keepdims=True))
    p_data = p_data / p_data.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(4, p=row) for row in p_data])
    logits_t = logits * true_t
    probs = np.exp(logits_t - logits_t.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    fitted = fit_temperature(probs, labels)
    assert fitted == pytest.approx(true_t, rel=0.15)
    assert nll(apply_temperature(probs, fitted), labels) <= nll(probs, labels) + 1e-9


def test_apply_temperature_preserves_argmax():
    rng = np.random.RandomState(11)
    probs = rng.dirichlet(np.ones(4), size=50)
    original = probs.argmax(axis=1)
    for temperature in (0.2, 0.35, 0.48, 1.0, 2.5, 5.0):
        scaled = apply_temperature(probs, temperature)
        assert np.array_equal(scaled.argmax(axis=1), original)
        assert np.allclose(scaled.sum(axis=1), 1.0)


@pytest.mark.parametrize("bad_t", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_apply_temperature_invalid_is_noop(bad_t):
    probs = np.array([[0.6, 0.4], [0.2, 0.8]])
    out = apply_temperature(probs, bad_t)
    assert np.allclose(out, probs)
    assert np.array_equal(out.argmax(axis=1), probs.argmax(axis=1))


def test_apply_temperature_one_is_noop_and_accepts_1d():
    probs = np.array([0.6, 0.4])
    out = apply_temperature(probs, 1.0)
    assert out.shape == (2,)
    assert np.allclose(out, probs)


def test_auc_score_perfect_inverted_and_ties():
    scores = [0.1, 0.2, 0.3, 0.4]
    assert auc_score(scores, [0, 0, 1, 1]) == pytest.approx(1.0)
    assert auc_score(scores, [1, 1, 0, 0]) == pytest.approx(0.0)
    assert auc_score([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    assert math.isnan(auc_score([0.1, 0.2], [1, 1]))


def test_platt_fit_and_apply():
    scores = np.linspace(-3.0, 3.0, 100)
    labels = (scores > 0).astype(int)
    a, b = platt_fit(scores, labels)
    assert a > 0.0
    probabilities = apply_platt(scores, a, b)
    assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)
    # Monótona no decreciente en el score (datos separables saturan a 0/1).
    assert np.all(np.diff(probabilities) >= 0.0)
    assert isinstance(apply_platt(0.5, a, b), float)
    assert 0.0 <= apply_platt(0.5, a, b) <= 1.0


def test_platt_fit_requires_both_classes():
    with pytest.raises(ValueError):
        platt_fit([0.1, 0.2], [1, 1])


def test_risk_coverage_hand_calculated():
    probs = np.array(
        [[0.9, 0.1], [0.6, 0.4], [0.55, 0.45], [0.2, 0.8]]
    )
    rows = risk_coverage(probs, [0, 0, 1, 1], [0.5, 0.8])
    assert rows[0]["coverage"] == pytest.approx(1.0)
    assert rows[0]["accuracy"] == pytest.approx(0.75)
    assert rows[0]["risk"] == pytest.approx(0.25)
    assert rows[0]["n"] == 4
    assert rows[1]["coverage"] == pytest.approx(0.5)
    assert rows[1]["accuracy"] == pytest.approx(1.0)
    assert rows[1]["n"] == 2

    empty = risk_coverage(probs, [0, 0, 1, 1], [0.99])
    assert empty[0]["coverage"] == pytest.approx(0.0)
    assert empty[0]["accuracy"] is None
    assert empty[0]["risk"] is None


def test_decide_platt_rejects_auc_below_threshold():
    rejected = decide_platt(0.3516, min_auc=0.55)
    assert rejected["usable"] is False
    assert rejected["reason"] == "score_auc_below_threshold"

    accepted = decide_platt(0.72, min_auc=0.55)
    assert accepted["usable"] is True
    assert accepted["reason"] is None

    assert decide_platt(float("nan"))["usable"] is False


def test_calibrate_recommends_threshold_preserving_accuracy_and_coverage():
    rows = [
        {"threshold": 0.5, "coverage": 1.0, "accuracy": 0.975},
        {"threshold": 0.6, "coverage": 0.9625, "accuracy": 0.987},
        {"threshold": 0.7, "coverage": 0.925, "accuracy": 1.0},
        {"threshold": 0.8, "coverage": 0.8875, "accuracy": 1.0},
    ]
    # 0.8 tiene cobertura < 90%: se descarta. 0.7 es el mayor con accuracy
    # máxima y cobertura >= 90% (la referencia del issue).
    assert calibrate._recommended_threshold(rows) == pytest.approx(0.7)

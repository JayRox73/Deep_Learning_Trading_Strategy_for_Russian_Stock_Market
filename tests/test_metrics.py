"""Tests for FQW classification metrics (no TF)."""

import numpy as np

from fqw.metrics.classification import weighted_f_score, weighted_f_score_dict


def test_weighted_f_score_perfect():
    y = np.array([0, 1, 2, 0, 1, 2])
    score = weighted_f_score(y, y)
    assert score == 1.0


def test_weighted_f_score_all_wrong():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])
    score = weighted_f_score(y_true, y_pred)
    assert 0.0 <= score < 1.0


def test_weighted_f_score_dict_keys():
    y = np.array([0, 1, 2])
    out = weighted_f_score_dict(y, y)
    assert set(out) >= {"WFS", "Accuracy", "E_1st", "E_2nd", "E_3rd"}
    assert out["Accuracy"] == 1.0

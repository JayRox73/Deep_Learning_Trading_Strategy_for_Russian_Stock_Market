"""Weighted F-score and shared classification metrics."""

import numpy as np


def weighted_f_score(
    y_true,
    y_pred,
    beta1: float = 0.5,
    beta2: float = 0.125,
    beta3: float = 0.125,
) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n_tu = np.sum((y_true == 1) & (y_pred == 1))
    n_td = np.sum((y_true == 2) & (y_pred == 2))
    n_tf = np.sum((y_true == 0) & (y_pred == 0))
    n_tp = n_tu + n_td + beta3 * n_tf
    e1 = np.sum((y_pred == 1) & (y_true == 2)) + np.sum((y_pred == 2) & (y_true == 1))
    e2 = np.sum((y_pred == 1) & (y_true == 0)) + np.sum((y_pred == 2) & (y_true == 0))
    e3 = np.sum((y_pred == 0) & (y_true == 1)) + np.sum((y_pred == 0) & (y_true == 2))
    denom = (1 + beta1**2 + beta2**2) * n_tp + e1 + beta1**2 * e2 + beta2**2 * e3
    return (1 + beta1**2 + beta2**2) * n_tp / denom if denom != 0 else 0.0


def weighted_f_score_dict(y_true, y_pred, beta1=0.5, beta2=0.125, beta3=0.125) -> dict:
    """CDT notebook variant returning a metrics dict."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n_tu = np.sum((y_true == 1) & (y_pred == 1))
    n_td = np.sum((y_true == 2) & (y_pred == 2))
    n_tf = np.sum((y_true == 0) & (y_pred == 0))
    e1 = np.sum((y_pred == 1) & (y_true == 2)) + np.sum((y_pred == 2) & (y_true == 1))
    e2 = np.sum((y_pred == 1) & (y_true == 0)) + np.sum((y_pred == 2) & (y_true == 0))
    e3 = np.sum((y_pred == 0) & (y_true == 1)) + np.sum((y_pred == 0) & (y_true == 2))
    n_tp = n_tu + n_td + beta3**2 * n_tf
    num = (1 + beta1**2 + beta2**2) * n_tp
    den = num + e1 + beta1**2 * e2 + beta2**2 * e3
    return {
        "WFS": num / den if den > 0 else 0,
        "Accuracy": float(np.mean(y_true == y_pred)),
        "E_1st": int(e1),
        "E_2nd": int(e2),
        "E_3rd": int(e3),
    }

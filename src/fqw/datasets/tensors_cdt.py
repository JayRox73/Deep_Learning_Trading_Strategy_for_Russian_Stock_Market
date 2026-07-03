"""Global z-score tensor construction for CDT models."""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from fqw.labeling import apply_volatility_labels


def create_tensors_cdt(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 24,
    alpha: float = 1.0,
):
    """Vectorized tensors with global z-score normalization."""
    df = df.copy().sort_values("DateTime").reset_index(drop=True)
    df = apply_volatility_labels(df, window_size=window_size, alpha=alpha)

    data_x = df[feature_cols].values.astype(np.float32)
    data_x[~np.isfinite(data_x)] = np.nan
    col_means = np.nanmean(data_x, axis=0)
    nan_mask = np.isnan(data_x)
    for i in range(data_x.shape[1]):
        data_x[nan_mask[:, i], i] = col_means[i]

    m_global = data_x.mean(axis=0, keepdims=True)
    s_global = data_x.std(axis=0, keepdims=True)
    s_global[s_global == 0] = 1.0
    data_x = (data_x - m_global) / s_global

    data_y = df["Target"].values
    dates = df["DateTime"].values
    prices = df["Close"].values

    n = len(df) - window_size * 2
    if n <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    x = sliding_window_view(data_x, window_shape=window_size, axis=0)[:n].transpose(0, 2, 1)
    i = window_size
    return x, data_y[i : i + n], dates[i : i + n], prices[i : i + n]

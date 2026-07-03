"""Tensor dataset construction for sequence models."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fqw.labeling import apply_volatility_labels


def create_tensors(
    df: pd.DataFrame,
    window_size: int = 24,
    alpha: float = 1.0,
    feature_cols: list[str] | None = None,
):
    """
    Build (X, y) tensors with per-window StandardScaler normalization.

    Returns X, y, dates, prices, feature_cols.
    """
    df = apply_volatility_labels(df, window_size=window_size, alpha=alpha)

    if feature_cols is None:
        feature_cols = ["Open", "High", "Low", "Close", "Volume"]

    data_x = df[feature_cols].values
    data_y = df["Target"].values
    dates = df["DateTime"].values
    prices = df["Close"].values

    x_list, y_list, date_list, price_list = [], [], [], []

    for i in range(window_size, len(df) - window_size):
        window = data_x[i - window_size : i]
        scaler = StandardScaler()
        norm_window = scaler.fit_transform(window)
        if np.isnan(norm_window).any():
            continue
        x_list.append(norm_window)
        y_list.append(data_y[i])
        date_list.append(dates[i])
        price_list.append(prices[i])

    return (
        np.array(x_list),
        np.array(y_list),
        np.array(date_list),
        np.array(price_list),
        feature_cols,
    )

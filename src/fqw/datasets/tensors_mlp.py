"""Tensor construction for MLP record-based walk-forward."""

import numpy as np

from fqw.labeling.cdt import create_labels_cdt


def create_tensors_mlp(
    df,
    feature_cols: list[str],
    window_size: int = 24,
    alpha: float = 0.55,
):
    """Build sequence tensors without per-window scaler (MLP pipeline)."""
    df = df.copy().sort_values("DateTime")
    labels = create_labels_cdt(df, window_size=window_size, alpha=alpha)

    data_x = df[feature_cols].values.astype(np.float32)
    dates = df["DateTime"].values
    prices = df["Close"].values

    x_list, y_list, date_list, price_list = [], [], [], []

    for i in range(window_size, len(df) - window_size):
        if labels[i] == -1 or np.isnan(labels[i]):
            continue
        window = data_x[i - window_size : i]
        if np.any(np.isnan(window)) or np.any(np.isinf(window)):
            continue
        x_list.append(window)
        y_list.append(labels[i])
        date_list.append(dates[i])
        price_list.append(prices[i])

    return np.array(x_list), np.array(y_list), np.array(date_list), np.array(price_list)

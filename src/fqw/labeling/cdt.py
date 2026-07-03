"""CDT-style labeling using interval-return volatility."""

import numpy as np
import pandas as pd


def create_labels_cdt(
    df: pd.DataFrame,
    window_size: int = 24,
    alpha: float = 0.55,
    close_col: str = "Close",
) -> np.ndarray:
    """Label using volatility of window-sized interval returns (MLP notebook)."""
    close = df[close_col]
    future_close = close.shift(-window_size)
    future_return = (future_close - close) / close

    interval_returns = close.pct_change(periods=window_size)
    volatility = interval_returns.rolling(window=10).std()

    threshold_up = alpha * volatility
    threshold_down = -alpha * volatility

    conditions = [future_return >= threshold_up, future_return <= threshold_down]
    labels = np.select(conditions, [1, 2], default=0).astype(int)
    labels[-window_size:] = -1
    return labels

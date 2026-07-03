"""Adaptive volatility labeling."""

import numpy as np
import pandas as pd


def apply_volatility_labels(
    df: pd.DataFrame,
    window_size: int = 24,
    alpha: float = 1.0,
    vol_rolling: int = 10,
    close_col: str = "Close",
) -> pd.DataFrame:
    """
    Label bars as Up (1), Down (2), or Flat (0) using adaptive volatility thresholds.

    future_change is compared against alpha * rolling volatility of 1-bar returns.
    """
    df = df.copy().sort_values("DateTime")
    returns = df[close_col].pct_change()
    volatility = returns.rolling(window=vol_rolling).std()
    future_change = (df[close_col].shift(-window_size) - df[close_col]) / df[close_col]

    conditions = [
        future_change >= alpha * volatility,
        future_change <= -alpha * volatility,
    ]
    df["Target"] = np.select(conditions, [1, 2], default=0)
    return df

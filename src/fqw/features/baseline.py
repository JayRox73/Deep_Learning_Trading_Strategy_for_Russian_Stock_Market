"""Feature engineering for classical ML baseline models."""

import numpy as np
import pandas as pd


def filter_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep MOEX main + evening session (10:00–23:50), weekdays only."""
    df = df.copy()
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df = df.set_index("DateTime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have 'DateTime' column or DatetimeIndex")

    df = df[df.index.dayofweek < 5]
    df = df.between_time("10:00", "23:50")
    return df.reset_index()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tabular features for Ridge/Lasso/LightGBM/CatBoost baseline."""
    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]

    df["return"] = close.pct_change()

    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"ret_lag_{lag}"] = df["return"].shift(lag)
        df[f"vol_lag_{lag}"] = volume.shift(lag)

    df["volatility"] = df["return"].rolling(20).std()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["MACD"] = ema12 - ema26

    if "DateTime" in df.columns:
        dt = pd.to_datetime(df["DateTime"])
        hour = dt.dt.hour + dt.dt.minute / 60.0
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        dow = dt.dt.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    candle_range = high - low
    df["body_ratio"] = (close - open_) / candle_range.replace(0, np.nan)
    df["upper_shadow"] = (
        high - pd.concat([close, open_], axis=1).max(axis=1)
    ) / candle_range.replace(0, np.nan)
    df["lower_shadow"] = (
        pd.concat([close, open_], axis=1).min(axis=1) - low
    ) / candle_range.replace(0, np.nan)

    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    df["volume_zscore"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

    return df.replace([np.inf, -np.inf], np.nan).dropna()

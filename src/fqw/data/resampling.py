"""Daily bar resampling."""

import pandas as pd


def prepare_daily_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday bars into daily OHLCV."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df = df.set_index("DateTime")

    df_daily = df.resample("D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    df_daily[["Open", "High", "Low", "Close"]] = df_daily[["Open", "High", "Low", "Close"]].ffill()
    df_daily["Volume"] = df_daily["Volume"].fillna(0)
    return df_daily.sort_index()

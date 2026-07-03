"""Market data cleaning and gap filling."""

import pandas as pd


def fill_time_gaps(df: pd.DataFrame, interval_name: str = "5min") -> pd.DataFrame:
    """Reindex to a regular time grid and forward-fill OHLC, zero-fill volume."""
    resample_map = {"5min": "5min", "15min": "15min", "1hour": "1h", "1day": "1D"}
    freq = resample_map.get(interval_name, "5min")

    df = df.copy()

    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df = df.set_index("DateTime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a 'DateTime' column or DatetimeIndex")

    df = df.sort_index()
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_range)
    df.index.name = "DateTime"

    price_cols = ["Open", "High", "Low", "Close"]
    df[price_cols] = df[price_cols].ffill().bfill()
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    return df.reset_index()


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace zero prices, forward/backward-fill OHLC, zero-fill volume."""
    df = df.copy()

    if "DateTime" in df.columns:
        df = df.set_index("DateTime").sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
        if df.index.name is None:
            df.index.name = "DateTime"
    else:
        raise ValueError("No DateTime column or index")

    cols_to_fix = ["Open", "High", "Low", "Close"]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].replace(0, pd.NA)

    df[cols_to_fix] = df[cols_to_fix].ffill().bfill()

    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    return df.reset_index()

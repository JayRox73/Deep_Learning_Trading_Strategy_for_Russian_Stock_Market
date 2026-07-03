"""Parquet loading and standard preprocessing pipeline."""

from pathlib import Path

import pandas as pd

from fqw.data.cleaning import clean_market_data, fill_time_gaps
from fqw.features.technical import add_technical_indicators
from fqw.features.wavelet import wavelet_denoise_5m_safe


def load_ticker_5min(
    ticker: str,
    data_dir: Path | str = "data",
    start: str = "2020-01-01",
    end: str = "2026-01-01",
) -> pd.DataFrame:
    """Load 5-minute parquet candles for a ticker."""
    path = Path(data_dir) / f"{ticker}_5min.parquet"
    df = pd.read_parquet(path)

    if df.index.name == "DateTime" or "DateTime" in getattr(df.index, "names", []):
        df = df.reset_index()
    elif "DateTime" not in df.columns:
        raise ValueError(f"No DateTime column in {path}")

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df[(df["DateTime"] >= start) & (df["DateTime"] < end)].copy()


def preprocess_ticker(
    df: pd.DataFrame,
    *,
    use_wavelet: bool = True,
    use_technical_indicators: bool = False,
    daily_window: int = 78,
) -> pd.DataFrame:
    """Standard cleaning pipeline: gaps → clean → optional wavelet → optional TIs."""
    df = fill_time_gaps(df, interval_name="5min")
    df = clean_market_data(df)

    ohlcv = ["Open", "High", "Low", "Close", "Volume"]
    if use_wavelet:
        for col in ohlcv:
            if col in df.columns:
                df[col] = wavelet_denoise_5m_safe(df[col], daily_window=daily_window)
        df = df.dropna(subset=[c for c in ohlcv if c in df.columns]).reset_index(drop=True)

    if use_technical_indicators:
        df = add_technical_indicators(df)
        df = df.dropna().reset_index(drop=True)

    return df

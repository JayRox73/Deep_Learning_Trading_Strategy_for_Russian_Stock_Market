"""Tinkoff Invest API data download with parquet caching."""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from fqw.data.cleaning import clean_market_data
from fqw.data.resampling import prepare_daily_data

INTERVALS = {
    "5min": ("CANDLE_INTERVAL_5_MIN", timedelta(days=7)),
    "15min": ("CANDLE_INTERVAL_15_MIN", timedelta(days=30)),
    "1hour": ("CANDLE_INTERVAL_HOUR", timedelta(days=90)),
    "1day": ("CANDLE_INTERVAL_DAY", timedelta(days=365)),
}


def _require_tinkoff():
    try:
        from tinkoff.invest import CandleInterval, Client
        from tinkoff.invest.utils import quotation_to_decimal
        return Client, CandleInterval, quotation_to_decimal
    except ImportError as exc:
        raise ImportError("Install optional dependency: pip install -e '.[data]'") from exc


def get_figi(ticker: str, token: str) -> str:
    Client, _, _ = _require_tinkoff()
    ticker = ticker.upper()
    with Client(token) as client:
        response = client.instruments.find_instrument(query=ticker)
        for instrument in response.instruments:
            if instrument.ticker == ticker and instrument.api_trade_available_flag:
                if instrument.class_code in ["TQBR", "TQCB", "SPBFUT", "CETS", "TQTF"]:
                    return instrument.figi
        for instrument in response.instruments:
            if instrument.ticker == ticker and instrument.api_trade_available_flag:
                return instrument.figi
    raise ValueError(f"No tradable instrument found for {ticker}")


def get_candles_data(
    ticker: str,
    token: str,
    data_dir: Path | str = "data",
    interval_name: str = "5min",
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    update_cache: bool = True,
) -> pd.DataFrame:
    """Download or update cached candle data from Tinkoff API."""
    Client, CandleInterval, _ = _require_tinkoff()
    ticker = ticker.upper()
    if interval_name not in INTERVALS:
        raise ValueError(f"Available intervals: {list(INTERVALS.keys())}")

    interval_attr, chunk_step = INTERVALS[interval_name]
    candle_interval = getattr(CandleInterval, interval_attr)
    data_dir = Path(data_dir)
    file_path = data_dir / f"{ticker}_{interval_name}.parquet"

    start_date = (start_date or datetime(2020, 1, 1)).replace(tzinfo=None)
    end_date = (end_date or datetime.now()).replace(tzinfo=None)
    df = pd.DataFrame()

    if file_path.exists():
        df = pd.read_parquet(file_path)
        df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)
        if not df.empty:
            cache_min, cache_max = df["DateTime"].min(), df["DateTime"].max()
            if (cache_min <= start_date and cache_max >= end_date - timedelta(minutes=1)) or not update_cache:
                mask = (df["DateTime"] >= start_date) & (df["DateTime"] <= end_date)
                return df.loc[mask].sort_values("DateTime", ascending=False)

    if not update_cache:
        raise FileNotFoundError(f"Cache for {ticker} is empty and update_cache=False")

    download_start = start_date if df.empty else max(start_date, df["DateTime"].max())
    figi = get_figi(ticker, token)
    new_rows = []
    current_from = download_start.replace(tzinfo=timezone.utc)
    target_to = end_date.replace(tzinfo=timezone.utc)

    with Client(token) as client:
        while current_from < target_to:
            current_to = min(current_from + chunk_step, target_to)
            for _ in range(3):
                try:
                    candles = client.market_data.get_candles(
                        figi=figi, from_=current_from, to=current_to, interval=candle_interval
                    ).candles
                    for c in candles:
                        new_rows.append(
                            {
                                "DateTime": c.time,
                                "Open": c.open.units + c.open.nano / 1e9,
                                "High": c.high.units + c.high.nano / 1e9,
                                "Low": c.low.units + c.low.nano / 1e9,
                                "Close": c.close.units + c.close.nano / 1e9,
                                "Volume": c.volume,
                            }
                        )
                    break
                except Exception as exc:
                    if "RESOURCE_EXHAUSTED" in str(exc):
                        time.sleep(60)
                    else:
                        time.sleep(2)
            current_from = current_to
            time.sleep(0.2)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        new_df["DateTime"] = pd.to_datetime(new_df["DateTime"]).dt.tz_localize(None)
        df = pd.concat([df, new_df]).drop_duplicates(subset=["DateTime"])

    if df.empty:
        raise RuntimeError(f"No data received for {ticker}")

    df = df.sort_values("DateTime", ascending=False)
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, compression="zstd", index=False)
    mask = (df["DateTime"] >= start_date) & (df["DateTime"] <= end_date)
    return df.loc[mask]


__all__ = ["get_figi", "get_candles_data", "INTERVALS", "clean_market_data", "prepare_daily_data"]

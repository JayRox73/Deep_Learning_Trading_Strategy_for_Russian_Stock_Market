"""Macroeconomic feature engineering for CDT experiments."""

from pathlib import Path

import numpy as np
import pandas as pd

KEY_RATE_HISTORY = pd.DataFrame(
    {
        "Date": pd.to_datetime(
            [
                "2020-02-10", "2020-04-27", "2020-06-22", "2020-07-27",
                "2021-03-22", "2021-04-26", "2021-06-15", "2021-07-26", "2021-09-13", "2021-10-25", "2021-12-20",
                "2022-02-28", "2022-04-11", "2022-05-27", "2022-06-14", "2022-07-25", "2022-09-19",
                "2023-07-24", "2023-08-15", "2023-09-18", "2023-10-30", "2023-12-18",
                "2024-07-26", "2024-10-25", "2024-12-20", "2025-02-14",
            ]
        ),
        "KeyRate": [
            0.060, 0.055, 0.045, 0.042, 0.045, 0.050, 0.055, 0.065, 0.068, 0.075, 0.085,
            0.200, 0.170, 0.140, 0.095, 0.080, 0.075, 0.085, 0.120, 0.130, 0.150, 0.160,
            0.180, 0.210, 0.210, 0.210,
        ],
    }
)

INFLATION_YEARLY = pd.DataFrame(
    {
        "Year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        "Inflation": [0.1291, 0.0538, 0.0252, 0.0427, 0.0305, 0.0491, 0.0839, 0.1192, 0.0742, 0.0951, 0.0599],
    }
)


def add_macro_features(df_ticker: pd.DataFrame, data_dir: Path | str = "data") -> pd.DataFrame:
    """Add MOEX index, volatility, key rate, and inflation without look-ahead."""
    df = df_ticker.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["Date"] = df["DateTime"].dt.normalize()
    data_dir = Path(data_dir)

    try:
        df_moex = pd.read_parquet(data_dir / "MOEX_1day.parquet")
        if "DateTime" in df_moex.columns:
            df_moex["DateTime"] = pd.to_datetime(df_moex["DateTime"])
            df_moex["Date"] = df_moex["DateTime"].dt.normalize()
        else:
            df_moex.index = pd.to_datetime(df_moex.index)
            df_moex["Date"] = df_moex.index.normalize()
        moex_daily = df_moex.groupby("Date").agg({"Close": "last"}).rename(columns={"Close": "MOEX_Close"})
        moex_daily["MOEX_Return"] = moex_daily["MOEX_Close"].pct_change()
        moex_daily["MOEX_Vol_20d"] = moex_daily["MOEX_Return"].rolling(20).std()
        moex_daily = moex_daily.drop(columns=["MOEX_Return"])
        moex_daily[["MOEX_Close", "MOEX_Vol_20d"]] = moex_daily[["MOEX_Close", "MOEX_Vol_20d"]].shift(1).ffill()
        df = df.merge(moex_daily, left_on="Date", right_index=True, how="left")
        df[["MOEX_Close", "MOEX_Vol_20d"]] = df[["MOEX_Close", "MOEX_Vol_20d"]].ffill()
        df["Relative_Strength"] = df["Close"] / df["MOEX_Close"]
    except Exception as exc:
        print(f"MOEX data unavailable: {exc}")
        df["MOEX_Close"] = 0
        df["MOEX_Vol_20d"] = 0
        df["Relative_Strength"] = 1

    df = df.sort_values("DateTime")
    key_rate_sorted = KEY_RATE_HISTORY.copy().sort_values("Date")
    key_rate_sorted["Date"] = key_rate_sorted["Date"] + pd.Timedelta(days=1)
    df = pd.merge_asof(df, key_rate_sorted, left_on="DateTime", right_on="Date", direction="backward")
    df = df.drop(columns=["Date_y"], errors="ignore")
    df["KeyRate"] = df["KeyRate"].ffill().bfill()

    df["Inflation_Year"] = df["DateTime"].dt.year - 1
    infl_map = dict(zip(INFLATION_YEARLY["Year"], INFLATION_YEARLY["Inflation"]))
    df["Inflation"] = df["Inflation_Year"].map(infl_map).ffill().bfill()

    drop_cols = ["Date_x", "Date", "Inflation_Year"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

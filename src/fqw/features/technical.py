"""Technical indicators for deep-learning models."""

import numpy as np
import pandas as pd


def add_technical_indicators(
    df: pd.DataFrame,
    sma_periods: list[int] | None = None,
    ema_periods: list[int] | None = None,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    cci_period: int = 20,
    adx_period: int = 14,
    roc_period: int = 12,
    cmf_period: int = 20,
) -> pd.DataFrame:
    if sma_periods is None:
        sma_periods = [20, 50, 200]
    if ema_periods is None:
        ema_periods = [12, 26]

    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    for period in sma_periods:
        df[f"SMA_{period}"] = close.rolling(window=period).mean()
    for period in ema_periods:
        df[f"EMA_{period}"] = close.ewm(span=period, adjust=False).mean()

    ema_f = close.ewm(span=macd_fast, adjust=False).mean()
    ema_s = close.ewm(span=macd_slow, adjust=False).mean()
    df["MACD"] = ema_f - ema_s
    df["MACD_Signal"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    sma_bb = close.rolling(window=bb_period).mean()
    std_bb = close.rolling(window=bb_period).std()
    df["BB_Upper"] = sma_bb + (2 * std_bb)
    df["BB_Lower"] = sma_bb - (2 * std_bb)

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=cci_period).mean()
    mad_tp = tp.rolling(window=cci_period).apply(lambda x: np.abs(x - x.mean()).mean())
    df["CCI"] = (tp - sma_tp) / (0.015 * mad_tp)

    df["Date"] = pd.to_datetime(df["DateTime"]).dt.date

    df["Typical_Price_Vol"] = volume * (high + low + close) / 3
    df["Cum_TP_Vol"] = df.groupby("Date")["Typical_Price_Vol"].cumsum()
    df["Cum_Vol"] = df.groupby("Date")["Volume"].cumsum()
    df["VWAP"] = df["Cum_TP_Vol"] / df["Cum_Vol"]
    df.drop(["Typical_Price_Vol", "Cum_TP_Vol", "Cum_Vol"], axis=1, inplace=True)

    obv_sign = np.sign(close.diff())
    df["OBV_Raw"] = (obv_sign * volume).fillna(0)
    df["OBV"] = df.groupby("Date")["OBV_Raw"].cumsum()
    df.drop("OBV_Raw", axis=1, inplace=True)

    plus_dm = high.diff().where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0)
    minus_dm = low.diff().abs().where((low.diff().abs() > high.diff()) & (low.diff().abs() > 0), 0)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(window=adx_period).mean()
    plus_di = 100 * (plus_dm.rolling(window=adx_period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=adx_period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["ADX"] = dx.rolling(window=adx_period).mean()

    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * volume
    df["MFL_Vol"] = mfv
    df["ADL"] = df.groupby("Date")["MFL_Vol"].cumsum()
    df["CMF"] = mfv.rolling(window=cmf_period).sum() / volume.rolling(window=cmf_period).sum()
    df.drop("MFL_Vol", axis=1, inplace=True)

    df["ROC"] = ((close - close.shift(roc_period)) / close.shift(roc_period)) * 100
    df.drop("Date", axis=1, inplace=True)
    return df

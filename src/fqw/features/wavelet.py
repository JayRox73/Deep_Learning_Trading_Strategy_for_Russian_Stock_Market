"""Causal wavelet denoising for intraday OHLCV series."""

import numpy as np
import pandas as pd
import pywt


def wavelet_denoise_5m_safe(
    series: pd.Series,
    wavelet: str = "sym4",
    level: int = 3,
    daily_window: int = 78,
) -> pd.Series:
    """Causal Symlet wavelet denoising without NaN/Inf artifacts."""
    n = len(series)
    values = series.values.astype(float)
    denoised = np.full(n, np.nan)
    denoised[:daily_window] = values[:daily_window]

    for i in range(daily_window, n):
        chunk = values[i - daily_window : i]
        coeffs = pywt.wavedec(chunk, wavelet, mode="constant", level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(daily_window))
        denoised_coeffs = [pywt.threshold(c, uthresh, mode="soft") for c in coeffs]
        rec = pywt.waverec(denoised_coeffs, wavelet, mode="constant")
        denoised[i] = rec[-1]

    result = pd.Series(denoised, index=series.index, name=series.name)
    return result.ffill().bfill().replace([np.inf, -np.inf], np.nan).ffill()

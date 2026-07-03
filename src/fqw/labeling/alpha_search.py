"""Grid search for optimal labeling alpha per ticker."""

from pathlib import Path

import numpy as np
import pandas as pd

from fqw.data.cleaning import clean_market_data, fill_time_gaps


def find_optimal_alpha(
    df: pd.DataFrame,
    window_size: int = 24,
    target_flat_pct: float = 0.60,
    alphas_range: np.ndarray | None = None,
    min_samples: int = 10_000,
) -> dict | None:
    """Find alpha that yields ~target_flat_pct flat labels."""
    alphas_range = alphas_range if alphas_range is not None else np.arange(0.1, 2.05, 0.05)

    close = df["Close"]
    future_return = (close.shift(-window_size) - close) / close
    returns_5m = close.pct_change()
    volatility_5m = returns_5m.rolling(window=window_size).std()
    volatility_2h_equiv = volatility_5m * np.sqrt(window_size)

    valid_idx = future_return.notna() & volatility_2h_equiv.notna() & (volatility_2h_equiv > 1e-7)
    future_return_valid = future_return[valid_idx]
    volatility_valid = volatility_2h_equiv[valid_idx]

    if len(future_return_valid) < min_samples:
        return None

    best_alpha, min_diff = None, 999.0
    total_valid = len(future_return_valid)
    best_flat_pct = best_up_pct = best_down_pct = 0.0

    for alpha in alphas_range:
        threshold_up = alpha * volatility_valid
        threshold_down = -alpha * volatility_valid
        is_up = future_return_valid >= threshold_up
        is_down = future_return_valid <= threshold_down
        is_flat = ~is_up & ~is_down
        flat_pct = is_flat.sum() / total_valid
        diff = abs(flat_pct - target_flat_pct)
        if diff < min_diff:
            min_diff = diff
            best_alpha = float(alpha)
            best_flat_pct = flat_pct
            best_up_pct = is_up.sum() / total_valid
            best_down_pct = is_down.sum() / total_valid

    return {
        "Best_Alpha": best_alpha,
        "Flat%": best_flat_pct * 100,
        "Up%": best_up_pct * 100,
        "Down%": best_down_pct * 100,
    }


def run_alpha_search(
    data_dir: Path | str = "data",
    window_size: int = 24,
    target_flat_pct: float = 0.60,
    start_date: str = "2020-01-01",
    alphas_range: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run alpha grid search across all *_5min.parquet files in data_dir."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*_5min.parquet"))
    results = []

    for i, file_path in enumerate(files):
        ticker = file_path.stem.replace("_5min", "")
        try:
            df = pd.read_parquet(file_path)
            df = df[df["DateTime"] >= start_date]
            df = fill_time_gaps(df)
            df = clean_market_data(df)
            row = find_optimal_alpha(
                df, window_size=window_size, target_flat_pct=target_flat_pct, alphas_range=alphas_range
            )
            if row:
                row["Ticker"] = ticker
                results.append(row)
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(files)} tickers...")
        except Exception as exc:
            print(f"Error {ticker}: {exc}")

    return pd.DataFrame(results)


def summarize_alpha_search(res_df: pd.DataFrame) -> None:
    """Print aggregate statistics for alpha search results."""
    if res_df.empty:
        print("No data for analysis.")
        return
    mean_alpha = res_df["Best_Alpha"].mean()
    median_alpha = res_df["Best_Alpha"].median()
    mode_alpha = res_df["Best_Alpha"].mode().iloc[0] if not res_df["Best_Alpha"].mode().empty else None
    print("=" * 70)
    print(f"Optimal alpha summary ({len(res_df)} tickers)")
    print("=" * 70)
    print(f"Mean:   {mean_alpha:.4f}")
    print(f"Median: {median_alpha:.4f}")
    print(f"Mode:   {mode_alpha}")
    print(res_df["Best_Alpha"].describe())
    print("\nLowest alpha (noisiest):")
    print(res_df.nsmallest(5, "Best_Alpha")[["Ticker", "Best_Alpha", "Flat%"]].to_string(index=False))
    print("\nHighest alpha (trendiest):")
    print(res_df.nlargest(5, "Best_Alpha")[["Ticker", "Best_Alpha", "Flat%"]].to_string(index=False))

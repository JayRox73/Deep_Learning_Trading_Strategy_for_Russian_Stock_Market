"""MLP threshold sweep and parquet export."""

from pathlib import Path

import pandas as pd

from fqw.backtest.mlp import backtest_mlp_probabilities
from fqw.config import MLPConfig
from fqw.data.loading import load_ticker_5min, preprocess_ticker
from fqw.features.technical import add_technical_indicators
from fqw.models.mlp import build_mlp_model
from fqw.training.moving_window import run_moving_window_backtest


def run_mlp_for_ticker(
    ticker: str,
    config: MLPConfig | None = None,
    *,
    use_technical_indicators: bool = False,
    output_dir: str | Path = "results/mlp",
) -> pd.DataFrame | None:
    """Train MLP walk-forward and save probability parquet."""
    cfg = config or MLPConfig()
    df = load_ticker_5min(ticker, data_dir=cfg.data_dir)
    df = preprocess_ticker(df, use_wavelet=False, use_technical_indicators=False)
    if use_technical_indicators:
        df = add_technical_indicators(df).dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ("DateTime", "Target", "index")]
    probs_df = run_moving_window_backtest(
        df,
        feature_cols,
        model_name=f"MLP {ticker}",
        config=cfg,
    )
    if probs_df is None or probs_df.empty:
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "w_TIs" if use_technical_indicators else "wo_TIs"
    path = out / f"{ticker}_MLP_{tag}_A{cfg.alpha}_probs.parquet"
    probs_df.to_parquet(path, index=False)
    print(f"Saved {path}")
    return probs_df


def sweep_mlp_thresholds(
    probs_df: pd.DataFrame,
    thresholds: list[float] | None = None,
    config: MLPConfig | None = None,
) -> pd.DataFrame:
    """Evaluate multiple confidence thresholds on saved MLP probabilities."""
    cfg = config or MLPConfig()
    thresholds = thresholds or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
    rows = []
    for thresh in thresholds:
        metrics = backtest_mlp_probabilities(probs_df, threshold=thresh, rf_annual=cfg.rf_annual)
        if metrics:
            rows.append(metrics)
    return pd.DataFrame(rows)

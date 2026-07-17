"""Batch experiment runners for CNN grid search."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fqw.backtest.metrics import analyze_trades
from fqw.backtest.risk import analyze_strategy_performance
from fqw.config import CNNConfig
from fqw.data.loading import load_ticker_5min, preprocess_ticker
from fqw.training.walk_forward import run_backtest_with_fine_tuning


def _trade_summary_row(
    ticker: str,
    alpha: float,
    confidence_threshold: float,
    trades_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    *,
    wavelet_used: bool,
    model_tag: str,
) -> dict:
    pnls = trades_df["Profit %"]
    if "Profit %" in trades_df.columns and pnls.max() <= 1:
        pnls_frac = pnls / 100
    else:
        pnls_frac = pnls / 100 if pnls.abs().max() > 1 else pnls

    n = len(pnls_frac)
    dt_series = pd.to_datetime(preds_df["DateTime"])
    total_years = max((dt_series.max() - dt_series.min()).days / 365.25, 0.1)
    trades_per_year = n / total_years
    sharpe = (pnls_frac.mean() / pnls_frac.std()) * np.sqrt(trades_per_year) if n > 1 else 0
    mde = 2.8 * pnls_frac.std() / np.sqrt(n) if n > 1 else 0
    stat_sig = abs(pnls_frac.mean()) > mde if n > 1 else False
    max_dd = (
        (trades_df["Cumulative PnL %"] - trades_df["Cumulative PnL %"].cummax()).min()
        if "Cumulative PnL %" in trades_df.columns
        else np.nan
    )

    return {
        "Ticker": ticker,
        "Alpha": alpha,
        "Confidence_Threshold": confidence_threshold,
        "Wavelet_Used": wavelet_used,
        "Model": model_tag,
        "Run_Date": datetime.now().strftime("%Y%m%d"),
        "Trades_Count": n,
        "Win_Rate_%": round((pnls_frac > 0).mean() * 100, 2) if n else 0,
        "Total_Return_%": round(pnls_frac.sum() * 100, 2),
        "Profit_Factor": round(
            pnls_frac[pnls_frac > 0].sum() / abs(pnls_frac[pnls_frac < 0].sum()), 2
        )
        if (pnls_frac < 0).any()
        else np.inf,
        "Sharpe_Ratio": round(sharpe, 2),
        "Max_Drawdown_%": round(max_dd, 2) if not np.isnan(max_dd) else np.nan,
        "MDE_%": round(mde * 100, 3),
        "Statistically_Significant": "Yes" if stat_sig else "No",
    }


def run_batch_backtest_to_csv(
    tickers: list[str],
    alpha: float,
    confidence_threshold: float,
    *,
    config: CNNConfig | None = None,
    filename: str = "backtest_results.csv",
    use_wavelet: bool = True,
    use_technical_indicators: bool = False,
    save_plots: bool = False,
) -> pd.DataFrame:
    """Run CNN backtests for multiple tickers and save aggregated CSV."""
    cfg = config or CNNConfig()
    summary = []

    for ticker in tickers:
        print(f"\n{'=' * 50}\nTicker: {ticker}\n{'=' * 50}")
        try:
            df = load_ticker_5min(ticker, data_dir=cfg.data_dir)
            df = preprocess_ticker(
                df,
                use_wavelet=use_wavelet,
                use_technical_indicators=use_technical_indicators,
            )
            if len(df) < 2000:
                print(f"⚠️ {ticker}: insufficient data ({len(df)})")
                continue

            preds_df, _ = run_backtest_with_fine_tuning(
                df=df,
                ticker_name=ticker,
                alpha=alpha,
                confidence_threshold=confidence_threshold,
                config=cfg,
                save_plots=save_plots,
            )
            if preds_df.empty:
                continue

            if use_technical_indicators:
                trades_df = analyze_strategy_performance(
                    preds_df,
                    commission=cfg.commission,
                    be_trigger=cfg.be_trigger,
                    min_hold_bars=cfg.window_size,
                )
            else:
                trades_df = analyze_trades(
                    preds_df, commission=cfg.commission, rf_annual=cfg.rf_annual
                )

            if trades_df is None or trades_df.empty:
                summary.append(
                    {
                        "Ticker": ticker,
                        "Alpha": alpha,
                        "Confidence_Threshold": confidence_threshold,
                        "Trades_Count": 0,
                    }
                )
                continue

            tag = "CNN_w_TIs" if use_technical_indicators else "CNN_wo_TIs"
            if use_wavelet:
                tag = f"CNN_Wavelet_{tag.split('_', 1)[1]}"

            summary.append(
                _trade_summary_row(
                    ticker,
                    alpha,
                    confidence_threshold,
                    trades_df,
                    preds_df,
                    wavelet_used=use_wavelet,
                    model_tag=tag,
                )
            )
        except Exception as exc:
            print(f"❌ {ticker}: {exc}")

    results = pd.DataFrame(summary)
    if not results.empty:
        out = Path(filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out, index=False)
        print(f"\nSaved: {out}")
    return results

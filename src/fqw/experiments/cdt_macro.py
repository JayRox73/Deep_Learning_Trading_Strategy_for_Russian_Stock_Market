"""CDT macro feature ablation runner."""

import pandas as pd

from fqw.config import CDTConfig, CDT_EXPERIMENTS
from fqw.data.cleaning import clean_market_data, fill_time_gaps
from fqw.data.loading import load_ticker_5min
from fqw.features.macro import add_macro_features
from fqw.training.cdt_walk_forward import run_cdt_backtest


def run_cdt_macro_ablation(config: CDTConfig | None = None) -> pd.DataFrame:
    """Run all CDT macro experiments across configured tickers."""
    cfg = config or CDTConfig()
    all_results = []

    for ticker in cfg.tickers:
        print(f"\n{'#' * 80}\n# TICKER: {ticker}\n{'#' * 80}")
        try:
            df = load_ticker_5min(ticker, data_dir=cfg.data_dir)
            df = fill_time_gaps(df)
            df = clean_market_data(df)
            df_macro = add_macro_features(df, data_dir=cfg.data_dir)

            for exp in CDT_EXPERIMENTS:
                model_name = f"{exp['name']} ({ticker})"
                result = run_cdt_backtest(df_macro.copy(), exp["features"], model_name, config=cfg)
                if result:
                    result["ticker"] = ticker
                    result["exp_name"] = exp["name"]
                    result["n_features"] = len(exp["features"])
                    all_results.append(result)
        except FileNotFoundError:
            print(f"⚠️ Missing data for {ticker}")
        except Exception as exc:
            print(f"❌ {ticker}: {exc}")

    return pd.DataFrame(all_results)

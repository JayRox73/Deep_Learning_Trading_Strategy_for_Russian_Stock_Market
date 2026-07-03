"""Classical ML baseline: Optuna tuning and z-score backtest."""

import time
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from fqw.config import BaselineConfig
from fqw.data.cleaning import clean_market_data
from fqw.features.baseline import add_features, filter_trading_hours

optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_model_from_trial(trial: optuna.Trial, model_name: str):
    if model_name == "Ridge":
        return Ridge(alpha=trial.suggest_float("alpha", 1e-3, 100.0, log=True)), True
    if model_name == "Lasso":
        return Lasso(alpha=trial.suggest_float("alpha", 1e-6, 1e-1, log=True), max_iter=5000), True
    if model_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 4000, step=500),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "num_leaves": trial.suggest_int("num_leaves", 4, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.8),
            "subsample": trial.suggest_float("subsample", 0.2, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "random_state": 42,
            "verbosity": -1,
        }
        return LGBMRegressor(**params), False
    if model_name == "CatBoost":
        params = {
            "iterations": trial.suggest_int("iterations", 500, 4000, step=500),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "depth": trial.suggest_int("depth", 2, 6),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.01, 10.0, log=True),
            "verbose": 0,
            "random_state": 42,
        }
        return CatBoostRegressor(**params), False
    raise ValueError(f"Unknown model: {model_name}")


def optimize_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_train_scaled: np.ndarray,
    config: BaselineConfig,
) -> tuple[dict, float]:
    tscv = TimeSeriesSplit(n_splits=config.n_cv_splits, gap=config.horizon)

    def objective(trial):
        _, uses_scaled = create_model_from_trial(trial, model_name)
        x = x_train_scaled if uses_scaled else x_train
        dir_acc_scores = []
        for train_idx, val_idx in tscv.split(x):
            x_tr, x_val = x[train_idx], x[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            if model_name == "Ridge":
                model_cv = Ridge(alpha=trial.params["alpha"])
            elif model_name == "Lasso":
                model_cv = Lasso(alpha=trial.params["alpha"], max_iter=5000)
            elif model_name == "LightGBM":
                p = dict(trial.params)
                p.update({"random_state": 42, "verbosity": -1})
                model_cv = LGBMRegressor(**p)
            else:
                p = dict(trial.params)
                p.update({"verbose": 0, "random_state": 42})
                model_cv = CatBoostRegressor(**p)
            model_cv.fit(x_tr, y_tr)
            y_pred = model_cv.predict(x_val)
            dir_acc_scores.append(np.mean(np.sign(y_val) == np.sign(y_pred)))
        return 1.0 - np.mean(dir_acc_scores)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.optuna_seed),
    )
    study.optimize(objective, n_trials=config.n_optuna_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def build_best_model(model_name: str, best_params: dict):
    if model_name == "Ridge":
        return Ridge(**best_params), True
    if model_name == "Lasso":
        return Lasso(**best_params, max_iter=5000), True
    if model_name == "LightGBM":
        return LGBMRegressor(**best_params, random_state=42, verbosity=-1), False
    if model_name == "CatBoost":
        return CatBoostRegressor(**best_params, verbose=0, random_state=42), False
    raise ValueError(f"Unknown model: {model_name}")


def run_regression_backtest(
    z_pred: np.ndarray,
    prices: np.ndarray,
    vol_test: np.ndarray,
    config: BaselineConfig,
) -> dict[str, Any]:
    trades = []
    equity = [1.0]
    position_open = False
    entry_price = 0.0
    entry_idx = 0
    is_short = False
    short_cost_per_bar = config.short_rate / config.bars_per_year

    for i in range(len(z_pred)):
        z_score = z_pred[i]
        price = prices[i]
        if position_open and (i >= entry_idx + config.horizon or i == len(z_pred) - 1):
            exit_price = price * (1 - config.slippage) if is_short else price * (1 + config.slippage)
            if is_short:
                ret = (entry_price - exit_price) / entry_price
                ret -= config.commission + short_cost_per_bar * (i - entry_idx)
            else:
                ret = (exit_price - entry_price) / entry_price - config.commission
            equity.append(equity[-1] * (1 + ret))
            trades.append(ret)
            position_open = False
        if not position_open:
            if z_score > config.z_threshold:
                position_open, is_short, entry_price, entry_idx = True, False, price * (1 + config.slippage) * (1 + config.commission), i
            elif z_score < -config.z_threshold:
                position_open, is_short, entry_price, entry_idx = True, True, price * (1 - config.slippage) * (1 - config.commission), i
        else:
            equity.append(equity[-1])

    equity = np.array(equity)
    returns_eq = np.diff(equity) / equity[:-1]
    if trades:
        pnls = np.array(trades)
        n = len(pnls)
        gross_profit = pnls[pnls > 0].sum()
        gross_loss = abs(pnls[pnls < 0].sum())
        return {
            "n_trades": n,
            "win_rate": (pnls > 0).mean() * 100,
            "total_return": (equity[-1] - 1) * 100,
            "avg_trade": pnls.mean() * 100,
            "profit_factor": gross_profit / gross_loss if gross_loss else float("inf"),
            "avg_rr": (pnls[pnls > 0].mean() / abs(pnls[pnls < 0].mean())) if (pnls < 0).any() else float("inf"),
            "sharpe": (np.mean(returns_eq) / np.std(returns_eq) * np.sqrt(config.bars_per_year)) if np.std(returns_eq) > 0 else 0,
            "mde": 2.8 * (pnls.std() / np.sqrt(n)) * 100 if n > 1 else 0,
            "stat_sig": abs(pnls.mean() * 100) > (2.8 * pnls.std() / np.sqrt(n) * 100) if n > 1 else False,
            "equity": equity,
        }
    return {
        "n_trades": 0, "win_rate": 0, "total_return": 0, "avg_trade": 0,
        "profit_factor": 0, "avg_rr": 0, "sharpe": 0, "mde": 0, "stat_sig": False, "equity": equity,
    }


def run_baseline_for_ticker(ticker: str, config: BaselineConfig | None = None, verbose: bool = True) -> list[dict]:
    cfg = config or BaselineConfig()
    fpath = cfg.data_dir / f"{ticker}_5min.parquet"
    if not fpath.exists():
        print(f"WARNING: {fpath} not found")
        return []

    df = pd.read_parquet(fpath)
    df = df[(df["DateTime"] >= cfg.start_date) & (df["DateTime"] < cfg.end_date)]
    df = filter_trading_hours(df)
    df = clean_market_data(df)
    if "DateTime" not in df.columns:
        df = df.reset_index()
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values("DateTime").reset_index(drop=True)
    df = add_features(df)
    df["rolling_vol"] = df["return"].rolling(20).std().shift(1)
    df["raw_target"] = df["Close"].pct_change(cfg.horizon).shift(-cfg.horizon)
    df["target"] = df["raw_target"] / df["rolling_vol"].replace(0, np.nan)
    df = df.dropna()

    start_date = df["DateTime"].min()
    train_end = start_date + pd.DateOffset(years=cfg.train_years)
    test_end = train_end + pd.DateOffset(years=cfg.test_years)
    train_df = df[df["DateTime"] < train_end].copy()
    test_df = df[(df["DateTime"] >= train_end) & (df["DateTime"] < test_end)].copy()
    if len(train_df) < 1000 or len(test_df) < 100:
        return []

    feature_cols = [c for c in df.columns if c not in ["DateTime", "target", "raw_target", "rolling_vol"]]
    x_train_raw = train_df[feature_cols].values
    y_train_arr = train_df["target"].values
    x_test_raw = test_df[feature_cols].values
    y_test_arr = test_df["target"].values
    prices_test = test_df["Close"].values
    vol_test = test_df["rolling_vol"].values

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_raw)
    x_test_scaled = scaler.transform(x_test_raw)
    sub_idx = np.arange(0, len(x_train_raw), cfg.optuna_subsample)

    results_rows = []
    model_predictions = {}
    for model_name in cfg.model_names:
        if verbose:
            print(f"\n{'=' * 20} {model_name}: Optuna {'=' * 20}")
        t0 = time.time()
        best_params, cv_score = optimize_model(
            model_name, x_train_raw[sub_idx], y_train_arr[sub_idx], x_train_scaled[sub_idx], cfg
        )
        final_model, uses_scaled = build_best_model(model_name, best_params)
        x_fit = x_train_scaled if uses_scaled else x_train_raw
        x_pred = x_test_scaled if uses_scaled else x_test_raw
        final_model.fit(x_fit, y_train_arr)
        y_pred_norm = final_model.predict(x_pred)
        model_predictions[model_name] = y_pred_norm
        y_pred_raw = y_pred_norm * vol_test
        mae = mean_absolute_error(y_test_arr * vol_test, y_pred_raw)
        rmse = float(np.sqrt(mean_squared_error(y_test_arr * vol_test, y_pred_raw)))
        metrics = run_regression_backtest(y_pred_norm, prices_test, vol_test, cfg)
        results_rows.append({
            "Ticker": ticker, "Model": model_name, "MAE": mae, "RMSE": rmse,
            "Trades": metrics["n_trades"], "Win Rate %": round(metrics["win_rate"], 1),
            "Total Return %": round(metrics["total_return"], 2),
            "Profit Factor": round(metrics["profit_factor"], 2),
            "Sharpe": round(metrics["sharpe"], 2),
            "Stat Sig": metrics["stat_sig"],
            "CV Score": round(cv_score, 4),
            "Tune Time": round(time.time() - t0, 1),
            "Best Params": str(best_params),
        })

    ridge_z = model_predictions.get("Ridge")
    lgbm_z = model_predictions.get("LightGBM")
    if ridge_z is not None and lgbm_z is not None:
        ensemble_z = (ridge_z + lgbm_z) / 2.0
        metrics_ens = run_regression_backtest(ensemble_z, prices_test, vol_test, cfg)
        results_rows.append({
            "Ticker": ticker, "Model": "Ensemble_Ridge_LGBM", "MAE": None, "RMSE": None,
            "Trades": metrics_ens["n_trades"], "Win Rate %": round(metrics_ens["win_rate"], 1),
            "Total Return %": round(metrics_ens["total_return"], 2),
            "Profit Factor": round(metrics_ens["profit_factor"], 2),
            "Sharpe": round(metrics_ens["sharpe"], 2),
            "Stat Sig": metrics_ens["stat_sig"],
        })
    return results_rows

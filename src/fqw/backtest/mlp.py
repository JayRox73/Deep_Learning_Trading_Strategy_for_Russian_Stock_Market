"""MLP probability-threshold backtest."""

import numpy as np
import pandas as pd

from fqw.metrics.classification import weighted_f_score


def backtest_mlp_probabilities(
    df: pd.DataFrame,
    threshold: float = 0.5,
    hold_on_flat: bool = True,
    commission: float = 0.0003,
    rf_annual: float = 0.09,
    eval_start: str = "2024-01-01",
    eval_end: str = "2026-01-01",
    return_equity: bool = False,
):
    """Vectorized backtest on MLP probability outputs."""
    df = df.copy().sort_values("DateTime")

    df["Predicted"] = 0
    df.loc[(df["Prob_Up"] > threshold) & (df["Prob_Up"] >= df["Prob_Down"]), "Predicted"] = 1
    df.loc[(df["Prob_Down"] > threshold) & (df["Prob_Down"] > df["Prob_Up"]), "Predicted"] = 2

    df_eval_wfs = df[(df["DateTime"] >= eval_start) & (df["DateTime"] < eval_end)]
    wfs = weighted_f_score(df_eval_wfs["Actual"], df_eval_wfs["Predicted"])

    df["Target_Position"] = 0
    df.loc[df["Predicted"] == 1, "Target_Position"] = 1
    df.loc[df["Predicted"] == 2, "Target_Position"] = -1
    if hold_on_flat:
        df["Target_Position"] = df["Target_Position"].replace(0, np.nan).ffill().fillna(0)

    df["Position"] = df["Target_Position"].shift(1).fillna(0)
    df["Market_Return"] = df["Price"].pct_change().fillna(0)
    df["Strategy_Return_Gross"] = df["Position"] * df["Market_Return"]
    df["Trade_Cost"] = df["Position"].diff().abs().fillna(0) * commission
    df["Strategy_Return_Net"] = df["Strategy_Return_Gross"] - df["Trade_Cost"]
    df["Equity"] = (1 + df["Strategy_Return_Net"]).cumprod()

    df_eval = df[(df["DateTime"] >= eval_start) & (df["DateTime"] < eval_end)].copy()
    if df_eval.empty:
        return None if not return_equity else (None, None)

    start_equity = df_eval["Equity"].iloc[0]
    df_eval["Equity_Norm"] = df_eval["Equity"] / start_equity

    df_eval["Date"] = pd.to_datetime(df_eval["DateTime"]).dt.date
    daily_returns = df_eval.groupby("Date")["Strategy_Return_Net"].apply(lambda x: (1 + x).prod() - 1)
    rf_daily = rf_annual / 252
    excess_daily = daily_returns - rf_daily
    sharpe = (excess_daily.mean() / excess_daily.std()) * np.sqrt(252) if excess_daily.std() > 0 else 0
    max_dd = (df_eval["Equity_Norm"] / df_eval["Equity_Norm"].cummax() - 1).min() * 100
    total_return = (df_eval["Equity_Norm"].iloc[-1] / df_eval["Equity_Norm"].iloc[0] - 1) * 100
    total_years = (pd.to_datetime(eval_end) - pd.to_datetime(eval_start)).days / 365.25
    position_changes = df_eval["Position"].diff().abs()
    n_trades = int((position_changes[position_changes > 0].count() + 1) / 2)

    metrics = {
        "threshold": threshold,
        "logic": "Flat=Hold" if hold_on_flat else "Flat=Cash",
        "weighted_f_score": wfs,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "n_trades": n_trades,
        "years": total_years,
        "final_equity_norm": df_eval["Equity_Norm"].iloc[-1],
    }

    if return_equity:
        equity_df = df_eval[["DateTime", "Equity_Norm"]].copy()
        equity_df["threshold"] = threshold
        return metrics, equity_df
    return metrics

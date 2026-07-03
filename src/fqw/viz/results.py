"""Visualization helpers for thesis result analysis."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from fqw.metrics.classification import weighted_f_score

DEFAULT_PROJECT_DIRS = [
    ("results/mlp", "MLP", "MLP"),
    ("results/cdt", "CDT", "CDT 1-D CNN"),
    ("results", "CNN", "CNN"),
]


def calculate_comprehensive_metrics(
    df: pd.DataFrame,
    threshold: float = 0.5,
    hold_period: int = 24,
    commission: float = 0.0003,
    rf_annual: float = 0.09,
    eval_start: str = "2024-01-01",
    eval_end: str = "2026-01-01",
):
    df = df.copy().sort_values("DateTime")
    if pd.api.types.is_integer_dtype(df["DateTime"]):
        max_val = df["DateTime"].max()
        unit = "ns" if max_val > 1e17 else "us" if max_val > 1e14 else "ms" if max_val > 1e11 else "s"
        df["DateTime"] = pd.to_datetime(df["DateTime"], unit=unit)
    else:
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")

    df["Predicted"] = 0
    df.loc[(df["Prob_Up"] > threshold) & (df["Prob_Up"] >= df["Prob_Down"]), "Predicted"] = 1
    df.loc[(df["Prob_Down"] > threshold) & (df["Prob_Down"] > df["Prob_Up"]), "Predicted"] = 2

    positions, current_pos, hold_counter = [], 0, 0
    for pred in df["Predicted"]:
        if hold_counter > 0:
            hold_counter -= 1
        else:
            if pred == 1:
                current_pos, hold_counter = 1, hold_period - 1
            elif pred == 2:
                current_pos, hold_counter = 0, hold_period - 1
            else:
                current_pos, hold_counter = 0, 0
        positions.append(current_pos)

    df["Position"] = pd.Series(positions).shift(1).fillna(0)
    df["Market_Return"] = df["Price"].pct_change().fillna(0)
    df["Strategy_Return_Gross"] = df["Position"] * df["Market_Return"]
    df["Trade_Cost"] = df["Position"].diff().abs().fillna(0) * commission
    df["Strategy_Return_Net"] = df["Strategy_Return_Gross"] - df["Trade_Cost"]
    df["Equity"] = (1 + df["Strategy_Return_Net"]).cumprod()

    df_eval = df[(df["DateTime"] >= eval_start) & (df["DateTime"] < eval_end)].copy()
    if df_eval.empty or len(df_eval) < 10:
        return None, {}

    start_equity = df_eval["Equity"].iloc[0]
    df_eval["Equity_Norm"] = df_eval["Equity"] / start_equity
    wfs = weighted_f_score(df_eval["Actual"], df_eval["Predicted"])
    total_return = (df_eval["Equity_Norm"].iloc[-1] / df_eval["Equity_Norm"].iloc[0] - 1) * 100
    max_dd = ((df_eval["Equity_Norm"] / df_eval["Equity_Norm"].cummax() - 1).min()) * 100

    df_eval["Date"] = df_eval["DateTime"].dt.date
    daily_returns = df_eval.groupby("Date")["Strategy_Return_Net"].apply(lambda x: (1 + x).prod() - 1)
    rf_daily = rf_annual / 252
    excess_daily = daily_returns - rf_daily
    sharpe = (excess_daily.mean() / excess_daily.std()) * np.sqrt(252) if excess_daily.std() > 0 else 0
    gross_profit = daily_returns[daily_returns > 0].sum()
    gross_loss = abs(daily_returns[daily_returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    win_rate = (daily_returns > 0).mean() * 100

    bh_return = (df_eval["Price"].iloc[-1] / df_eval["Price"].iloc[0] - 1) * 100
    bh_max_dd = ((df_eval["Price"] / df_eval["Price"].cummax()) - 1).min() * 100

    bh_daily = df_eval.groupby("Date")["Market_Return"].apply(lambda x: (1 + x).prod() - 1)
    bh_excess = bh_daily - rf_daily
    bh_sharpe = (bh_excess.mean() / bh_excess.std()) * np.sqrt(252) if bh_excess.std() > 0 else 0
    dd_reduction = bh_max_dd - max_dd
    t_stat, p_value = stats.ttest_1samp(excess_daily, 0) if len(excess_daily) > 1 else (0, 1)
    is_significant = "Yes" if p_value < 0.05 and total_return > bh_return else "No"

    metrics = {
        "WFS": round(wfs, 4),
        "Win Rate %": round(win_rate, 2),
        "Profit Factor": round(profit_factor, 2),
        "Return, %": round(total_return, 2),
        "Max DD, %": round(max_dd, 2),
        "Sharpe": round(sharpe, 2),
        "B&H Return, %": round(bh_return, 2),
        "B&H Max DD, %": round(bh_max_dd, 2),
        "B&H Sharpe": round(bh_sharpe, 2),
        "Alpha (Return), %": round(total_return - bh_return, 2),
        "DD Reduction, %": round(dd_reduction, 2),
        "Signif.": is_significant,
        "Trades": int(df_eval["Position"].diff().abs().sum() / 2),
    }
    return df_eval, metrics


def parse_model_variant(filename: str, short_id: str) -> str:
    core_name = re.sub(r"^[A-Z]+_", "", filename)
    core_name = re.sub(r"_A\d+[\.,]\d+_probs\.parquet$", "", core_name)
    variant = re.sub(rf"^{short_id}_?", "", core_name, flags=re.IGNORECASE)
    return variant.replace("_", " ").strip()


def plot_equity_curves(data_store: dict, plots_dir: Path | str, colors: dict | None = None):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tickers = list(data_store.keys())
    if not tickers:
        return
    fig, axes = plt.subplots(len(tickers), 1, figsize=(12, 4 * len(tickers)), sharex=True)
    if len(tickers) == 1:
        axes = [axes]
    colors = colors or {}
    for i, ticker in enumerate(tickers):
        ax = axes[i]
        models_data = data_store[ticker]
        first_model_df = list(models_data.values())[0][0]
        bh_norm = first_model_df["Price"] / first_model_df["Price"].iloc[0]
        ax.plot(first_model_df["DateTime"], bh_norm, label="Buy & Hold", linestyle="--", color="#888888")
        for model_key, (df_eval, _, alpha_val, model_name) in models_data.items():
            ax.plot(df_eval["DateTime"], df_eval["Equity_Norm"], label=f"{model_name} (α={alpha_val})", color=colors.get(model_key, None))
        ax.set_title(ticker)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.axhline(1.0, color="black", alpha=0.2)
    plt.tight_layout()
    plt.savefig(plots_dir / "equity_curves.png", bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(metrics_df: pd.DataFrame, plots_dir: Path | str, colors: dict | None = None):
    if metrics_df.empty:
        return
    plots_dir = Path(plots_dir)
    best_df = metrics_df.loc[metrics_df.groupby(["Ticker", "Model", "Alpha"])["Sharpe"].idxmax()].copy()
    best_df["Model_Alpha"] = best_df["Model"] + " [α=" + best_df["Alpha"].astype(str) + "]"
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    palette = colors or "colorblind"
    sns.barplot(data=best_df, x="Ticker", y="Return, %", hue="Model_Alpha", ax=axes[0, 0], palette=palette)
    sns.barplot(data=best_df, x="Ticker", y="Sharpe", hue="Model_Alpha", ax=axes[0, 1], palette=palette)
    sns.barplot(data=best_df, x="Ticker", y="Profit Factor", hue="Model_Alpha", ax=axes[1, 0], palette=palette)
    sns.barplot(data=best_df, x="Ticker", y="WFS", hue="Model_Alpha", ax=axes[1, 1], palette=palette)
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_comparison.png", bbox_inches="tight")
    plt.close()


def plot_risk_return_scatter(metrics_df: pd.DataFrame, plots_dir: Path | str, colors: dict | None = None):
    if metrics_df.empty:
        return
    plots_dir = Path(plots_dir)
    best_df = metrics_df.loc[metrics_df.groupby(["Ticker", "Model", "Alpha"])["Sharpe"].idxmax()].copy()
    best_df["Model_Alpha"] = best_df["Model"] + " [α=" + best_df["Alpha"].astype(str) + "]"
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=best_df, x="Sharpe", y="Return, %", hue="Model_Alpha", style="Ticker",
        s=150, palette=colors or "colorblind", edgecolor="black", linewidth=0.8,
    )
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.tight_layout()
    plt.savefig(plots_dir / "risk_return_scatter.png", bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(metrics_df: pd.DataFrame, plots_dir: Path | str):
    if metrics_df.empty:
        return
    plots_dir = Path(plots_dir)
    numeric_cols = ["WFS", "Win Rate %", "Return, %", "Sharpe", "Profit Factor", "Max DD, %", "Trades"]
    corr_matrix = metrics_df[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f",
        square=True, linewidths=0.5, mask=mask, cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix of Strategy Metrics")
    plt.tight_layout()
    plt.savefig(plots_dir / "correlation_heatmap.png", bbox_inches="tight")
    plt.close()


def scan_probability_results(
    project_dirs: list[tuple[str, str, str]] | None = None,
    threshold_range: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """Scan result directories for *_probs.parquet and compute metrics per threshold."""
    project_dirs = project_dirs or DEFAULT_PROJECT_DIRS
    threshold_range = threshold_range if threshold_range is not None else np.arange(0.10, 0.90, 0.05)

    all_metrics_rows = []
    equity_curves_store: dict = {}
    colors_dynamic: dict = {}
    color_counter = 0
    unique_palette = [
        "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
        "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324",
    ]

    for proj_dir, short_id, display_base in project_dirs:
        results_dir = Path(proj_dir)
        if not results_dir.exists():
            print(f"[SKIP] {results_dir}")
            continue
        files = [f for f in results_dir.iterdir() if f.name.endswith("_probs.parquet")]
        if not files:
            continue

        for file_path in sorted(files):
            file = file_path.name
            alpha_match = re.search(r"_A(\d+\.\d+)_probs\.parquet", file)
            alpha_val = alpha_match.group(1) if alpha_match else "X.XX"
            core_name = re.sub(r"_A\d+[\.,]\d+_probs\.parquet$", "", file)
            core_name = re.sub(r"_Regular", "", core_name, flags=re.IGNORECASE).strip("_")
            match = re.search(rf"(_|^){short_id}(_|$)", core_name, re.IGNORECASE)
            if match:
                ticker = core_name[: match.start()].strip("_")
                variant = core_name[match.end() :].strip("_").replace("_", " ")
            else:
                parts = core_name.split("_", 1)
                ticker, variant = parts[0], (parts[1].replace("_", " ") if len(parts) > 1 else "")
            model_name = f"{display_base} ({variant})" if variant else display_base
            model_alpha_key = f"{model_name} [α={alpha_val}]"
            if model_alpha_key not in colors_dynamic:
                colors_dynamic[model_alpha_key] = unique_palette[color_counter % len(unique_palette)]
                color_counter += 1
            colors_dynamic["Buy_Hold"] = "#888888"

            df_raw = pd.read_parquet(file_path)
            best_sharpe, best_thresh, best_eval_df, best_metrics = -999, 0, None, None
            for thresh in threshold_range:
                eval_df, metrics = calculate_comprehensive_metrics(df_raw, threshold=thresh)
                if metrics and metrics["Trades"] > 0:
                    row = {"Ticker": ticker, "Model": model_name, "Alpha": alpha_val, "Threshold": thresh}
                    row.update(metrics)
                    all_metrics_rows.append(row)
                    if metrics["Sharpe"] > best_sharpe:
                        best_sharpe, best_thresh = metrics["Sharpe"], thresh
                        best_eval_df, best_metrics = eval_df, metrics
            if best_eval_df is not None:
                equity_curves_store.setdefault(ticker, {})[model_alpha_key] = (
                    best_eval_df, best_thresh, alpha_val, model_name,
                )
                print(f"  {ticker} | {model_name} [α={alpha_val}]: Sharpe={best_metrics['Sharpe']:.2f}")

    return pd.DataFrame(all_metrics_rows), equity_curves_store, colors_dynamic

"""Trade-level performance metrics."""

import numpy as np
import pandas as pd

from fqw.metrics.classification import weighted_f_score

__all__ = ["analyze_trades", "weighted_f_score"]


def analyze_trades(df: pd.DataFrame, commission: float = 0.0003, rf_annual: float = 0.15):
    """Analyze trades with Profit Factor, Sharpe, MDE, and statistical significance."""
    trades = []
    in_position = False
    entry_price = 0.0
    entry_date = None

    for i in range(len(df)):
        signal = df["Predicted"].iloc[i]
        price = df["Price"].iloc[i]
        date = df["DateTime"].iloc[i]

        if signal == 1 and not in_position:
            in_position = True
            entry_price = price
            entry_date = date
        elif (signal == 2 or i == len(df) - 1) and in_position:
            duration_delta = date - entry_date
            pnl = (price - entry_price) / entry_price
            pnl_net = pnl - (commission * 2)
            trades.append(
                {
                    "Entry Date": entry_date.strftime("%Y-%m-%d"),
                    "Exit Date": date.strftime("%Y-%m-%d"),
                    "Entry Price": round(entry_price, 2),
                    "Exit Price": round(price, 2),
                    "Profit %": round(pnl_net * 100, 2),
                    "Duration Days": duration_delta.days,
                    "Duration Months": round(duration_delta.days / 30.4375, 2),
                    "Duration Years": round(duration_delta.days / 365.25, 2),
                }
            )
            in_position = False

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("No trades found.")
        return None

    pnls = trades_df["Profit %"] / 100
    n = len(pnls)
    win_rate = (pnls > 0).mean() * 100
    total_return = pnls.sum() * 100
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

    avg_win = pnls[pnls > 0].mean() if (pnls > 0).any() else 0
    avg_loss = abs(pnls[pnls < 0].mean()) if (pnls < 0).any() else 0
    _avg_rr = avg_win / avg_loss if avg_loss != 0 else np.inf

    avg_duration_years = trades_df["Duration Years"].mean()
    rf_per_trade = rf_annual / (1 / avg_duration_years) if avg_duration_years > 0 else 0
    std_pnl = pnls.std()
    sharpe = (pnls.mean() - rf_per_trade) / std_pnl * np.sqrt(n) if std_pnl > 0 else 0
    mde = 2.8 * std_pnl / np.sqrt(n) if n > 1 else 0
    stat_significant = abs(pnls.mean()) > mde

    print("=" * 70)
    print("STRATEGY SUMMARY")
    print("=" * 70)
    print(f"Trades:            {n}")
    print(f"Win Rate:          {win_rate:.1f}%")
    print(f"Total Return:      {total_return:.1f}%")
    print(f"Profit Factor:     {profit_factor:.2f}")
    print(f"Sharpe:            {sharpe:.2f}")
    print(f"MDE:               {mde * 100:.2f}%")
    print(f"Significant:       {'YES' if stat_significant else 'NO'}")
    print("=" * 70)

    return trades_df

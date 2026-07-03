"""Risk management and strategy simulation."""

import numpy as np
import pandas as pd


def analyze_strategy_performance(
    df: pd.DataFrame,
    commission: float = 0.0003,
    be_trigger: float = 1.0,
    min_hold_bars: int = 24,
) -> pd.DataFrame | None:
    """Simulate long-only strategy with breakeven stop and minimum hold period."""
    trades = []
    in_pos = False
    entry_p, entry_d = 0.0, None
    is_breakeven = False
    entry_bar_index = 0

    for i in range(len(df)):
        sig = df["Predicted"].iloc[i]
        price = df["Price"].iloc[i]
        date = df["DateTime"].iloc[i]

        if sig == 1 and not in_pos:
            in_pos = True
            entry_p = price
            entry_d = date
            is_breakeven = False
            entry_bar_index = i
            continue

        if not in_pos:
            continue

        current_profit_pct = (price - entry_p) / entry_p * 100
        bars_in_position = i - entry_bar_index

        if not is_breakeven and current_profit_pct >= be_trigger:
            is_breakeven = True

        if is_breakeven and current_profit_pct <= 0:
            duration = date - entry_d
            trades.append(
                {
                    "Entry Date": entry_d.strftime("%Y-%m-%d %H:%M"),
                    "Exit Date": date.strftime("%Y-%m-%d %H:%M"),
                    "Profit %": round(-(commission * 2) * 100, 2),
                    "Type": "Breakeven",
                    "Duration Days": duration.days + duration.seconds / 86400.0,
                    "Duration Months": round(duration.days / 30.4375, 2),
                }
            )
            in_pos = False
            continue

        if sig == 2 or i == len(df) - 1:
            if bars_in_position < min_hold_bars and i != len(df) - 1:
                continue
            pnl = (price - entry_p) / entry_p
            pnl_net = (pnl - commission * 2) * 100
            duration = date - entry_d
            trades.append(
                {
                    "Entry Date": entry_d.strftime("%Y-%m-%d %H:%M"),
                    "Exit Date": date.strftime("%Y-%m-%d %H:%M"),
                    "Profit %": round(pnl_net, 2),
                    "Type": "Signal",
                    "Duration Days": duration.days + duration.seconds / 86400.0,
                    "Duration Months": round(duration.days / 30.4375, 2),
                }
            )
            in_pos = False

    t_df = pd.DataFrame(trades)
    if t_df.empty:
        print("No trades found.")
        return None

    t_df["Cumulative PnL %"] = t_df["Profit %"].cumsum()
    pnls = t_df["Profit %"]
    win_rate = len(pnls[pnls > 0]) / len(pnls) * 100
    pf = pnls[pnls > 0].sum() / abs(pnls[pnls < 0].sum()) if any(pnls < 0) else np.inf
    max_dd = (t_df["Cumulative PnL %"] - t_df["Cumulative PnL %"].cummax()).min()

    print("=" * 60)
    print(f"STRATEGY (min_hold={min_hold_bars}, breakeven={be_trigger}%)")
    print("=" * 60)
    print(f"Trades:       {len(t_df)}")
    print(f"Win Rate:     {win_rate:.2f}%")
    print(f"Total Return: {pnls.sum():.2f}%")
    print(f"Max DD:       {max_dd:.2f}%")
    print(f"Profit Factor:{pf:.2f}")
    print("=" * 60)
    return t_df

# Deep Learning Trading Strategies for the Russian Stock Market

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

Research codebase for a master's thesis (ITMO University, 2026) on **intraday trading strategies** for the Moscow Exchange (MOEX). The project compares classical ML baselines and several deep learning architectures on 5-minute OHLCV data.

> **Disclaimer.** This repository is academic research code, not financial advice. Past backtest performance does not guarantee future results.

---

## Experimental Results

All figures below are from the **previous research pipeline** (pre-refactor notebooks). Evaluation window for DL models: **2024-01-01 — 2026-01-01**, walk-forward backtests with commission **0.03%**. Baseline track uses a separate regression setup (4 years train / 2 years test, z-score signals).

### Best strategies per ticker (DL models)

| Ticker | Model | Return | Sharpe | Max DD | WFS | Significant |
|--------|-------|--------|--------|--------|-----|-------------|
| **GLDRUB_TOM** | MLP (wo TIs) | **78.92%** | 0.81 | −13.8% | 0.84 | Yes |
| **YDEX** | LSTM Attention (v2) | **57.77%** | 0.67 | −16.8% | 0.65 | Yes |
| **YDEX** | CNN (Wavelet wo TIs) | **53.42%** | 0.47 | −15.7% | 0.57 | Yes |
| **SBER** | LSTM Attention (v2) | **36.18%** | 0.37 | −4.1% | 0.88 | Yes |
| **GLDRUB_TOM** | CNN (Wavelet wo TIs) | 32.84% | 0.15 | −8.1% | 0.84 | Yes |
| **GLDRUB_TOM** | LSTM Attention (v2 TIs) | 30.71% | 0.51 | −13.2% | 0.53 | No |
| **GLDRUB_TOM** | CNN (w TIs) | 28.16% | 0.51 | −9.3% | 0.54 | Yes |
| **SBER** | MLP (wo TIs) | 28.85% | 0.08 | −24.1% | 0.58 | No |
| **LKOH** | CNN (wo TIs) | 20.49% | −0.24 | −10.4% | 0.50 | No |

### Full results by ticker

<details>
<summary><b>GLDRUB_TOM</b></summary>

| Model | Return | Sharpe | Max DD | WFS | Signif. |
|-------|--------|--------|--------|-----|---------|
| MLP (wo TIs) | 78.92% | 0.81 | −13.8% | 0.84 | Yes |
| CNN (Wavelet wo TIs) | 32.84% | 0.15 | −8.1% | 0.84 | Yes |
| LSTM Attention (v2 TIs) | 30.71% | 0.51 | −13.2% | 0.53 | No |
| CNN (w TIs) | 28.16% | 0.51 | −9.3% | 0.54 | Yes |
| MLP (w TIs) | 23.60% | 0.28 | −16.1% | 0.53 | No |
| LSTM Attention (v2) | 20.24% | −0.22 | −12.3% | 0.94 | No |
| CNN (wo TIs) | 20.96% | −0.31 | −8.5% | 0.49 | Yes |
| CDT 1-D CNN (wo TIs) | 13.19% | −0.43 | −14.1% | 0.82 | No |

</details>

<details>
<summary><b>YDEX</b></summary>

| Model | Return | Sharpe | Max DD | WFS | Signif. |
|-------|--------|--------|--------|-----|---------|
| LSTM Attention (v2) | 57.77% | 0.67 | −16.8% | 0.65 | Yes |
| MLP (wo TIs) | 57.70% | 0.40 | −35.9% | 0.61 | No |
| CNN (Wavelet wo TIs) | 53.42% | 0.47 | −15.7% | 0.57 | Yes |
| MLP (w TIs) | 39.22% | 0.26 | −23.4% | 0.53 | No |
| CNN (w TIs) | 36.86% | 0.24 | −23.1% | 0.54 | No |
| CDT 1-D CNN (wo TIs) | 32.55% | 0.12 | −25.2% | 0.61 | No |
| CNN (wo TIs) | 10.96% | −0.35 | −24.8% | 0.50 | No |
| LSTM Attention (v2 TIs) | 10.05% | −0.19 | −30.8% | 0.52 | No |

</details>

<details>
<summary><b>SBER</b></summary>

| Model | Return | Sharpe | Max DD | WFS | Signif. |
|-------|--------|--------|--------|-----|---------|
| LSTM Attention (v2) | 36.18% | 0.37 | −4.1% | 0.88 | Yes |
| MLP (wo TIs) | 28.85% | 0.08 | −24.1% | 0.58 | No |
| CNN (Wavelet wo TIs) | 3.95% | −0.57 | −17.9% | 0.56 | No |
| CNN (wo TIs) | −1.93% | −0.95 | −19.7% | 0.45 | No |
| CDT 1-D CNN (wo TIs) | −7.47% | −0.97 | −21.0% | 0.61 | No |
| CNN (w TIs) | −12.05% | −0.79 | −19.3% | 0.53 | No |
| MLP (w TIs) | −12.05% | −1.09 | −23.7% | 0.52 | No |
| LSTM Attention (v2 TIs) | −22.93% | −0.99 | −28.3% | 0.55 | No |

</details>

<details>
<summary><b>LKOH</b></summary>

| Model | Return | Sharpe | Max DD | WFS | Signif. |
|-------|--------|--------|--------|-----|---------|
| CNN (wo TIs) | 20.49% | −0.24 | −10.4% | 0.50 | No |
| LSTM Attention (v2) | −4.36% | −1.02 | −17.0% | 0.63 | No |
| CNN (w TIs) | −4.99% | −1.31 | −12.8% | 0.61 | No |
| LSTM Attention (v2 TIs) | −3.66% | −1.55 | −20.2% | 0.66 | No |
| MLP (wo TIs) | −19.20% | −0.73 | −38.8% | 0.59 | No |
| MLP (w TIs) | −27.97% | −1.14 | −36.4% | 0.51 | No |
| CDT 1-D CNN (wo TIs) | −33.76% | −1.72 | −46.8% | 0.60 | No |
| CNN (Wavelet wo TIs) | −43.12% | −1.75 | −49.4% | 0.52 | No |

</details>

### Classical ML baseline (best per ticker)

| Ticker | Best model | Return | Sharpe | Win Rate | Trades |
|--------|------------|--------|--------|----------|--------|
| YDEX | CatBoost | 42.64% | 2.77 | 45.2% | 498 |
| YDEX | Ridge | 42.21% | 10.72 | 60.4% | 91 |
| TATN | Ridge | 31.55% | 16.35 | 68.8% | 48 |
| SBER | CatBoost | 14.45% | 2.31 | 51.9% | 295 |
| LKOH | Ridge | 4.60% | 13.42 | 69.6% | 23 |

> DL models with wavelet denoising and MLP showed the strongest risk-adjusted results on **GLDRUB_TOM** and **YDEX**. Classical baselines were competitive on **YDEX** (Ridge/CatBoost) but underperformed on gold. LKOH remained challenging across all architectures.

---

## What This Repo Contains

The project is organized as a **Python library** (`src/fqw/`) plus **thin Jupyter notebooks** that orchestrate experiments. Core logic lives in the library; notebooks are entry points.

```
┌─────────────────────────────────────────────────────────────┐
│  notebooks/          Thin orchestrators (Run All)           │
│  01 … 08             import fqw, set config, call runners   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  src/fqw/            Reusable library                       │
│  data · features · labeling · datasets · models             │
│  training · backtest · experiments · viz                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  data/               5-min parquet candles (not in git)     │
│  results/            Backtest CSV / probability parquet     │
└─────────────────────────────────────────────────────────────┘
```

### Research Tracks

| # | Notebook | Description | Library entry point |
|---|----------|-------------|---------------------|
| 01 | `01_dataset_formation.ipynb` | Download candles via Tinkoff Invest API | `fqw.data.api` |
| 02 | `02_baseline.ipynb` | Ridge, Lasso, LightGBM, CatBoost + Optuna | `fqw.experiments.baseline` |
| 03 | `03_models_cnn.ipynb` | **Main pipeline:** wavelet + 1D CNN | `fqw.experiments.batch_cnn` |
| 04 | `04_mlp.ipynb` | Deep MLP, record-based windows | `fqw.experiments.mlp` |
| 05 | `05_cdt_macro.ipynb` | CDT 1D CNN + macro feature ablation | `fqw.experiments.cdt_macro` |
| 06 | `06_visualization.ipynb` | Equity curves, metrics, heatmaps | `fqw.viz` |
| 08 | `08_alpha_search.ipynb` | Per-ticker optimal labeling alpha | `fqw.labeling.alpha_search` |

Legacy monolithic notebooks are preserved in `archive/*_legacy.ipynb` for reference.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/JayRox73/Deep_Learning_Trading_Strategy_for_Russian_Stock_Market.git
cd Deep_Learning_Trading_Strategy_for_Russian_Stock_Market

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

Alternative (extras via pyproject):

```bash
pip install -e ".[baseline,data]"
```

### 2. Prepare market data

**Option A — use existing parquet files**

Place 5-minute candles in `data/` (or symlink to your data folder):

```text
data/
├── SBER_5min.parquet
├── YDEX_5min.parquet
├── LKOH_5min.parquet
└── MOEX_1day.parquet      # required for CDT macro experiments (05)
```

**Option B — download via API**

Create `.env` in the project root:

```env
TOKEN=your_tinkoff_invest_api_token
```

Then open and run `notebooks/01_dataset_formation.ipynb`.

### 3. Run an experiment

**Via Jupyter (recommended):**

```bash
jupyter notebook notebooks/
```

Open the notebook you need and run all cells. Paths in notebooks are relative to `notebooks/` (`../data`, `../results`).

**Via Python:**

```python
from fqw.config import CNNConfig
from fqw.experiments import run_batch_backtest_to_csv

cfg = CNNConfig(data_dir="data")

run_batch_backtest_to_csv(
    tickers=["YDEX"],
    alpha=1.0,
    confidence_threshold=0.75,
    config=cfg,
    use_wavelet=True,
    filename="results/cnn_yndx_smoke.csv",
)
```

---

## Repository Layout

```text
.
├── src/fqw/                  # Core library
│   ├── config.py             # CNNConfig, MLPConfig, CDTConfig, BaselineConfig
│   ├── data/                 # API, loading, cleaning, resampling
│   ├── features/             # Technical indicators, wavelet, macro, baseline
│   ├── labeling/             # Volatility labels, alpha search
│   ├── datasets/             # Tensor builders (CNN / MLP / CDT)
│   ├── models/               # CNN, MLP, CDT architectures
│   ├── training/             # Walk-forward and moving-window trainers
│   ├── backtest/             # Trade analysis, risk rules
│   ├── metrics/              # Weighted F-score and classification metrics
│   ├── experiments/          # High-level experiment runners
│   └── viz/                  # Results scanning and plotting
├── notebooks/                # Thin orchestrators — start here
├── archive/                  # Legacy notebooks (pre-refactor)
├── data/                     # Market data (gitignored; symlink OK)
├── results/                  # Experiment outputs (gitignored)
├── pyproject.toml
└── requirements.txt
```

---

## Methodology (Summary)

| Aspect | Setting |
|--------|---------|
| **Data** | 5-minute MOEX candles, 2020–2026 |
| **Walk-forward** | 48 months train → 1 month test → retrain |
| **Labels** | Adaptive volatility: Up (1) / Down (2) / Flat (0) |
| **CNN preprocessing** | Symlet-4 wavelet denoising, level 3 |
| **Costs** | Commission 0.03%; slippage 0.02% (baseline track) |
| **Risk management** | Breakeven stop, minimum hold period |
| **Metrics** | Sharpe, profit factor, weighted F-score, win rate |

Three DL tracks use **different tensor pipelines** (per-window scaler for CNN, session scaler for MLP, global z-score for CDT). Each architecture was tuned for its own pipeline.

### Default thesis tickers

`SBER`, `MGNT`, `VTBR`, `TATN`, `LKOH`, `YDEX`, `GLDRUB_TOM`

Configure via `CNNConfig`, `MLPConfig`, `CDTConfig`, or `BaselineConfig` in `src/fqw/config.py`.

---

## Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: fqw` | Run `pip install -e .` from repo root |
| `FileNotFoundError` for parquet | Check `data_dir` — use `../data` from notebooks, `data` from repo root |
| CDT macro fails | Ensure `MOEX_1day.parquet` exists in `data/` |
| Empty visualization | Run model notebooks first; outputs go to `results/` as `*_probs.parquet` |
| Tinkoff API errors | Verify `TOKEN` in `.env`; respect API rate limits |

---

## Tech Stack

- **Python** 3.10+
- **TensorFlow / Keras** 2.15 — CNN, MLP, CDT
- **scikit-learn**, **pandas**, **numpy**, **PyWavelets**, **scipy**
- **LightGBM**, **CatBoost**, **Optuna** — baseline track
- **Tinkoff Invest API** — historical data download

---

## Author

**Alexey Smirnov**  
Master's Thesis — Data Analysis / Quantitative Finance  
ITMO University, 2026

---

## License

[Apache License 2.0](LICENSE)

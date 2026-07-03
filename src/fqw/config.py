"""Experiment configuration presets."""

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_TICKERS = ["SBER", "MGNT", "VTBR", "TATN", "LKOH", "YDEX", "GLDRUB_TOM"]

DEFAULT_PARAM_PAIRS = [
    (0.55, 0.50),
    (0.55, 0.55),
    (0.55, 0.60),
    (0.80, 0.50),
    (0.80, 0.55),
    (0.80, 0.60),
    (0.80, 0.65),
    (0.80, 0.70),
    (0.80, 0.75),
]


@dataclass
class CNNConfig:
    """Settings for the main 1D CNN + wavelet pipeline (models.ipynb)."""

    train_months: int = 48
    test_months: int = 1
    window_size: int = 24
    alpha: float = 1.0
    learning_rate: float = 1e-4
    confidence_threshold: float = 0.75
    min_hold_bars: int = 6
    commission: float = 0.0003
    rf_annual: float = 0.15
    be_trigger: float = 1.0
    initial_epochs: int = 15
    fine_tune_epochs: int = 1
    batch_size: int = 32
    dropout: float = 0.3
    data_dir: Path = field(default_factory=lambda: Path("data"))
    tickers: list[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    param_pairs: list[tuple[float, float]] = field(
        default_factory=lambda: list(DEFAULT_PARAM_PAIRS)
    )


@dataclass
class MLPConfig:
    """Settings for deep MLP pipeline (04_mlp.ipynb)."""

    window_size: int = 24
    alpha: float = 0.55
    train_records: int = 50_000
    val_records: int = 2_000
    test_records: int = 2_000
    step_records: int = 2_000
    learning_rate: float = 1e-3
    dropout: float = 0.7
    l2_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 15
    commission: float = 0.0003
    rf_annual: float = 0.09
    data_dir: Path = field(default_factory=lambda: Path("data"))
    tickers: list[str] = field(default_factory=lambda: ["SBER", "LKOH", "YDEX", "GLDRUB_TOM"])


@dataclass
class CDTConfig:
    """Settings for CDT 1D CNN + macro ablation (05_cdt_macro.ipynb)."""

    train_months: int = 48
    test_months: int = 1
    window_size: int = 24
    alpha: float = 1.0
    learning_rate: float = 1e-3
    dropout: float = 0.3
    l2_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 30
    confidence_threshold: float = 0.50
    beta1: float = 0.5
    beta2: float = 0.125
    beta3: float = 0.125
    commission: float = 0.0003
    be_trigger_pct: float = 1.0
    data_dir: Path = field(default_factory=lambda: Path("data"))
    tickers: list[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))


CDT_EXPERIMENTS = [
    {"name": "CDT OHLCV", "features": ["Open", "High", "Low", "Close", "Volume"]},
    {
        "name": "CDT + MOEX",
        "features": ["Open", "High", "Low", "Close", "Volume", "MOEX_Close", "MOEX_Vol_20d", "Relative_Strength"],
    },
    {
        "name": "CDT + Rate/Infl",
        "features": ["Open", "High", "Low", "Close", "Volume", "KeyRate", "Inflation"],
    },
    {
        "name": "CDT + All Macro",
        "features": [
            "Open", "High", "Low", "Close", "Volume",
            "MOEX_Close", "MOEX_Vol_20d", "Relative_Strength", "KeyRate", "Inflation",
        ],
    },
]


@dataclass
class BaselineConfig:
    """Settings for classical ML baseline (baseline.ipynb)."""

    horizon: int = 12
    train_years: int = 4
    test_years: int = 2
    z_threshold: float = 1.0
    commission: float = 0.0003
    slippage: float = 0.0002
    short_rate: float = 0.10
    bars_per_year: int = 252 * 78
    n_optuna_trials: int = 50
    n_cv_splits: int = 3
    optuna_seed: int = 42
    optuna_subsample: int = 5
    data_dir: Path = field(default_factory=lambda: Path("data"))
    start_date: str = "2020-01-01"
    end_date: str = "2026-01-01"
    target_tickers: list[str] = field(default_factory=lambda: ["SBER", "LKOH", "YDEX", "GLDRUB_TOM"])
    model_names: list[str] = field(default_factory=lambda: ["Ridge", "Lasso", "LightGBM", "CatBoost"])

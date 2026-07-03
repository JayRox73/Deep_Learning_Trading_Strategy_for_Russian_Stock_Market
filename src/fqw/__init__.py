"""FQW — Deep learning trading strategies for the Russian stock market."""

from fqw.config import (
    BaselineConfig,
    CDTConfig,
    CNNConfig,
    DEFAULT_TICKERS,
    MLPConfig,
)

__all__ = [
    "CNNConfig",
    "MLPConfig",
    "CDTConfig",
    "BaselineConfig",
    "DEFAULT_TICKERS",
]

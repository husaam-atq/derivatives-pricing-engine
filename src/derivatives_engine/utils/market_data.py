"""Sample and optional market data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def project_root() -> Path:
    """Return repository root inferred from the installed source file location."""

    return Path(__file__).resolve().parents[3]


def load_sample_options(path: str | Path | None = None) -> pd.DataFrame:
    """Load the bundled synthetic sample option chain."""

    sample_path = (
        Path(path)
        if path is not None
        else project_root() / "data" / "sample_options.csv"
    )
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample option data not found at {sample_path}")
    return pd.read_csv(sample_path)


def try_fetch_yfinance_price(ticker: str) -> float | None:
    """Best-effort yfinance spot fetch. Returns None if unavailable."""

    try:
        import yfinance as yf  # type: ignore[import-not-found]

        history = yf.Ticker(ticker).history(period="5d")
        if history.empty:
            return None
        return float(history["Close"].dropna().iloc[-1])
    except Exception:
        return None

"""Tests for implied volatility solvers."""

from __future__ import annotations

import pytest

from derivatives_engine.models.black_scholes import price
from derivatives_engine.risk.implied_volatility import implied_volatility


def test_brent_recovers_known_volatility() -> None:
    market = price(100, 100, 1, 0.05, 0.0, 0.30, "call")
    result = implied_volatility(market, 100, 100, 1, 0.05, 0.0, "call", "brent")
    assert result.implied_volatility == pytest.approx(0.30, abs=1e-6)


def test_newton_recovers_known_volatility() -> None:
    market = price(100, 100, 1, 0.05, 0.0, 0.25, "put")
    result = implied_volatility(market, 100, 100, 1, 0.05, 0.0, "put", "newton", 0.2)
    assert result.implied_volatility == pytest.approx(0.25, abs=1e-5)


def test_auto_falls_back_when_newton_is_unstable() -> None:
    market = price(100, 100, 1, 0.05, 0.0, 0.20, "call")
    result = implied_volatility(market, 100, 100, 1, 0.05, 0.0, "call", "auto", 8.0)
    assert result.implied_volatility == pytest.approx(0.20, abs=1e-6)
    assert result.method in {"newton", "brent"}


def test_invalid_market_price_raises() -> None:
    with pytest.raises(ValueError):
        implied_volatility(200.0, 100, 100, 1, 0.05, 0.0, "call", "brent")

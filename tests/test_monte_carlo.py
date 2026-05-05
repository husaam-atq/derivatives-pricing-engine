"""Tests for Monte Carlo pricing."""

from __future__ import annotations

from derivatives_engine.models.black_scholes import call_price
from derivatives_engine.models.monte_carlo import (
    compare_variance_reduction,
    price_european_option,
)


def test_european_mc_close_to_black_scholes() -> None:
    analytical = call_price(100, 100, 1, 0.05, 0.0, 0.20)
    result = price_european_option(100, 100, 1, 0.05, 0.0, 0.20, "call", 100_000, 123)
    assert abs(result.price - analytical) < 0.15 or (
        result.confidence_interval[0] <= analytical <= result.confidence_interval[1]
    )
    assert result.standard_error > 0
    assert result.confidence_interval[0] < result.price < result.confidence_interval[1]


def test_variance_reduction_reduces_standard_error() -> None:
    table = compare_variance_reduction(
        100, 100, 1, 0.05, 0.0, 0.20, "call", 100_000, 789
    )
    plain = float(table.loc[table["method"] == "plain", "standard_error"].iloc[0])
    antithetic = float(
        table.loc[table["method"] == "antithetic", "standard_error"].iloc[0]
    )
    control = float(
        table.loc[table["method"] == "control_variate_stock", "standard_error"].iloc[0]
    )
    assert antithetic < plain
    assert control < plain


def test_reproducible_seed() -> None:
    first = price_european_option(100, 100, 1, 0.05, 0.0, 0.20, "call", 10_000, 7)
    second = price_european_option(100, 100, 1, 0.05, 0.0, 0.20, "call", 10_000, 7)
    assert first.price == second.price
    assert first.standard_error == second.standard_error

"""Tests for CRR binomial tree pricing."""

from __future__ import annotations

import pytest

from derivatives_engine.models.binomial_tree import convergence_table, crr_price
from derivatives_engine.models.black_scholes import price


def test_european_tree_converges_to_black_scholes() -> None:
    table = convergence_table(100, 100, 1, 0.05, 0.0, 0.20, "call")
    assert table.loc[table["steps"] == 1000, "absolute_error"].iloc[0] < 0.02


def test_american_put_at_least_european_put() -> None:
    european = crr_price(100, 100, 1, 0.05, 0.0, 0.20, "put", "european", 500)
    american = crr_price(100, 100, 1, 0.05, 0.0, 0.20, "put", "american", 500)
    assert american >= european


def test_american_call_without_dividends_matches_european_call() -> None:
    american = crr_price(100, 100, 1, 0.05, 0.0, 0.20, "call", "american", 1000)
    analytical = price(100, 100, 1, 0.05, 0.0, 0.20, "call")
    assert american == pytest.approx(analytical, abs=0.02)

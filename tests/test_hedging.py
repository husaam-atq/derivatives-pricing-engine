"""Tests for delta hedging simulator."""

from __future__ import annotations

from derivatives_engine.risk.hedging import (
    compare_rebalance_frequencies,
    simulate_delta_hedging,
)


def test_hedging_simulator_runs_and_outputs_columns() -> None:
    result = simulate_delta_hedging(
        100, 100, 1, 0.05, 0.0, 0.20, "call", "short", 500, 252, 5, 0.0, 42
    )
    assert len(result.pnl) == 500
    assert {"step", "time", "spot", "option_value", "stock_shares", "cash"}.issubset(
        result.path_details.columns
    )
    assert {"mean_pnl", "std_pnl", "p05", "p50", "p95"}.issubset(result.summary)


def test_more_frequent_rebalancing_reduces_error_without_costs() -> None:
    comparison = compare_rebalance_frequencies(
        100,
        100,
        1,
        0.05,
        0.0,
        0.20,
        "call",
        "short",
        (12, 252),
        n_paths=1_000,
        n_steps=252,
        transaction_cost_bps=0.0,
        seed=42,
    )
    monthly = float(
        comparison.loc[comparison["rebalances_per_year"] == 12, "std_pnl"].iloc[0]
    )
    daily = float(
        comparison.loc[comparison["rebalances_per_year"] == 252, "std_pnl"].iloc[0]
    )
    assert daily < monthly

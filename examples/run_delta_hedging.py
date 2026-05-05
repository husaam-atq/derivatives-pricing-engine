# ruff: noqa: E402, I001
"""Example: run a Black-Scholes delta hedging frequency comparison."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from derivatives_engine.risk.hedging import (
    compare_rebalance_frequencies,
    simulate_delta_hedging,
)


def main() -> None:
    result = simulate_delta_hedging(
        100.0,
        100.0,
        1.0,
        0.05,
        0.0,
        0.20,
        "call",
        "short",
        n_paths=2_000,
        n_steps=252,
        rebalance_every=1,
        transaction_cost_bps=0.0,
        seed=42,
    )
    print("Delta hedging example")
    for key, value in result.summary.items():
        print(f"{key}: {value:.6f}")
    print("\nRebalance frequency comparison:")
    comparison = compare_rebalance_frequencies(
        100.0,
        100.0,
        1.0,
        0.05,
        0.0,
        0.20,
        frequencies_per_year=(12, 52, 252),
        n_paths=2_000,
        seed=42,
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

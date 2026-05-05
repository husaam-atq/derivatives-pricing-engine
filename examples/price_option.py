# ruff: noqa: E402, I001
"""Example: price a vanilla option and compute Greeks / implied volatility."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from derivatives_engine.models.black_scholes import (
    call_price,
    put_call_parity_error,
    put_price,
)
from derivatives_engine.risk.greeks import greek_table
from derivatives_engine.risk.implied_volatility import implied_volatility


def main() -> None:
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.0
    sigma = 0.20

    call = call_price(S, K, T, r, q, sigma)
    put = put_price(S, K, T, r, q, sigma)
    iv = implied_volatility(call, S, K, T, r, q, "call", "auto")

    print("Black-Scholes-Merton example")
    print(f"Call price: {call:.6f}")
    print(f"Put price:  {put:.6f}")
    print(f"Put-call parity error: {put_call_parity_error(S, K, T, r, q, sigma):.3e}")
    print(f"Recovered implied volatility: {iv.implied_volatility:.6f} ({iv.method})")
    print("Greeks:")
    for name, value in greek_table(S, K, T, r, q, sigma, "call").items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()

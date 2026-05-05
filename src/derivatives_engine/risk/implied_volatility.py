"""Robust implied volatility solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from derivatives_engine.models.black_scholes import price
from derivatives_engine.risk.greeks import vega

OptionType = Literal["call", "put"]
Method = Literal["brent", "newton", "auto"]


@dataclass(frozen=True)
class ImpliedVolResult:
    """Implied volatility solver result."""

    implied_volatility: float
    converged: bool
    method: str
    iterations: int
    fallback_used: bool = False


def no_arbitrage_bounds(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> tuple[float, float]:
    """Return European option lower and upper no-arbitrage bounds."""

    if S <= 0 or K <= 0 or T < 0:
        raise ValueError("S and K must be positive and T must be non-negative.")
    discounted_spot = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)
    if option_type == "call":
        return max(0.0, discounted_spot - discounted_strike), discounted_spot
    if option_type == "put":
        return max(0.0, discounted_strike - discounted_spot), discounted_strike
    raise ValueError("option_type must be either 'call' or 'put'.")


def validate_market_price(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    tolerance: float = 1e-10,
) -> None:
    """Raise ValueError if a market price violates European no-arbitrage bounds."""

    lower, upper = no_arbitrage_bounds(S, K, T, r, q, option_type)
    if market_price < lower - tolerance or market_price > upper + tolerance:
        raise ValueError(
            "Market price violates no-arbitrage bounds: "
            f"price={market_price:.10g}, lower={lower:.10g}, upper={upper:.10g}."
        )


def implied_volatility_brent(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    low: float = 1e-9,
    high: float = 5.0,
    tolerance: float = 1e-12,
    max_iterations: int = 100,
) -> ImpliedVolResult:
    """Solve implied volatility using Brent's bracketing method."""

    validate_market_price(market_price, S, K, T, r, q, option_type)
    lower, upper = no_arbitrage_bounds(S, K, T, r, q, option_type)
    if abs(market_price - lower) < 1e-12:
        return ImpliedVolResult(0.0, True, "brent", 0)
    if abs(market_price - upper) < 1e-12:
        return ImpliedVolResult(high, True, "brent", 0)

    def objective(vol: float) -> float:
        return float(price(S, K, T, r, q, vol, option_type) - market_price)

    f_low = objective(low)
    f_high = objective(high)
    while f_high < 0 and high < 20.0:
        high *= 2.0
        f_high = objective(high)

    if f_low * f_high > 0:
        raise ValueError("Unable to bracket implied volatility root.")

    root, info = brentq(
        objective,
        low,
        high,
        xtol=tolerance,
        rtol=1e-12,
        maxiter=max_iterations,
        full_output=True,
    )
    return ImpliedVolResult(float(root), bool(info.converged), "brent", info.iterations)


def implied_volatility_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    initial_guess: float = 0.2,
    tolerance: float = 1e-10,
    max_iterations: int = 50,
) -> ImpliedVolResult:
    """Solve implied volatility using Newton-Raphson iterations."""

    validate_market_price(market_price, S, K, T, r, q, option_type)
    sigma = float(initial_guess)
    if sigma <= 0:
        sigma = 0.2

    for iteration in range(1, max_iterations + 1):
        model_price = float(price(S, K, T, r, q, sigma, option_type))
        diff = model_price - market_price
        if abs(diff) < tolerance:
            return ImpliedVolResult(sigma, True, "newton", iteration)
        sensitivity = vega(S, K, T, r, q, sigma)
        if not np.isfinite(sensitivity) or sensitivity < 1e-10:
            break
        sigma -= diff / sensitivity
        if not np.isfinite(sigma) or sigma <= 0 or sigma > 10.0:
            break

    raise RuntimeError("Newton-Raphson implied volatility solver failed to converge.")


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    method: Method = "auto",
    initial_guess: float = 0.2,
) -> ImpliedVolResult:
    """Solve implied volatility with Brent, Newton, or Newton-with-Brent fallback."""

    if method == "brent":
        return implied_volatility_brent(market_price, S, K, T, r, q, option_type)
    if method == "newton":
        return implied_volatility_newton(
            market_price, S, K, T, r, q, option_type, initial_guess
        )
    if method != "auto":
        raise ValueError("method must be 'brent', 'newton', or 'auto'.")

    try:
        return implied_volatility_newton(
            market_price, S, K, T, r, q, option_type, initial_guess
        )
    except (RuntimeError, ValueError):
        brent_result = implied_volatility_brent(
            market_price, S, K, T, r, q, option_type
        )
        return ImpliedVolResult(
            brent_result.implied_volatility,
            brent_result.converged,
            "brent",
            brent_result.iterations,
            fallback_used=True,
        )


def implied_volatility_vectorized(
    market_prices: np.ndarray,
    S: float | np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray = 0.0,
    option_type: OptionType = "call",
    method: Method = "brent",
) -> np.ndarray:
    """Vectorised helper that loops over scalar robust solvers."""

    prices_arr, S_arr, K_arr, T_arr, r_arr, q_arr = np.broadcast_arrays(
        np.asarray(market_prices, dtype=float),
        np.asarray(S, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(r, dtype=float),
        np.asarray(q, dtype=float),
    )
    output = np.empty_like(prices_arr, dtype=float)
    for idx in np.ndindex(prices_arr.shape):
        output[idx] = implied_volatility(
            float(prices_arr[idx]),
            float(S_arr[idx]),
            float(K_arr[idx]),
            float(T_arr[idx]),
            float(r_arr[idx]),
            float(q_arr[idx]),
            option_type,
            method,
        ).implied_volatility
    return output

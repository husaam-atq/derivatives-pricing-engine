"""Analytical and finite-difference Black-Scholes-Merton Greeks."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import norm

from derivatives_engine.models.black_scholes import d1, d2, price

OptionType = Literal["call", "put"]


def _check_type(option_type: str) -> None:
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")


def delta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
) -> float:
    """Return option delta."""

    _check_type(option_type)
    d1_val = float(d1(S, K, T, r, q, sigma))
    if option_type == "call":
        return float(np.exp(-q * T) * norm.cdf(d1_val))
    return float(np.exp(-q * T) * (norm.cdf(d1_val) - 1.0))


def gamma(
    S: float, K: float, T: float, r: float, q: float = 0.0, sigma: float = 0.2
) -> float:
    """Return option gamma. Same for calls and puts under BSM."""

    d1_val = float(d1(S, K, T, r, q, sigma))
    return float(np.exp(-q * T) * norm.pdf(d1_val) / (S * sigma * np.sqrt(T)))


def vega(
    S: float, K: float, T: float, r: float, q: float = 0.0, sigma: float = 0.2
) -> float:
    """Return vega per 1.00 volatility change."""

    d1_val = float(d1(S, K, T, r, q, sigma))
    return float(S * np.exp(-q * T) * norm.pdf(d1_val) * np.sqrt(T))


def vega_per_vol_point(
    S: float, K: float, T: float, r: float, q: float = 0.0, sigma: float = 0.2
) -> float:
    """Return vega per one volatility point, e.g. sigma + 0.01."""

    return vega(S, K, T, r, q, sigma) / 100.0


def theta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
) -> float:
    """Return annualised calendar theta.

    Convention: theta is option value decay with calendar time, i.e.
    ``-dV/dT`` where ``T`` is remaining maturity in years.
    """

    _check_type(option_type)
    d1_val = float(d1(S, K, T, r, q, sigma))
    d2_val = float(d2(S, K, T, r, q, sigma))
    first_term = -S * np.exp(-q * T) * norm.pdf(d1_val) * sigma / (2.0 * np.sqrt(T))
    if option_type == "call":
        return float(
            first_term
            - r * K * np.exp(-r * T) * norm.cdf(d2_val)
            + q * S * np.exp(-q * T) * norm.cdf(d1_val)
        )
    return float(
        first_term
        + r * K * np.exp(-r * T) * norm.cdf(-d2_val)
        - q * S * np.exp(-q * T) * norm.cdf(-d1_val)
    )


def daily_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
    days_per_year: int = 365,
) -> float:
    """Return calendar theta per day."""

    return theta(S, K, T, r, q, sigma, option_type) / days_per_year


def rho(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
) -> float:
    """Return rho per 1.00 interest-rate change."""

    _check_type(option_type)
    d2_val = float(d2(S, K, T, r, q, sigma))
    if option_type == "call":
        return float(K * T * np.exp(-r * T) * norm.cdf(d2_val))
    return float(-K * T * np.exp(-r * T) * norm.cdf(-d2_val))


def rho_per_bp(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
) -> float:
    """Return rho per basis point rate move."""

    return rho(S, K, T, r, q, sigma, option_type) / 10_000.0


def finite_difference_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
    h: float = 1e-3,
) -> float:
    """Central finite-difference delta."""

    return (
        price(S + h, K, T, r, q, sigma, option_type)
        - price(S - h, K, T, r, q, sigma, option_type)
    ) / (2.0 * h)


def finite_difference_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
    h: float = 1e-2,
) -> float:
    """Central finite-difference gamma."""

    return (
        price(S + h, K, T, r, q, sigma, option_type)
        - 2.0 * price(S, K, T, r, q, sigma, option_type)
        + price(S - h, K, T, r, q, sigma, option_type)
    ) / (h**2)


def finite_difference_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
    h: float = 1e-4,
) -> float:
    """Central finite-difference vega per 1.00 volatility change."""

    return (
        price(S, K, T, r, q, sigma + h, option_type)
        - price(S, K, T, r, q, sigma - h, option_type)
    ) / (2.0 * h)


def finite_difference_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
    h: float = 1e-4,
) -> float:
    """Central finite-difference annualised calendar theta.

    Uses ``(V(T-h) - V(T+h)) / (2h)`` to match the ``-dV/dT`` convention.
    """

    if T <= h:
        raise ValueError("T must be larger than h for finite-difference theta.")
    return (
        price(S, K, T - h, r, q, sigma, option_type)
        - price(S, K, T + h, r, q, sigma, option_type)
    ) / (2.0 * h)


def finite_difference_rho(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
    h: float = 1e-5,
) -> float:
    """Central finite-difference rho per 1.00 interest-rate change."""

    return (
        price(S, K, T, r + h, q, sigma, option_type)
        - price(S, K, T, r - h, q, sigma, option_type)
    ) / (2.0 * h)


def greek_table(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
) -> dict[str, float]:
    """Return a compact dictionary of analytical Greeks."""

    return {
        "delta": delta(S, K, T, r, q, sigma, option_type),
        "gamma": gamma(S, K, T, r, q, sigma),
        "vega": vega(S, K, T, r, q, sigma),
        "vega_per_vol_point": vega_per_vol_point(S, K, T, r, q, sigma),
        "theta_annual": theta(S, K, T, r, q, sigma, option_type),
        "theta_daily": daily_theta(S, K, T, r, q, sigma, option_type),
        "rho": rho(S, K, T, r, q, sigma, option_type),
        "rho_per_bp": rho_per_bp(S, K, T, r, q, sigma, option_type),
    }

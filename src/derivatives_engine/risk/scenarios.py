"""Scenario and stress testing utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from derivatives_engine.models.black_scholes import price
from derivatives_engine.risk.greeks import greek_table

OptionType = Literal["call", "put"]


def spot_shock_table(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
    shocks: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return option prices under spot shocks."""

    if shocks is None:
        shocks = np.linspace(-0.2, 0.2, 9)
    base = float(price(S, K, T, r, q, sigma, option_type))
    rows = []
    for shock in shocks:
        shocked_spot = S * (1.0 + shock)
        shocked_price = float(price(shocked_spot, K, T, r, q, sigma, option_type))
        rows.append(
            {
                "spot_shock": shock,
                "spot": shocked_spot,
                "price": shocked_price,
                "price_change": shocked_price - base,
            }
        )
    return pd.DataFrame(rows)


def volatility_shock_table(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
    vol_point_shocks: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return option prices under absolute volatility-point shocks."""

    if vol_point_shocks is None:
        vol_point_shocks = np.linspace(-0.10, 0.10, 9)
    base = float(price(S, K, T, r, q, sigma, option_type))
    rows = []
    for shock in vol_point_shocks:
        shocked_sigma = max(1e-6, sigma + shock)
        shocked_price = float(price(S, K, T, r, q, shocked_sigma, option_type))
        rows.append(
            {
                "vol_shock": shock,
                "sigma": shocked_sigma,
                "price": shocked_price,
                "price_change": shocked_price - base,
            }
        )
    return pd.DataFrame(rows)


def rate_shock_table(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
    rate_shocks: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return option prices under rate shocks."""

    if rate_shocks is None:
        rate_shocks = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
    base = float(price(S, K, T, r, q, sigma, option_type))
    rows = []
    for shock in rate_shocks:
        shocked_price = float(price(S, K, T, r + shock, q, sigma, option_type))
        rows.append(
            {
                "rate_shock": shock,
                "rate": r + shock,
                "price": shocked_price,
                "price_change": shocked_price - base,
            }
        )
    return pd.DataFrame(rows)


def time_decay_table(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
    observations: int = 10,
) -> pd.DataFrame:
    """Return option prices as maturity decays toward expiry."""

    maturities = np.linspace(T, 1e-6, observations)
    rows = []
    for maturity in maturities:
        rows.append(
            {
                "remaining_maturity": maturity,
                "price": float(price(S, K, maturity, r, q, sigma, option_type)),
            }
        )
    return pd.DataFrame(rows)


def combined_stress_matrix(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
    spot_shocks: np.ndarray | None = None,
    vol_shocks: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return matrix of option prices under spot and volatility shocks."""

    if spot_shocks is None:
        spot_shocks = np.linspace(-0.2, 0.2, 9)
    if vol_shocks is None:
        vol_shocks = np.linspace(-0.10, 0.10, 9)
    rows = []
    for spot_shock in spot_shocks:
        row = {"spot_shock": spot_shock}
        for vol_shock in vol_shocks:
            stressed_price = float(
                price(
                    S * (1.0 + spot_shock),
                    K,
                    T,
                    r,
                    q,
                    max(1e-6, sigma + vol_shock),
                    option_type,
                )
            )
            row[f"vol_{vol_shock:+.2f}"] = stressed_price
        rows.append(row)
    return pd.DataFrame(rows)


def greek_exposure_table(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
    quantity: float = 1.0,
) -> pd.DataFrame:
    """Return Greek exposures for an option position."""

    values = greek_table(S, K, T, r, q, sigma, option_type)
    return pd.DataFrame(
        {
            "greek": list(values.keys()),
            "per_contract": list(values.values()),
            "position_exposure": [quantity * v for v in values.values()],
        }
    )

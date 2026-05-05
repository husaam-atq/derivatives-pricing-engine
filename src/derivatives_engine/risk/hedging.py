"""Delta hedging simulation under Black-Scholes assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

from derivatives_engine.config import TRADING_DAYS_PER_YEAR
from derivatives_engine.models.black_scholes import price
from derivatives_engine.risk.greeks import delta

OptionType = Literal["call", "put"]
Position = Literal["short", "long"]


@dataclass(frozen=True)
class HedgingResult:
    """Delta hedging simulation result."""

    pnl: np.ndarray
    summary: dict[str, float]
    path_details: pd.DataFrame


def _vectorized_delta(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
) -> np.ndarray:
    """Fast vectorised BSM delta for hedging paths."""

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


def simulate_delta_hedging(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
    position: Position = "short",
    n_paths: int = 5_000,
    n_steps: int = TRADING_DAYS_PER_YEAR,
    rebalance_every: int = 1,
    transaction_cost_bps: float = 0.0,
    seed: int | None = None,
    real_world_drift: float | None = None,
) -> HedgingResult:
    """Simulate dynamic delta hedging for a European option.

    ``position='short'`` means the desk sells one option and delta hedges it.
    P&L is reported after liquidating the hedge and settling the option payoff.
    """

    if position not in {"short", "long"}:
        raise ValueError("position must be either 'short' or 'long'.")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")
    if S0 <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S0, K, T and sigma must be positive.")
    if n_paths < 1 or n_steps < 1 or rebalance_every < 1:
        raise ValueError("n_paths, n_steps and rebalance_every must be positive.")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (
        real_world_drift if real_world_drift is not None else r - q
    ) - 0.5 * sigma**2
    shocks = rng.standard_normal((n_paths, n_steps))
    log_returns = drift * dt + sigma * np.sqrt(dt) * shocks
    spots = np.empty((n_paths, n_steps + 1), dtype=float)
    spots[:, 0] = S0
    spots[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))

    option_sign = -1.0 if position == "short" else 1.0
    option_premium = float(price(S0, K, T, r, q, sigma, option_type))
    option_delta = delta(S0, K, T, r, q, sigma, option_type)
    stock_shares = np.full(n_paths, -option_sign * option_delta)
    cash = np.full(n_paths, -option_sign * option_premium) - stock_shares * S0
    total_costs = np.abs(stock_shares) * S0 * transaction_cost_bps / 10_000.0
    cash -= total_costs

    detail_rows = []
    for step in range(n_steps + 1):
        remaining_T = max(T - step * dt, 0.0)
        spot = spots[:, step]
        if step > 0:
            cash *= np.exp(r * dt)
            if q != 0.0:
                cash += stock_shares * spot * q * dt

        if step < n_steps and step > 0 and step % rebalance_every == 0:
            new_shares = -option_sign * _vectorized_delta(
                spot,
                K,
                max(remaining_T, 1e-8),
                r,
                q,
                sigma,
                option_type,
            )
            trade = new_shares - stock_shares
            costs = np.abs(trade) * spot * transaction_cost_bps / 10_000.0
            cash -= trade * spot + costs
            total_costs += costs
            stock_shares = new_shares
        elif step == n_steps:
            pass

        if step == 0 or step % max(1, n_steps // 20) == 0 or step == n_steps:
            if remaining_T > 0:
                option_value = float(
                    price(float(spot[0]), K, remaining_T, r, q, sigma, option_type)
                )
                option_delta_for_detail = float(
                    delta(float(spot[0]), K, remaining_T, r, q, sigma, option_type)
                )
            else:
                option_value = (
                    max(float(spot[0]) - K, 0.0)
                    if option_type == "call"
                    else max(K - float(spot[0]), 0.0)
                )
                option_delta_for_detail = (
                    1.0
                    if (option_type == "call" and spot[0] > K)
                    else -1.0 if (option_type == "put" and spot[0] < K) else 0.0
                )
            portfolio_value = (
                cash[0] + stock_shares[0] * spot[0] + option_sign * option_value
            )
            detail_rows.append(
                {
                    "step": step,
                    "time": step * dt,
                    "spot": float(spot[0]),
                    "option_value": option_value,
                    "option_delta": option_delta_for_detail,
                    "stock_shares": float(stock_shares[0]),
                    "cash": float(cash[0]),
                    "portfolio_value": float(portfolio_value),
                }
            )

    terminal_spot = spots[:, -1]
    payoff = (
        np.maximum(terminal_spot - K, 0.0)
        if option_type == "call"
        else np.maximum(K - terminal_spot, 0.0)
    )
    cash += stock_shares * terminal_spot
    cash += option_sign * payoff
    pnl = cash

    summary = {
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl, ddof=1)),
        "p05": float(np.percentile(pnl, 5)),
        "p50": float(np.percentile(pnl, 50)),
        "p95": float(np.percentile(pnl, 95)),
        "mean_transaction_costs": float(np.mean(total_costs)),
    }
    return HedgingResult(
        pnl=pnl, summary=summary, path_details=pd.DataFrame(detail_rows)
    )


def compare_rebalance_frequencies(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
    position: Position = "short",
    frequencies_per_year: list[int] | tuple[int, ...] = (12, 52, 252),
    n_paths: int = 5_000,
    n_steps: int = TRADING_DAYS_PER_YEAR,
    transaction_cost_bps: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare hedging error across rebalance frequencies."""

    rows = []
    for frequency in frequencies_per_year:
        rebalance_every = max(1, round(n_steps / frequency))
        result = simulate_delta_hedging(
            S0,
            K,
            T,
            r,
            q,
            sigma,
            option_type,
            position,
            n_paths,
            n_steps,
            rebalance_every,
            transaction_cost_bps,
            seed,
        )
        rows.append(
            {
                "rebalances_per_year": frequency,
                "rebalance_every_steps": rebalance_every,
                **result.summary,
            }
        )
    return pd.DataFrame(rows)

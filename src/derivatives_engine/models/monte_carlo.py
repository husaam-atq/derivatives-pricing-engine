"""Monte Carlo pricing under geometric Brownian motion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

from derivatives_engine.config import DEFAULT_CONFIDENCE_LEVEL, HAS_CUPY
from derivatives_engine.models.black_scholes import price as black_scholes_price

OptionType = Literal["call", "put"]
BarrierType = Literal["up-and-out", "down-and-out"]


@dataclass(frozen=True)
class MonteCarloResult:
    """Monte Carlo pricing result."""

    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    paths: int
    time_steps: int
    method: str
    metadata: dict[str, float | str | bool] = field(default_factory=dict)


def _validate_mc_inputs(
    S0: float,
    K: float,
    T: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    option_type: str = "call",
) -> None:
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2.")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")


def _confidence_interval(
    estimate: float, standard_error: float, confidence_level: float
) -> tuple[float, float]:
    z_score = norm.ppf(0.5 + confidence_level / 2.0)
    return estimate - z_score * standard_error, estimate + z_score * standard_error


def gpu_available() -> bool:
    """Return True if CuPy is importable and reports at least one CUDA device."""

    if not HAS_CUPY:
        return False
    try:
        import cupy as cp  # type: ignore[import-not-found]

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def simulate_gbm_paths(
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    antithetic: bool = False,
    use_gpu: bool = False,
) -> np.ndarray:
    """Simulate risk-neutral GBM paths using exact lognormal steps.

    If ``use_gpu`` is requested but CuPy/CUDA is unavailable, the function
    silently falls back to NumPy and still returns a NumPy array.
    """

    _validate_mc_inputs(S0, S0, T, sigma, n_paths, n_steps)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    if use_gpu and gpu_available():
        import cupy as cp  # type: ignore[import-not-found]

        rng = cp.random.default_rng(seed)
        if antithetic:
            half_paths = (n_paths + 1) // 2
            z_half = rng.standard_normal((half_paths, n_steps))
            z = cp.concatenate([z_half, -z_half], axis=0)[:n_paths]
        else:
            z = rng.standard_normal((n_paths, n_steps))
        log_returns = drift + diffusion * z
        log_paths = cp.cumsum(log_returns, axis=1)
        paths = cp.empty((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = S0
        paths[:, 1:] = S0 * cp.exp(log_paths)
        return cp.asnumpy(paths)

    rng = np.random.default_rng(seed)
    if antithetic:
        half_paths = (n_paths + 1) // 2
        z_half = rng.standard_normal((half_paths, n_steps))
        z = np.concatenate([z_half, -z_half], axis=0)[:n_paths]
    else:
        z = rng.standard_normal((n_paths, n_steps))

    log_returns = drift + diffusion * z
    log_paths = np.cumsum(log_returns, axis=1)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths


def _terminal_payoff(ST: np.ndarray, K: float, option_type: OptionType) -> np.ndarray:
    if option_type == "call":
        return np.maximum(ST - K, 0.0)
    return np.maximum(K - ST, 0.0)


def price_european_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    seed: int | None = None,
    antithetic: bool = False,
    control_variate: bool = False,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> MonteCarloResult:
    """Price a European vanilla option by Monte Carlo.

    Control variate uses the discounted terminal stock price with known
    expectation ``S0 * exp(-qT)``.
    """

    _validate_mc_inputs(S0, K, T, sigma, n_paths, 1, option_type)
    discount = np.exp(-r * T)
    rng = np.random.default_rng(seed)

    if antithetic and not control_variate:
        n_pairs = (n_paths + 1) // 2
        z = rng.standard_normal(n_pairs)
        ST_plus = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        ST_minus = S0 * np.exp((r - q - 0.5 * sigma**2) * T - sigma * np.sqrt(T) * z)
        pair_payoffs = 0.5 * (
            discount * _terminal_payoff(ST_plus, K, option_type)
            + discount * _terminal_payoff(ST_minus, K, option_type)
        )
        observations = pair_payoffs
        effective_paths = 2 * n_pairs
        method = "antithetic"
    else:
        z = rng.standard_normal(n_paths)
        ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        discounted_payoffs = discount * _terminal_payoff(ST, K, option_type)
        observations = discounted_payoffs
        effective_paths = n_paths
        method = "plain"
        if control_variate:
            discounted_stock = discount * ST
            known_mean = S0 * np.exp(-q * T)
            covariance = np.cov(discounted_payoffs, discounted_stock, ddof=1)[0, 1]
            variance = np.var(discounted_stock, ddof=1)
            beta = covariance / variance if variance > 0 else 0.0
            observations = discounted_payoffs - beta * (discounted_stock - known_mean)
            method = "control_variate_stock"

    estimate = float(np.mean(observations))
    standard_error = float(np.std(observations, ddof=1) / np.sqrt(len(observations)))
    ci = _confidence_interval(estimate, standard_error, confidence_level)
    return MonteCarloResult(
        price=estimate,
        standard_error=standard_error,
        confidence_interval=(float(ci[0]), float(ci[1])),
        confidence_level=confidence_level,
        paths=effective_paths,
        time_steps=1,
        method=method,
        metadata={
            "antithetic": antithetic,
            "control_variate": control_variate,
            "analytical_bsm_price": float(
                black_scholes_price(S0, K, T, r, q, sigma, option_type)
            ),
        },
    )


def price_asian_arithmetic_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = None,
    antithetic: bool = False,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> MonteCarloResult:
    """Price an arithmetic-average Asian option by Monte Carlo."""

    paths = simulate_gbm_paths(S0, T, r, q, sigma, n_paths, n_steps, seed, antithetic)
    averages = paths[:, 1:].mean(axis=1)
    discounted_payoffs = np.exp(-r * T) * _terminal_payoff(averages, K, option_type)
    estimate = float(np.mean(discounted_payoffs))
    standard_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths))
    ci = _confidence_interval(estimate, standard_error, confidence_level)
    return MonteCarloResult(
        estimate,
        standard_error,
        (float(ci[0]), float(ci[1])),
        confidence_level,
        n_paths,
        n_steps,
        "asian_arithmetic_antithetic" if antithetic else "asian_arithmetic",
    )


def price_barrier_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
    barrier: float = 130.0,
    barrier_type: BarrierType = "up-and-out",
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = None,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> MonteCarloResult:
    """Price a simple knock-out barrier option by Monte Carlo."""

    if barrier_type not in {"up-and-out", "down-and-out"}:
        raise ValueError("barrier_type must be 'up-and-out' or 'down-and-out'.")
    paths = simulate_gbm_paths(S0, T, r, q, sigma, n_paths, n_steps, seed)
    if barrier_type == "up-and-out":
        alive = paths.max(axis=1) < barrier
    else:
        alive = paths.min(axis=1) > barrier
    terminal_payoffs = _terminal_payoff(paths[:, -1], K, option_type)
    discounted_payoffs = np.exp(-r * T) * terminal_payoffs * alive
    estimate = float(np.mean(discounted_payoffs))
    standard_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths))
    ci = _confidence_interval(estimate, standard_error, confidence_level)
    return MonteCarloResult(
        estimate,
        standard_error,
        (float(ci[0]), float(ci[1])),
        confidence_level,
        n_paths,
        n_steps,
        barrier_type,
        {"barrier": barrier},
    )


def compare_variance_reduction(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a comparison table for plain, antithetic and control variate MC."""

    plain = price_european_option(S0, K, T, r, q, sigma, option_type, n_paths, seed)
    anti = price_european_option(
        S0, K, T, r, q, sigma, option_type, n_paths, seed, antithetic=True
    )
    control = price_european_option(
        S0, K, T, r, q, sigma, option_type, n_paths, seed, control_variate=True
    )
    rows = []
    for result in [plain, anti, control]:
        rows.append(
            {
                "method": result.method,
                "price": result.price,
                "standard_error": result.standard_error,
                "ci_lower": result.confidence_interval[0],
                "ci_upper": result.confidence_interval[1],
                "standard_error_reduction_pct": (
                    0.0
                    if result.method == "plain"
                    else (1.0 - result.standard_error / plain.standard_error) * 100.0
                ),
            }
        )
    return pd.DataFrame(rows)

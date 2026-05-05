"""Heston stochastic volatility model simulation and pricing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from derivatives_engine.config import DEFAULT_CONFIDENCE_LEVEL

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class HestonParams:
    """Heston model parameters.

    Attributes:
        v0: Initial variance.
        kappa: Mean reversion speed of variance.
        theta: Long-run variance level.
        sigma_v: Volatility of variance.
        rho: Correlation between spot and variance Brownian shocks.
    """

    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    sigma_v: float = 0.35
    rho: float = -0.6

    def validate(self) -> None:
        if self.v0 < 0:
            raise ValueError("v0 must be non-negative.")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        if self.theta <= 0:
            raise ValueError("theta must be positive.")
        if self.sigma_v <= 0:
            raise ValueError("sigma_v must be positive.")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must be strictly between -1 and 1.")


@dataclass(frozen=True)
class HestonSimulationResult:
    """Heston path simulation output."""

    spot_paths: np.ndarray
    variance_paths: np.ndarray
    time_grid: np.ndarray


@dataclass(frozen=True)
class HestonPricingResult:
    """Heston Monte Carlo pricing output."""

    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    paths: int
    time_steps: int


def simulate_heston_paths(
    S0: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    n_paths: int = 50_000,
    n_steps: int = 252,
    seed: int | None = None,
) -> HestonSimulationResult:
    """Simulate Heston spot and variance paths using full truncation Euler."""

    params.validate()
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_paths < 1 or n_steps < 1:
        raise ValueError("n_paths and n_steps must be positive.")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    spots = np.empty((n_paths, n_steps + 1), dtype=float)
    variances = np.empty_like(spots)
    spots[:, 0] = S0
    variances[:, 0] = params.v0

    rho_complement = np.sqrt(1.0 - params.rho**2)
    for step in range(n_steps):
        z_spot = rng.standard_normal(n_paths)
        z_independent = rng.standard_normal(n_paths)
        dW_spot = sqrt_dt * z_spot
        dW_var = sqrt_dt * (params.rho * z_spot + rho_complement * z_independent)

        variance_positive = np.maximum(variances[:, step], 0.0)
        variances[:, step + 1] = (
            variances[:, step]
            + params.kappa * (params.theta - variance_positive) * dt
            + params.sigma_v * np.sqrt(variance_positive) * dW_var
        )
        variances[:, step + 1] = np.maximum(variances[:, step + 1], 0.0)
        spots[:, step + 1] = spots[:, step] * np.exp(
            (r - q - 0.5 * variance_positive) * dt
            + np.sqrt(variance_positive) * dW_spot
        )

    return HestonSimulationResult(
        spot_paths=spots,
        variance_paths=variances,
        time_grid=np.linspace(0.0, T, n_steps + 1),
    )


def price_heston_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = None,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> HestonPricingResult:
    """Price a European option with Heston Monte Carlo simulation."""

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")
    simulation = simulate_heston_paths(S0, T, r, q, params, n_paths, n_steps, seed)
    terminal = simulation.spot_paths[:, -1]
    if option_type == "call":
        payoffs = np.maximum(terminal - K, 0.0)
    else:
        payoffs = np.maximum(K - terminal, 0.0)
    discounted = np.exp(-r * T) * payoffs
    estimate = float(np.mean(discounted))
    standard_error = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
    z_score = norm.ppf(0.5 + confidence_level / 2.0)
    ci = estimate - z_score * standard_error, estimate + z_score * standard_error
    return HestonPricingResult(
        estimate,
        standard_error,
        (float(ci[0]), float(ci[1])),
        confidence_level,
        n_paths,
        n_steps,
    )


def heston_characteristic_function(
    u: complex,
    S0: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
) -> complex:
    """Return the Heston characteristic function of log(S_T)."""

    params.validate()
    i = 1j
    kappa = params.kappa
    theta = params.theta
    sigma_v = params.sigma_v
    rho = params.rho
    v0 = params.v0

    d = np.sqrt((rho * sigma_v * i * u - kappa) ** 2 + sigma_v**2 * (i * u + u**2))
    g = (kappa - rho * sigma_v * i * u - d) / (kappa - rho * sigma_v * i * u + d)
    exp_neg_dT = np.exp(-d * T)
    C = i * u * (np.log(S0) + (r - q) * T) + (kappa * theta / sigma_v**2) * (
        (kappa - rho * sigma_v * i * u - d) * T
        - 2.0 * np.log((1.0 - g * exp_neg_dT) / (1.0 - g))
    )
    D = (
        (kappa - rho * sigma_v * i * u - d)
        / sigma_v**2
        * ((1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT))
    )
    return complex(np.exp(C + D * v0))


def _heston_probability(
    j: int,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    integration_limit: float,
) -> float:
    log_strike = np.log(K)
    phi_minus_i = heston_characteristic_function(-1j, S0, T, r, q, params)

    def integrand(u: float) -> float:
        if u == 0.0:
            return 0.0
        if j == 1:
            numerator = heston_characteristic_function(u - 1j, S0, T, r, q, params)
            characteristic = numerator / phi_minus_i
        else:
            characteristic = heston_characteristic_function(u, S0, T, r, q, params)
        value = np.exp(-1j * u * log_strike) * characteristic / (1j * u)
        return float(np.real(value))

    integral, _ = quad(
        integrand,
        1e-8,
        integration_limit,
        epsabs=1e-7,
        epsrel=1e-7,
        limit=150,
    )
    return float(0.5 + integral / np.pi)


def heston_call_price_cf(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    integration_limit: float = 75.0,
) -> float:
    """Price a European call using the Heston characteristic function."""

    params.validate()
    if S0 <= 0 or K <= 0 or T <= 0:
        raise ValueError("S0, K and T must be positive.")
    p1 = _heston_probability(1, S0, K, T, r, q, params, integration_limit)
    p2 = _heston_probability(2, S0, K, T, r, q, params, integration_limit)
    return float(S0 * np.exp(-q * T) * p1 - K * np.exp(-r * T) * p2)


def heston_price_cf(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    option_type: OptionType = "call",
    integration_limit: float = 75.0,
) -> float:
    """Price a European call or put using the Heston characteristic function."""

    call = heston_call_price_cf(S0, K, T, r, q, params, integration_limit)
    if option_type == "call":
        return call
    if option_type == "put":
        return float(call - S0 * np.exp(-q * T) + K * np.exp(-r * T))
    raise ValueError("option_type must be either 'call' or 'put'.")

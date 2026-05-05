"""Black-Scholes-Merton analytical pricing functions."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

OptionType = Literal["call", "put"]


def _validate_option_type(option_type: str) -> None:
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")


def _return_scalar_if_scalar(
    value: np.ndarray, *inputs: ArrayLike
) -> float | np.ndarray:
    if all(np.isscalar(item) for item in inputs):
        return float(np.asarray(value))
    return value


def _broadcast_inputs(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = np.broadcast_arrays(
        np.asarray(S, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(r, dtype=float),
        np.asarray(q, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    S_arr, K_arr, T_arr, r_arr, q_arr, sigma_arr = arrays
    if np.any(S_arr <= 0):
        raise ValueError("Spot price S must be positive.")
    if np.any(K_arr <= 0):
        raise ValueError("Strike K must be positive.")
    if np.any(T_arr < 0):
        raise ValueError("Time to maturity T must be non-negative.")
    if np.any(sigma_arr < 0):
        raise ValueError("Volatility sigma must be non-negative.")
    return arrays


def d1(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    sigma: ArrayLike = 0.2,
) -> float | np.ndarray:
    """Return Black-Scholes-Merton d1.

    Parameters use annualised continuous compounding. ``q`` is a continuous
    dividend yield.
    """

    S_arr, K_arr, T_arr, r_arr, q_arr, sigma_arr = _broadcast_inputs(
        S, K, T, r, q, sigma
    )
    if np.any(T_arr <= 0) or np.any(sigma_arr <= 0):
        raise ValueError("d1 is undefined for T <= 0 or sigma <= 0.")
    value = (np.log(S_arr / K_arr) + (r_arr - q_arr + 0.5 * sigma_arr**2) * T_arr) / (
        sigma_arr * np.sqrt(T_arr)
    )
    return _return_scalar_if_scalar(value, S, K, T, r, q, sigma)


def d2(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    sigma: ArrayLike = 0.2,
) -> float | np.ndarray:
    """Return Black-Scholes-Merton d2."""

    S_arr, K_arr, T_arr, r_arr, q_arr, sigma_arr = _broadcast_inputs(
        S, K, T, r, q, sigma
    )
    value = np.asarray(d1(S_arr, K_arr, T_arr, r_arr, q_arr, sigma_arr)) - (
        sigma_arr * np.sqrt(T_arr)
    )
    return _return_scalar_if_scalar(value, S, K, T, r, q, sigma)


def _zero_vol_price(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    option_type: OptionType,
) -> np.ndarray:
    forward_intrinsic = S * np.exp(-q * T) - K * np.exp(-r * T)
    if option_type == "call":
        return np.maximum(forward_intrinsic, 0.0)
    return np.maximum(-forward_intrinsic, 0.0)


def price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    sigma: ArrayLike = 0.2,
    option_type: OptionType = "call",
) -> float | np.ndarray:
    """Price a European option using the Black-Scholes-Merton formula.

    ``T=0`` returns intrinsic value. ``sigma=0`` returns the deterministic
    discounted payoff under the risk-neutral forward.
    """

    _validate_option_type(option_type)
    S_arr, K_arr, T_arr, r_arr, q_arr, sigma_arr = _broadcast_inputs(
        S, K, T, r, q, sigma
    )
    values = np.empty_like(S_arr, dtype=float)

    expiry = T_arr == 0
    deterministic = (~expiry) & (sigma_arr == 0)
    analytical = (~expiry) & (sigma_arr > 0)

    if np.any(expiry):
        if option_type == "call":
            values[expiry] = np.maximum(S_arr[expiry] - K_arr[expiry], 0.0)
        else:
            values[expiry] = np.maximum(K_arr[expiry] - S_arr[expiry], 0.0)

    if np.any(deterministic):
        values[deterministic] = _zero_vol_price(
            S_arr[deterministic],
            K_arr[deterministic],
            T_arr[deterministic],
            r_arr[deterministic],
            q_arr[deterministic],
            option_type,
        )

    if np.any(analytical):
        d1_val = np.asarray(
            d1(
                S_arr[analytical],
                K_arr[analytical],
                T_arr[analytical],
                r_arr[analytical],
                q_arr[analytical],
                sigma_arr[analytical],
            )
        )
        d2_val = d1_val - sigma_arr[analytical] * np.sqrt(T_arr[analytical])
        discounted_spot = S_arr[analytical] * np.exp(
            -q_arr[analytical] * T_arr[analytical]
        )
        discounted_strike = K_arr[analytical] * np.exp(
            -r_arr[analytical] * T_arr[analytical]
        )
        if option_type == "call":
            values[analytical] = discounted_spot * norm.cdf(
                d1_val
            ) - discounted_strike * norm.cdf(d2_val)
        else:
            values[analytical] = discounted_strike * norm.cdf(
                -d2_val
            ) - discounted_spot * norm.cdf(-d1_val)

    return _return_scalar_if_scalar(values, S, K, T, r, q, sigma)


def call_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    sigma: ArrayLike = 0.2,
) -> float | np.ndarray:
    """Return the Black-Scholes-Merton European call price."""

    return price(S, K, T, r, q, sigma, option_type="call")


def put_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    sigma: ArrayLike = 0.2,
) -> float | np.ndarray:
    """Return the Black-Scholes-Merton European put price."""

    return price(S, K, T, r, q, sigma, option_type="put")


def put_call_parity_error(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    sigma: ArrayLike = 0.2,
) -> float | np.ndarray:
    """Return call - put - discounted forward parity value.

    For internally generated Black-Scholes prices this should be close to zero.
    """

    S_arr, K_arr, T_arr, r_arr, q_arr, sigma_arr = _broadcast_inputs(
        S, K, T, r, q, sigma
    )
    error = call_price(S_arr, K_arr, T_arr, r_arr, q_arr, sigma_arr) - put_price(
        S_arr, K_arr, T_arr, r_arr, q_arr, sigma_arr
    )
    error = error - (S_arr * np.exp(-q_arr * T_arr) - K_arr * np.exp(-r_arr * T_arr))
    return _return_scalar_if_scalar(np.asarray(error), S, K, T, r, q, sigma)

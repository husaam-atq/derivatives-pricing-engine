"""Cox-Ross-Rubinstein binomial tree pricing."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from derivatives_engine.models.black_scholes import price as black_scholes_price

OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]


def _validate_inputs(
    S: float,
    K: float,
    T: float,
    sigma: float,
    steps: int,
    option_type: str,
    exercise_style: str,
) -> None:
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if exercise_style not in {"european", "american"}:
        raise ValueError("exercise_style must be 'european' or 'american'.")


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
    exercise_style: ExerciseStyle = "european",
    steps: int = 500,
) -> float:
    """Price an option with a Cox-Ross-Rubinstein recombining tree."""

    _validate_inputs(S, K, T, sigma, steps, option_type, exercise_style)
    if T == 0:
        return float(max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0))
    if sigma == 0:
        return float(black_scholes_price(S, K, T, r, q, 0.0, option_type))

    dt = T / steps
    sqrt_dt = np.sqrt(dt)
    up = np.exp(sigma * sqrt_dt)
    down = 1.0 / up
    growth = np.exp((r - q) * dt)
    risk_neutral_prob = (growth - down) / (up - down)
    if not 0.0 <= risk_neutral_prob <= 1.0:
        raise ValueError(
            "Risk-neutral probability outside [0, 1]. Increase steps or check inputs."
        )

    discount = np.exp(-r * dt)
    j = np.arange(steps + 1)
    terminal_spots = S * (up**j) * (down ** (steps - j))
    if option_type == "call":
        values = np.maximum(terminal_spots - K, 0.0)
    else:
        values = np.maximum(K - terminal_spots, 0.0)

    for step in range(steps - 1, -1, -1):
        values = discount * (
            risk_neutral_prob * values[1 : step + 2]
            + (1.0 - risk_neutral_prob) * values[0 : step + 1]
        )
        if exercise_style == "american":
            j = np.arange(step + 1)
            spots = S * (up**j) * (down ** (step - j))
            intrinsic = (
                np.maximum(spots - K, 0.0)
                if option_type == "call"
                else np.maximum(K - spots, 0.0)
            )
            values = np.maximum(values, intrinsic)

    return float(values[0])


def convergence_table(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    sigma: float = 0.2,
    option_type: OptionType = "call",
    steps_list: list[int] | tuple[int, ...] = (10, 25, 50, 100, 250, 500, 1000),
) -> pd.DataFrame:
    """Return CRR prices and errors against Black-Scholes for increasing steps."""

    bs_price = float(black_scholes_price(S, K, T, r, q, sigma, option_type))
    rows = []
    for steps in steps_list:
        tree_price = crr_price(S, K, T, r, q, sigma, option_type, "european", steps)
        rows.append(
            {
                "steps": steps,
                "binomial_price": tree_price,
                "black_scholes_price": bs_price,
                "absolute_error": abs(tree_price - bs_price),
            }
        )
    return pd.DataFrame(rows)

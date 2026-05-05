"""Tests for Heston model functionality."""

from __future__ import annotations

import numpy as np

from derivatives_engine.models.heston import (
    HestonParams,
    price_heston_mc,
    simulate_heston_paths,
)


def test_heston_simulation_shapes_and_non_negative_variance() -> None:
    params = HestonParams()
    result = simulate_heston_paths(
        100, 1, 0.03, 0.0, params, n_paths=500, n_steps=50, seed=1
    )
    assert result.spot_paths.shape == (500, 51)
    assert result.variance_paths.shape == (500, 51)
    assert np.all(result.variance_paths >= 0)


def test_heston_mc_price_is_valid() -> None:
    result = price_heston_mc(
        100, 100, 1, 0.03, 0.0, HestonParams(), "call", 5_000, 64, 2
    )
    assert np.isfinite(result.price)
    assert result.price > 0
    assert result.standard_error > 0

"""Tests for Black-Scholes-Merton pricing."""

from __future__ import annotations

import numpy as np
import pytest

from derivatives_engine.models.black_scholes import (
    call_price,
    price,
    put_call_parity_error,
    put_price,
)


def test_textbook_benchmark_values() -> None:
    assert call_price(100, 100, 1, 0.05, 0.0, 0.20) == pytest.approx(10.4506, abs=1e-4)
    assert put_price(100, 100, 1, 0.05, 0.0, 0.20) == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity() -> None:
    error = put_call_parity_error(100, 100, 1, 0.05, 0.0, 0.20)
    assert abs(error) < 1e-8


def test_vectorized_pricing() -> None:
    strikes = np.array([90, 100, 110])
    calls = call_price(100, strikes, 1, 0.05, 0.0, 0.20)
    assert calls.shape == strikes.shape
    assert np.all(np.diff(calls) < 0)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        price(-100, 100, 1, 0.05, 0.0, 0.20, "call")
    with pytest.raises(ValueError):
        price(100, 100, -1, 0.05, 0.0, 0.20, "call")
    with pytest.raises(ValueError):
        price(100, 100, 1, 0.05, 0.0, 0.20, "straddle")  # type: ignore[arg-type]

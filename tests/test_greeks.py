"""Tests for analytical and finite-difference Greeks."""

from __future__ import annotations

import pytest

from derivatives_engine.risk.greeks import (
    delta,
    finite_difference_delta,
    finite_difference_gamma,
    finite_difference_rho,
    finite_difference_theta,
    finite_difference_vega,
    gamma,
    rho,
    theta,
    vega,
)


def test_greek_sign_sanity() -> None:
    S, K, T, r, q, sigma = 100, 100, 1, 0.05, 0.0, 0.20
    assert 0 < delta(S, K, T, r, q, sigma, "call") < 1
    assert -1 < delta(S, K, T, r, q, sigma, "put") < 0
    assert gamma(S, K, T, r, q, sigma) > 0
    assert vega(S, K, T, r, q, sigma) > 0
    assert rho(S, K, T, r, q, sigma, "call") > 0
    assert rho(S, K, T, r, q, sigma, "put") < 0


def test_analytical_vs_finite_difference_greeks() -> None:
    S, K, T, r, q, sigma = 100, 100, 1, 0.05, 0.0, 0.20
    assert delta(S, K, T, r, q, sigma, "call") == pytest.approx(
        finite_difference_delta(S, K, T, r, q, sigma, "call"), abs=1e-4
    )
    assert gamma(S, K, T, r, q, sigma) == pytest.approx(
        finite_difference_gamma(S, K, T, r, q, sigma, "call"), abs=1e-4
    )
    assert vega(S, K, T, r, q, sigma) == pytest.approx(
        finite_difference_vega(S, K, T, r, q, sigma, "call"), rel=1e-3
    )
    assert rho(S, K, T, r, q, sigma, "call") == pytest.approx(
        finite_difference_rho(S, K, T, r, q, sigma, "call"), rel=1e-3
    )
    assert theta(S, K, T, r, q, sigma, "call") == pytest.approx(
        finite_difference_theta(S, K, T, r, q, sigma, "call"), abs=2e-3
    )

"""Tests for numerical convergence reporting utilities."""

from __future__ import annotations

from derivatives_engine.utils.convergence import (
    binomial_convergence_results,
    generate_numerical_convergence_report,
    greek_comparison_results,
    monte_carlo_convergence_results,
    variance_reduction_results,
)


def test_convergence_helpers_return_expected_columns() -> None:
    binomial = binomial_convergence_results(steps=(25, 50))
    assert {
        "steps",
        "binomial_price",
        "black_scholes_price",
        "absolute_error",
    }.issubset(binomial.columns)
    assert len(binomial) == 2

    monte_carlo = monte_carlo_convergence_results(path_counts=(1_000, 2_000), seed=10)
    assert {"paths", "mc_price", "standard_error", "analytical_inside_ci"}.issubset(
        monte_carlo.columns
    )
    assert len(monte_carlo) == 2

    variance = variance_reduction_results(n_paths=5_000, seed=11)
    assert {"method", "standard_error", "standard_error_reduction_pct"}.issubset(
        variance.columns
    )

    greeks = greek_comparison_results()
    assert {"greek", "analytical", "finite_difference", "absolute_error"}.issubset(
        greeks.columns
    )


def test_numerical_convergence_report_generation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    results, csv_path, report_path = generate_numerical_convergence_report(tmp_path)
    assert not results.empty
    assert csv_path.exists()
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Numerical Convergence Report" in report
    assert "Binomial Tree Convergence" in report
    assert "Monte Carlo Convergence" in report

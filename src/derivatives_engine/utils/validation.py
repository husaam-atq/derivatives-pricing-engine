"""Validation benchmarks and report generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from derivatives_engine.calibration.heston_calibration import (
    calibrate_heston,
    generate_synthetic_heston_chain,
    parameter_comparison_table,
)
from derivatives_engine.models.binomial_tree import convergence_table
from derivatives_engine.models.black_scholes import (
    call_price,
    price,
    put_call_parity_error,
    put_price,
)
from derivatives_engine.models.heston import (
    HestonParams,
    price_heston_mc,
    simulate_heston_paths,
)
from derivatives_engine.models.monte_carlo import (
    compare_variance_reduction,
    price_european_option,
)
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
from derivatives_engine.risk.hedging import compare_rebalance_frequencies
from derivatives_engine.risk.implied_volatility import implied_volatility


def _row(
    category: str,
    benchmark: str,
    metric: str,
    value: float | str,
    target: str,
    passed: bool,
    notes: str = "",
) -> dict[str, float | str | bool]:
    return {
        "category": category,
        "benchmark": benchmark,
        "metric": metric,
        "value": value,
        "target": target,
        "passed": bool(passed),
        "notes": notes,
    }


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render a small DataFrame as a GitHub-flavoured Markdown table."""

    display = df if max_rows is None else df.head(max_rows)
    columns = list(display.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for _, row in display.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.8g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_validation_benchmarks() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run all benchmark checks and return summary rows plus detail tables."""

    rows: list[dict[str, float | str | bool]] = []
    details: dict[str, pd.DataFrame] = {}
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.0
    sigma = 0.20

    call = float(call_price(S, K, T, r, q, sigma))
    put = float(put_price(S, K, T, r, q, sigma))
    rows.append(
        _row(
            "Black-Scholes",
            "Textbook call benchmark",
            "absolute_error",
            abs(call - 10.4506),
            "< 1e-4",
            abs(call - 10.4506) < 1e-4,
            f"price={call:.8f}",
        )
    )
    rows.append(
        _row(
            "Black-Scholes",
            "Textbook put benchmark",
            "absolute_error",
            abs(put - 5.5735),
            "< 1e-4",
            abs(put - 5.5735) < 1e-4,
            f"price={put:.8f}",
        )
    )
    parity_error = float(put_call_parity_error(S, K, T, r, q, sigma))
    rows.append(
        _row(
            "Black-Scholes",
            "Put-call parity",
            "absolute_error",
            abs(parity_error),
            "< 1e-8",
            abs(parity_error) < 1e-8,
        )
    )

    for known_sigma in [0.15, 0.20, 0.35]:
        market = float(price(S, K, T, r, q, known_sigma, "call"))
        brent = implied_volatility(market, S, K, T, r, q, "call", "brent")
        newton = implied_volatility(market, S, K, T, r, q, "call", "newton", 0.25)
        rows.append(
            _row(
                "Implied volatility",
                f"Brent recovers sigma={known_sigma:.2f}",
                "absolute_error",
                abs(brent.implied_volatility - known_sigma),
                "< 1e-6",
                abs(brent.implied_volatility - known_sigma) < 1e-6,
            )
        )
        rows.append(
            _row(
                "Implied volatility",
                f"Newton recovers sigma={known_sigma:.2f}",
                "absolute_error",
                abs(newton.implied_volatility - known_sigma),
                "< 1e-5",
                abs(newton.implied_volatility - known_sigma) < 1e-5,
            )
        )

    greek_rows = []
    analytical = {
        "delta": delta(S, K, T, r, q, sigma, "call"),
        "gamma": gamma(S, K, T, r, q, sigma),
        "vega": vega(S, K, T, r, q, sigma),
        "theta": theta(S, K, T, r, q, sigma, "call"),
        "rho": rho(S, K, T, r, q, sigma, "call"),
    }
    numerical = {
        "delta": finite_difference_delta(S, K, T, r, q, sigma, "call"),
        "gamma": finite_difference_gamma(S, K, T, r, q, sigma, "call"),
        "vega": finite_difference_vega(S, K, T, r, q, sigma, "call"),
        "theta": finite_difference_theta(S, K, T, r, q, sigma, "call"),
        "rho": finite_difference_rho(S, K, T, r, q, sigma, "call"),
    }
    tolerances = {
        "delta": 1e-4,
        "gamma": 1e-4,
        "vega": 1e-3,
        "theta": 2e-3,
        "rho": 1e-3,
    }
    for name, analytical_value in analytical.items():
        numerical_value = numerical[name]
        abs_error = abs(analytical_value - numerical_value)
        rel_error = abs_error / max(abs(analytical_value), 1e-12)
        metric_value = rel_error if name in {"vega", "rho"} else abs_error
        metric_name = "relative_error" if name in {"vega", "rho"} else "absolute_error"
        passed = metric_value < tolerances[name]
        rows.append(
            _row(
                "Greeks",
                f"Analytical vs finite-difference {name}",
                metric_name,
                metric_value,
                f"< {tolerances[name]}",
                passed,
                "Theta uses annualised calendar -dV/dT convention.",
            )
        )
        greek_rows.append(
            {
                "greek": name,
                "analytical": analytical_value,
                "finite_difference": numerical_value,
                "absolute_error": abs_error,
                "relative_error": rel_error,
            }
        )
    details["greeks"] = pd.DataFrame(greek_rows)

    for option_type in ["call", "put"]:
        table = convergence_table(S, K, T, r, q, sigma, option_type)
        details[f"binomial_{option_type}"] = table
        error_1000 = float(table.loc[table["steps"] == 1000, "absolute_error"].iloc[0])
        rows.append(
            _row(
                "Binomial tree",
                f"CRR {option_type} convergence at 1000 steps",
                "absolute_error",
                error_1000,
                "< 0.02",
                error_1000 < 0.02,
            )
        )

    mc_call = price_european_option(S, K, T, r, q, sigma, "call", 100_000, 123)
    mc_put = price_european_option(S, K, T, r, q, sigma, "put", 100_000, 456)
    for option_type, mc_result, analytical_price in [
        ("call", mc_call, call),
        ("put", mc_put, put),
    ]:
        error = abs(mc_result.price - analytical_price)
        inside_ci = (
            mc_result.confidence_interval[0]
            <= analytical_price
            <= mc_result.confidence_interval[1]
        )
        rows.append(
            _row(
                "Monte Carlo",
                f"European {option_type} vs Black-Scholes",
                "absolute_error",
                error,
                "< 0.15 or analytical inside 95% CI",
                error < 0.15 or inside_ci,
                (
                    f"price={mc_result.price:.6f}, analytical={analytical_price:.6f}, "
                    f"CI=({mc_result.confidence_interval[0]:.6f}, {mc_result.confidence_interval[1]:.6f})"
                ),
            )
        )
    variance_table = compare_variance_reduction(
        S, K, T, r, q, sigma, "call", 100_000, 789
    )
    details["variance_reduction"] = variance_table
    plain_se = float(
        variance_table.loc[variance_table["method"] == "plain", "standard_error"].iloc[
            0
        ]
    )
    for method in ["antithetic", "control_variate_stock"]:
        method_se = float(
            variance_table.loc[
                variance_table["method"] == method, "standard_error"
            ].iloc[0]
        )
        rows.append(
            _row(
                "Monte Carlo",
                f"{method} standard error reduction",
                "standard_error",
                method_se,
                f"< plain SE {plain_se:.6f}",
                method_se < plain_se,
            )
        )

    heston_params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.35, rho=-0.6)
    simulation = simulate_heston_paths(S, T, 0.03, 0.0, heston_params, 2_000, 64, 11)
    shape_ok = simulation.spot_paths.shape == (
        2_000,
        65,
    ) and simulation.variance_paths.shape == (2_000, 65)
    non_negative = bool(np.all(simulation.variance_paths >= 0.0))
    rows.append(
        _row(
            "Heston",
            "Simulation output shapes",
            "passed",
            str(shape_ok),
            "True",
            shape_ok,
        )
    )
    rows.append(
        _row(
            "Heston",
            "Full truncation non-negative variance",
            "passed",
            str(non_negative),
            "True",
            non_negative,
        )
    )
    heston_mc = price_heston_mc(
        S, K, T, 0.03, 0.0, heston_params, "call", 20_000, 128, 22
    )
    valid_price = np.isfinite(heston_mc.price) and heston_mc.price > 0
    rows.append(
        _row(
            "Heston",
            "Monte Carlo option price finite",
            "price",
            heston_mc.price,
            "finite and > 0",
            bool(valid_price),
            f"SE={heston_mc.standard_error:.6f}",
        )
    )

    true_params = HestonParams(v0=0.04, kappa=1.4, theta=0.04, sigma_v=0.35, rho=-0.55)
    chain = generate_synthetic_heston_chain(
        S,
        0.03,
        0.0,
        true_params,
        strikes=np.array([85.0, 100.0, 115.0]),
        maturities=np.array([0.5, 1.0, 1.5]),
    )
    calibration = calibrate_heston(
        chain,
        initial_params=HestonParams(
            v0=0.045, kappa=1.2, theta=0.045, sigma_v=0.40, rho=-0.45
        ),
        max_nfev=50,
    )
    details["heston_calibration_prices"] = calibration.fitted_prices
    details["heston_calibration_params"] = parameter_comparison_table(
        true_params, calibration.params
    )
    rows.append(
        _row(
            "Heston calibration",
            "Synthetic chain pricing fit",
            "RMSE",
            calibration.rmse,
            "< 0.05",
            calibration.rmse < 0.05,
            "Synthetic calibration is deterministic but can remain non-unique.",
        )
    )
    rows.append(
        _row(
            "Heston calibration",
            "Optimizer success flag",
            "success",
            str(calibration.success),
            "True",
            calibration.success,
            calibration.message,
        )
    )

    hedging = compare_rebalance_frequencies(
        S,
        K,
        T,
        r,
        q,
        sigma,
        "call",
        "short",
        frequencies_per_year=(12, 52, 252),
        n_paths=4_000,
        n_steps=252,
        transaction_cost_bps=0.0,
        seed=101,
    )
    details["hedging"] = hedging
    monthly_std = float(
        hedging.loc[hedging["rebalances_per_year"] == 12, "std_pnl"].iloc[0]
    )
    daily_std = float(
        hedging.loc[hedging["rebalances_per_year"] == 252, "std_pnl"].iloc[0]
    )
    rows.append(
        _row(
            "Delta hedging",
            "Daily vs monthly hedging error",
            "std_pnl",
            daily_std,
            f"< monthly std {monthly_std:.6f}",
            daily_std < monthly_std,
            "No transaction costs; same random seed by frequency.",
        )
    )

    return pd.DataFrame(rows), details


def generate_validation_report(
    output_dir: str | Path = "reports",
) -> tuple[pd.DataFrame, Path, Path]:
    """Run validation benchmarks and write CSV plus Markdown report."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results, details = run_validation_benchmarks()
    csv_path = output_path / "benchmark_results.csv"
    report_path = output_path / "validation_report.md"
    results.to_csv(csv_path, index=False)

    passed_count = int(results["passed"].sum())
    total_count = len(results)
    lines = [
        "# Validation Report",
        "",
        "This report is generated from live benchmark checks in `derivatives_engine.utils.validation`.",
        "It is intended to validate known analytical values, numerical convergence, stochastic error bars, and workflow-level sanity checks.",
        "",
        f"**Overall:** {passed_count}/{total_count} benchmark checks passed.",
        "",
        "## Benchmark Summary",
        "",
        _markdown_table(results),
        "",
        "## Greek Convention",
        "",
        "Vega is reported per 1.00 volatility change, with helper output available per one volatility point. Rho is per 1.00 rate change, with helper output available per basis point. Theta is annualised calendar theta using the `-dV/dT` convention; daily theta divides by calendar days.",
        "",
        "## Binomial Convergence",
        "",
        "European CRR prices converge toward Black-Scholes as the number of steps increases. The benchmark checks the 1000-step error target for calls and puts.",
        "",
        _markdown_table(details["binomial_call"]),
        "",
        "## Monte Carlo Variance Reduction",
        "",
        _markdown_table(details["variance_reduction"]),
        "",
        "## Heston Calibration",
        "",
        "Synthetic Heston calibration uses deterministic characteristic-function prices. The recovered parameters should be interpreted cautiously because Heston calibration can be non-unique and sensitive to strike/maturity coverage, objective scaling, and market data quality.",
        "",
        _markdown_table(details["heston_calibration_params"]),
        "",
        "## Delta Hedging",
        "",
        "The hedging benchmark compares rebalancing frequencies under Black-Scholes assumptions with zero transaction costs. More frequent rebalancing should generally reduce hedging error dispersion in this idealised setting.",
        "",
        _markdown_table(details["hedging"]),
        "",
        "## Limitations",
        "",
        "- Monte Carlo checks are deterministic by seed but still represent statistical estimators.",
        "- Heston Monte Carlo uses full truncation Euler; discretisation bias is possible.",
        "- Heston calibration is intentionally demonstrated on synthetic data and should not be treated as live-market evidence.",
        "- The validation suite checks credible benchmark behaviour; it is not a model approval document.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return results, csv_path, report_path

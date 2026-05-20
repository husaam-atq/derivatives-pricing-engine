"""Numerical convergence reporting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from derivatives_engine.models.binomial_tree import convergence_table
from derivatives_engine.models.black_scholes import price
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

OptionType = Literal["call", "put"]


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a compact GitHub-flavoured Markdown table."""

    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.8g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def binomial_convergence_results(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.20,
    option_type: OptionType = "call",
    steps: tuple[int, ...] = (25, 50, 100, 250, 500, 1000),
) -> pd.DataFrame:
    """Return CRR convergence results against Black-Scholes."""

    table = convergence_table(S, K, T, r, q, sigma, option_type, steps)
    table.insert(0, "section", "binomial_convergence")
    table.insert(1, "option_type", option_type)
    return table


def monte_carlo_convergence_results(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.20,
    option_type: OptionType = "call",
    path_counts: tuple[int, ...] = (1_000, 5_000, 10_000, 50_000, 100_000),
    seed: int = 2026,
) -> pd.DataFrame:
    """Return Monte Carlo convergence results across path counts."""

    analytical_price = float(price(S, K, T, r, q, sigma, option_type))
    rows = []
    for offset, n_paths in enumerate(path_counts):
        result = price_european_option(
            S,
            K,
            T,
            r,
            q,
            sigma,
            option_type,
            n_paths=n_paths,
            seed=seed + offset,
        )
        rows.append(
            {
                "section": "monte_carlo_convergence",
                "option_type": option_type,
                "paths": n_paths,
                "mc_price": result.price,
                "black_scholes_price": analytical_price,
                "absolute_error": abs(result.price - analytical_price),
                "standard_error": result.standard_error,
                "ci_lower": result.confidence_interval[0],
                "ci_upper": result.confidence_interval[1],
                "analytical_inside_ci": (
                    result.confidence_interval[0]
                    <= analytical_price
                    <= result.confidence_interval[1]
                ),
            }
        )
    return pd.DataFrame(rows)


def variance_reduction_results(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.20,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    seed: int = 789,
) -> pd.DataFrame:
    """Return plain, antithetic and control-variate standard error comparison."""

    table = compare_variance_reduction(
        S,
        K,
        T,
        r,
        q,
        sigma,
        option_type,
        n_paths=n_paths,
        seed=seed,
    )
    table.insert(0, "section", "variance_reduction")
    table.insert(1, "option_type", option_type)
    return table


def greek_comparison_results(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.20,
    option_type: OptionType = "call",
) -> pd.DataFrame:
    """Return analytical vs finite-difference Greek comparison."""

    analytical = {
        "delta": delta(S, K, T, r, q, sigma, option_type),
        "gamma": gamma(S, K, T, r, q, sigma),
        "vega": vega(S, K, T, r, q, sigma),
        "theta": theta(S, K, T, r, q, sigma, option_type),
        "rho": rho(S, K, T, r, q, sigma, option_type),
    }
    finite_difference = {
        "delta": finite_difference_delta(S, K, T, r, q, sigma, option_type),
        "gamma": finite_difference_gamma(S, K, T, r, q, sigma, option_type),
        "vega": finite_difference_vega(S, K, T, r, q, sigma, option_type),
        "theta": finite_difference_theta(S, K, T, r, q, sigma, option_type),
        "rho": finite_difference_rho(S, K, T, r, q, sigma, option_type),
    }
    rows = []
    for greek_name, analytical_value in analytical.items():
        numerical_value = finite_difference[greek_name]
        rows.append(
            {
                "section": "greek_comparison",
                "option_type": option_type,
                "greek": greek_name,
                "analytical": analytical_value,
                "finite_difference": numerical_value,
                "absolute_error": abs(analytical_value - numerical_value),
                "relative_error": abs(analytical_value - numerical_value)
                / max(abs(analytical_value), 1e-12),
            }
        )
    return pd.DataFrame(rows)


def generate_numerical_convergence_report(
    output_dir: str | Path = "reports",
) -> tuple[pd.DataFrame, Path, Path]:
    """Generate numerical convergence CSV and Markdown report."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    binomial = binomial_convergence_results()
    monte_carlo = monte_carlo_convergence_results()
    variance = variance_reduction_results()
    greeks = greek_comparison_results()

    combined = pd.concat(
        [
            binomial,
            monte_carlo,
            variance,
            greeks,
        ],
        ignore_index=True,
        sort=False,
    )

    csv_path = output_path / "numerical_convergence_results.csv"
    report_path = output_path / "numerical_convergence_report.md"
    combined.to_csv(csv_path, index=False)

    plain_se = float(
        variance.loc[variance["method"] == "plain", "standard_error"].iloc[0]
    )
    antithetic_se = float(
        variance.loc[variance["method"] == "antithetic", "standard_error"].iloc[0]
    )
    control_se = float(
        variance.loc[
            variance["method"] == "control_variate_stock", "standard_error"
        ].iloc[0]
    )

    lines = [
        "# Numerical Convergence Report",
        "",
        "This report is generated by `examples/generate_numerical_convergence_report.py`.",
        "All values are produced from executable package functions with deterministic seeds.",
        "",
        "## Executive Summary",
        "",
        "- CRR binomial prices converge toward the analytical Black-Scholes call benchmark as steps increase.",
        "- Monte Carlo estimates show sampling error and confidence intervals consistent with stochastic simulation.",
        "- Antithetic and control-variate estimators reduce standard error in the validation setup.",
        "- Analytical Greeks match finite-difference Greeks within small numerical errors.",
        "",
        "## Binomial Tree Convergence",
        "",
        _markdown_table(
            binomial[
                [
                    "steps",
                    "binomial_price",
                    "black_scholes_price",
                    "absolute_error",
                ]
            ]
        ),
        "",
        "The CRR tree is a discrete approximation to the same risk-neutral GBM",
        "dynamics that lead to the Black-Scholes formula. The absolute error",
        "falls as the number of steps increases, with small oscillations typical",
        "of lattice convergence.",
        "",
        "## Monte Carlo Convergence",
        "",
        _markdown_table(
            monte_carlo[
                [
                    "paths",
                    "mc_price",
                    "black_scholes_price",
                    "absolute_error",
                    "standard_error",
                    "ci_lower",
                    "ci_upper",
                    "analytical_inside_ci",
                ]
            ]
        ),
        "",
        "Monte Carlo convergence is statistical rather than monotonic. Increasing",
        "the number of paths generally reduces standard error at the expected",
        "`O(1 / sqrt(N))` rate, but individual pricing errors remain random.",
        "",
        "## Variance Reduction",
        "",
        _markdown_table(
            variance[
                [
                    "method",
                    "price",
                    "standard_error",
                    "ci_lower",
                    "ci_upper",
                    "standard_error_reduction_pct",
                ]
            ]
        ),
        "",
        f"In this deterministic run, plain Monte Carlo standard error is {plain_se:.6f}.",
        f"Antithetic variates reduce it to {antithetic_se:.6f}, and the stock",
        f"control variate reduces it to {control_se:.6f}. These methods target",
        "the same risk-neutral expectation while reducing estimator variance.",
        "",
        "## Analytical vs Finite-Difference Greeks",
        "",
        _markdown_table(
            greeks[
                [
                    "greek",
                    "analytical",
                    "finite_difference",
                    "absolute_error",
                    "relative_error",
                ]
            ]
        ),
        "",
        "Finite-difference Greeks provide an independent bump-and-revalue check on",
        "the analytical formulas used by the risk module.",
        "",
        "## Interpretation",
        "",
        "The numerical evidence supports the repository's validation philosophy:",
        "use analytical formulas where available, compare numerical methods",
        "against those benchmarks, report uncertainty for stochastic estimators,",
        "and avoid treating synthetic checks as formal model approval.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return combined, csv_path, report_path

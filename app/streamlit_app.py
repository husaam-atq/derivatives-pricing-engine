# ruff: noqa: E402, I001
"""Streamlit dashboard for the derivatives pricing engine."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from derivatives_engine.calibration.volatility_surface import (  # noqa: E402
    build_volatility_surface,
    plot_volatility_smile,
    plot_volatility_surface,
)
from derivatives_engine.models.binomial_tree import (
    convergence_table,
    crr_price,
)  # noqa: E402
from derivatives_engine.models.black_scholes import (
    call_price,
    price,
    put_call_parity_error,
    put_price,
)  # noqa: E402
from derivatives_engine.models.heston import (
    HestonParams,
    price_heston_mc,
    simulate_heston_paths,
)  # noqa: E402
from derivatives_engine.models.monte_carlo import (  # noqa: E402
    compare_variance_reduction,
    price_asian_arithmetic_option,
    price_barrier_option,
    price_european_option,
)
from derivatives_engine.risk.greeks import greek_table  # noqa: E402
from derivatives_engine.risk.hedging import (
    compare_rebalance_frequencies,
    simulate_delta_hedging,
)  # noqa: E402
from derivatives_engine.risk.implied_volatility import implied_volatility  # noqa: E402
from derivatives_engine.risk.scenarios import (  # noqa: E402
    combined_stress_matrix,
    greek_exposure_table,
    spot_shock_table,
    volatility_shock_table,
)
from derivatives_engine.utils.market_data import load_sample_options  # noqa: E402
from derivatives_engine.utils.plotting import histogram, plot_paths  # noqa: E402

st.set_page_config(
    page_title="Derivatives Pricing Engine",
    page_icon="",
    layout="wide",
)


@st.cache_data
def _sample_options() -> pd.DataFrame:
    return load_sample_options(ROOT / "data" / "sample_options.csv")


def _sidebar_inputs() -> dict[str, float | str]:
    st.sidebar.header("Contract")
    return {
        "S": st.sidebar.number_input("Spot", min_value=1.0, value=100.0, step=1.0),
        "K": st.sidebar.number_input("Strike", min_value=1.0, value=100.0, step=1.0),
        "T": st.sidebar.number_input("Maturity", min_value=0.01, value=1.0, step=0.05),
        "r": st.sidebar.number_input("Rate", value=0.05, step=0.005, format="%.4f"),
        "q": st.sidebar.number_input(
            "Dividend yield", value=0.0, step=0.005, format="%.4f"
        ),
        "sigma": st.sidebar.slider(
            "Volatility", min_value=0.01, max_value=1.0, value=0.20, step=0.01
        ),
        "option_type": st.sidebar.selectbox("Option type", ["call", "put"]),
    }


def _metric_row(metrics: dict[str, float | str]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items(), strict=True):
        if isinstance(value, float):
            col.metric(label, f"{value:.6f}")
        else:
            col.metric(label, value)


def overview(params: dict[str, float | str]) -> None:
    st.title("Derivatives Pricing Engine")
    call = float(
        call_price(
            params["S"],
            params["K"],
            params["T"],
            params["r"],
            params["q"],
            params["sigma"],
        )
    )
    put = float(
        put_price(
            params["S"],
            params["K"],
            params["T"],
            params["r"],
            params["q"],
            params["sigma"],
        )
    )
    _metric_row(
        {
            "Call": call,
            "Put": put,
            "Parity error": float(
                put_call_parity_error(
                    params["S"],
                    params["K"],
                    params["T"],
                    params["r"],
                    params["q"],
                    params["sigma"],
                )
            ),
        }
    )
    results_path = ROOT / "reports" / "benchmark_results.csv"
    if results_path.exists():
        st.subheader("Validation Benchmarks")
        results = pd.read_csv(results_path)
        _metric_row(
            {
                "Checks": f"{int(results['passed'].sum())}/{len(results)}",
                "Failures": int((~results["passed"]).sum()),
                "Categories": int(results["category"].nunique()),
            }
        )
        st.dataframe(results, use_container_width=True, hide_index=True)


def black_scholes_pricer(params: dict[str, float | str]) -> None:
    st.header("Black-Scholes Pricer")
    option_price = float(
        price(
            params["S"],
            params["K"],
            params["T"],
            params["r"],
            params["q"],
            params["sigma"],
            params["option_type"],
        )
    )
    _metric_row({"Option price": option_price})
    spots = np.linspace(params["S"] * 0.6, params["S"] * 1.4, 80)
    prices = price(
        spots,
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params["option_type"],
    )
    fig = go.Figure(data=[go.Scatter(x=spots, y=prices, mode="lines")])
    fig.update_layout(
        xaxis_title="Spot", yaxis_title="Option price", template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)


def greeks_visualiser(params: dict[str, float | str]) -> None:
    st.header("Greeks Visualiser")
    values = greek_table(
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params["option_type"],
    )
    _metric_row(
        {
            k: v
            for k, v in values.items()
            if k in {"delta", "gamma", "vega", "theta_annual", "rho"}
        }
    )
    spots = np.linspace(params["S"] * 0.6, params["S"] * 1.4, 80)
    rows = []
    for spot in spots:
        row = greek_table(
            spot,
            params["K"],
            params["T"],
            params["r"],
            params["q"],
            params["sigma"],
            params["option_type"],
        )
        rows.append(
            {
                "spot": spot,
                "delta": row["delta"],
                "gamma": row["gamma"],
                "vega": row["vega"],
            }
        )
    data = pd.DataFrame(rows)
    fig = go.Figure()
    for column in ["delta", "gamma", "vega"]:
        fig.add_trace(go.Scatter(x=data["spot"], y=data[column], name=column))
    fig.update_layout(
        xaxis_title="Spot", yaxis_title="Greek value", template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)


def implied_vol_solver(params: dict[str, float | str]) -> None:
    st.header("Implied Volatility Solver")
    theoretical = float(
        price(
            params["S"],
            params["K"],
            params["T"],
            params["r"],
            params["q"],
            params["sigma"],
            params["option_type"],
        )
    )
    market_price = st.number_input(
        "Market price", min_value=0.0001, value=theoretical, step=0.1
    )
    brent = implied_volatility(
        market_price,
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["option_type"],
        "brent",
    )
    auto = implied_volatility(
        market_price,
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["option_type"],
        "auto",
    )
    _metric_row(
        {
            "Brent IV": brent.implied_volatility,
            "Auto IV": auto.implied_volatility,
            "Fallback": str(auto.fallback_used),
        }
    )


def binomial_convergence(params: dict[str, float | str]) -> None:
    st.header("Binomial Tree Convergence")
    steps = st.slider("Tree steps", 10, 2000, 500, 10)
    tree_price = crr_price(
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params["option_type"],
        "american",
        steps,
    )
    european_table = convergence_table(
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params["option_type"],
    )
    _metric_row({"American tree price": tree_price})
    fig = go.Figure(
        data=[
            go.Scatter(
                x=european_table["steps"],
                y=european_table["absolute_error"],
                mode="lines+markers",
            )
        ]
    )
    fig.update_layout(
        xaxis_title="Steps", yaxis_title="Absolute error", template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(european_table, use_container_width=True, hide_index=True)


def monte_carlo_pricer(params: dict[str, float | str]) -> None:
    st.header("Monte Carlo Pricer")
    n_paths = st.slider("Paths", 10_000, 300_000, 100_000, 10_000)
    seed = st.number_input("Seed", value=42, step=1)
    vanilla = price_european_option(
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params["option_type"],
        n_paths,
        int(seed),
    )
    anti = price_european_option(
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params["option_type"],
        n_paths,
        int(seed),
        antithetic=True,
    )
    asian = price_asian_arithmetic_option(
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params["option_type"],
        30_000,
        126,
        int(seed),
    )
    barrier = price_barrier_option(
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params["option_type"],
        params["S"] * 1.3,
        "up-and-out",
        30_000,
        126,
        int(seed),
    )
    _metric_row(
        {
            "Vanilla MC": vanilla.price,
            "Std error": vanilla.standard_error,
            "Antithetic SE": anti.standard_error,
            "Asian": asian.price,
            "Up-and-out": barrier.price,
        }
    )
    st.dataframe(
        compare_variance_reduction(
            params["S"],
            params["K"],
            params["T"],
            params["r"],
            params["q"],
            params["sigma"],
            params["option_type"],
            n_paths,
            int(seed),
        ),
        use_container_width=True,
        hide_index=True,
    )


def heston_simulation(params: dict[str, float | str]) -> None:
    st.header("Heston Simulation")
    kappa = st.slider("Kappa", 0.1, 5.0, 1.5, 0.1)
    theta_v = st.slider("Long-run variance", 0.005, 0.20, 0.04, 0.005)
    sigma_v = st.slider("Vol of variance", 0.05, 1.0, 0.35, 0.05)
    rho = st.slider("Correlation", -0.95, 0.95, -0.60, 0.05)
    heston_params = HestonParams(params["sigma"] ** 2, kappa, theta_v, sigma_v, rho)
    simulation = simulate_heston_paths(
        params["S"], params["T"], params["r"], params["q"], heston_params, 400, 126, 7
    )
    heston_price = price_heston_mc(
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        heston_params,
        params["option_type"],
        20_000,
        126,
        8,
    )
    _metric_row(
        {"Heston MC": heston_price.price, "Std error": heston_price.standard_error}
    )
    st.plotly_chart(
        plot_paths(simulation.time_grid, simulation.spot_paths, "Heston spot paths"),
        use_container_width=True,
    )
    st.plotly_chart(
        plot_paths(
            simulation.time_grid,
            np.sqrt(simulation.variance_paths),
            "Heston volatility paths",
        ),
        use_container_width=True,
    )


def volatility_surface_view() -> None:
    st.header("Volatility Surface")
    chain = _sample_options()
    surface = build_volatility_surface(chain)
    st.plotly_chart(plot_volatility_smile(chain), use_container_width=True)
    st.plotly_chart(plot_volatility_surface(surface), use_container_width=True)


def delta_hedging_view(params: dict[str, float | str]) -> None:
    st.header("Delta Hedging Simulator")
    costs = st.slider("Transaction costs (bps)", 0.0, 20.0, 0.0, 0.5)
    comparison = compare_rebalance_frequencies(
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params["option_type"],
        "short",
        (12, 52, 252),
        2_000,
        252,
        costs,
        42,
    )
    result = simulate_delta_hedging(
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params["option_type"],
        "short",
        2_000,
        252,
        1,
        costs,
        42,
    )
    _metric_row(result.summary)
    st.plotly_chart(
        histogram(result.pnl, "Hedging P&L Distribution", "P&L"),
        use_container_width=True,
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)


def scenario_analysis(params: dict[str, float | str]) -> None:
    st.header("Scenario Analysis")
    spot = spot_shock_table(
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params["option_type"],
    )
    vol = volatility_shock_table(
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params["option_type"],
    )
    matrix = combined_stress_matrix(
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params["option_type"],
    )
    exposures = greek_exposure_table(
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params["option_type"],
    )
    col1, col2 = st.columns(2)
    col1.dataframe(spot, use_container_width=True, hide_index=True)
    col2.dataframe(vol, use_container_width=True, hide_index=True)
    st.dataframe(matrix, use_container_width=True, hide_index=True)
    st.dataframe(exposures, use_container_width=True, hide_index=True)


def main() -> None:
    params = _sidebar_inputs()
    section = st.sidebar.radio(
        "Section",
        [
            "Project overview",
            "Black-Scholes pricer",
            "Greeks visualiser",
            "Implied volatility solver",
            "Binomial tree convergence",
            "Monte Carlo pricer",
            "Heston simulation",
            "Volatility surface",
            "Delta hedging simulator",
            "Scenario analysis",
            "Validation benchmarks summary",
        ],
    )
    if section == "Project overview":
        overview(params)
    elif section == "Black-Scholes pricer":
        black_scholes_pricer(params)
    elif section == "Greeks visualiser":
        greeks_visualiser(params)
    elif section == "Implied volatility solver":
        implied_vol_solver(params)
    elif section == "Binomial tree convergence":
        binomial_convergence(params)
    elif section == "Monte Carlo pricer":
        monte_carlo_pricer(params)
    elif section == "Heston simulation":
        heston_simulation(params)
    elif section == "Volatility surface":
        volatility_surface_view()
    elif section == "Delta hedging simulator":
        delta_hedging_view(params)
    elif section == "Scenario analysis":
        scenario_analysis(params)
    else:
        overview(params)


if __name__ == "__main__":
    main()

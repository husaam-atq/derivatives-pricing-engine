"""Volatility surface construction and plotting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata

from derivatives_engine.risk.implied_volatility import implied_volatility


def enrich_with_implied_volatility(option_chain: pd.DataFrame) -> pd.DataFrame:
    """Add an implied_vol column to an option chain if it is missing or incomplete."""

    required = {
        "spot",
        "strike",
        "maturity",
        "rate",
        "dividend_yield",
        "option_type",
        "market_price",
    }
    missing = required.difference(option_chain.columns)
    if missing:
        raise ValueError(f"option_chain is missing required columns: {sorted(missing)}")
    enriched = option_chain.copy()
    if "implied_vol" not in enriched.columns:
        enriched["implied_vol"] = np.nan
    for idx, row in enriched[enriched["implied_vol"].isna()].iterrows():
        try:
            enriched.loc[idx, "implied_vol"] = implied_volatility(
                float(row["market_price"]),
                float(row["spot"]),
                float(row["strike"]),
                float(row["maturity"]),
                float(row["rate"]),
                float(row["dividend_yield"]),
                str(row["option_type"]),
                method="brent",
            ).implied_volatility
        except ValueError:
            enriched.loc[idx, "implied_vol"] = np.nan
    return enriched


def build_volatility_surface(
    option_chain: pd.DataFrame,
    strike_points: int = 40,
    maturity_points: int = 30,
    method: str = "linear",
) -> pd.DataFrame:
    """Interpolate an implied volatility surface over strikes and maturities."""

    enriched = enrich_with_implied_volatility(option_chain)
    clean = enriched.dropna(subset=["strike", "maturity", "implied_vol"])
    if clean.empty:
        raise ValueError("No valid implied volatility points available.")

    strikes = np.linspace(clean["strike"].min(), clean["strike"].max(), strike_points)
    maturities = np.linspace(
        clean["maturity"].min(), clean["maturity"].max(), maturity_points
    )
    strike_grid, maturity_grid = np.meshgrid(strikes, maturities)
    points = clean[["strike", "maturity"]].to_numpy()
    values = clean["implied_vol"].to_numpy()
    vol_grid = griddata(points, values, (strike_grid, maturity_grid), method=method)
    if np.isnan(vol_grid).any():
        nearest = griddata(
            points, values, (strike_grid, maturity_grid), method="nearest"
        )
        vol_grid = np.where(np.isnan(vol_grid), nearest, vol_grid)

    return pd.DataFrame(
        {
            "strike": strike_grid.ravel(),
            "maturity": maturity_grid.ravel(),
            "implied_vol": vol_grid.ravel(),
        }
    )


def smile_by_maturity(option_chain: pd.DataFrame) -> pd.DataFrame:
    """Return sorted smile data by maturity."""

    enriched = enrich_with_implied_volatility(option_chain)
    return enriched.sort_values(["maturity", "strike"])[
        ["maturity", "strike", "implied_vol"]
    ]


def plot_volatility_smile(option_chain: pd.DataFrame) -> go.Figure:
    """Plot volatility smiles by maturity using Plotly."""

    data = smile_by_maturity(option_chain)
    fig = go.Figure()
    for maturity, group in data.groupby("maturity"):
        fig.add_trace(
            go.Scatter(
                x=group["strike"],
                y=group["implied_vol"],
                mode="lines+markers",
                name=f"T={maturity:.2f}y",
            )
        )
    fig.update_layout(
        title="Implied Volatility Smile",
        xaxis_title="Strike",
        yaxis_title="Implied volatility",
        template="plotly_white",
    )
    return fig


def plot_volatility_surface(surface: pd.DataFrame) -> go.Figure:
    """Plot a 3D implied volatility surface using Plotly."""

    pivot = surface.pivot(index="maturity", columns="strike", values="implied_vol")
    fig = go.Figure(
        data=[
            go.Surface(
                x=pivot.columns.to_numpy(),
                y=pivot.index.to_numpy(),
                z=pivot.to_numpy(),
                colorscale="Viridis",
            )
        ]
    )
    fig.update_layout(
        title="Interpolated Implied Volatility Surface",
        scene={
            "xaxis_title": "Strike",
            "yaxis_title": "Maturity",
            "zaxis_title": "Implied volatility",
        },
        template="plotly_white",
    )
    return fig

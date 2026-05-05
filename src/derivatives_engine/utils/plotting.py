"""Shared plotting helpers for examples, notebooks and dashboard."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def plot_paths(
    time_grid: np.ndarray, paths: np.ndarray, title: str, max_paths: int = 25
) -> go.Figure:
    """Plot a subset of simulated paths."""

    fig = go.Figure()
    for idx in range(min(max_paths, paths.shape[0])):
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=paths[idx],
                mode="lines",
                opacity=0.45,
                showlegend=False,
            )
        )
    fig.update_layout(
        title=title, xaxis_title="Time", yaxis_title="Value", template="plotly_white"
    )
    return fig


def histogram(values: np.ndarray, title: str, xaxis_title: str) -> go.Figure:
    """Return a Plotly histogram."""

    fig = go.Figure(data=[go.Histogram(x=values, nbinsx=60)])
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Count",
        template="plotly_white",
    )
    return fig

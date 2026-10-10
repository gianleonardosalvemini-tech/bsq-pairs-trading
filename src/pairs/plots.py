"""Shared look for every chart, and PNG export for the README."""

from __future__ import annotations

import logging
from pathlib import Path

import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def finish_figure(fig: go.Figure, title: str, x_title: str, y_title: str, source: str) -> go.Figure:
    """Apply the common layout: title, axis labels with units, source note under the chart.

    Parameters
    ----------
    fig : go.Figure
        Chart with its traces already added.
    title : str
        One sentence saying what the chart shows.
    x_title, y_title : str
        Axis labels, with units in brackets.
    source : str
        Data source and period, printed under the chart.

    Returns
    -------
    go.Figure
        The same figure, restyled in place.

    Raises
    ------
    ValueError
        If plotly rejects a layout value (raised by plotly).

    Notes
    -----
    One function so that every chart in every phase looks the same: a reader learns the
    layout once (title on top, source at the bottom, legend above the plot).
    """
    fig.update_layout(
        title={"text": title, "x": 0.01},
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        width=1000,
        height=500,
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0.01},
        margin={"b": 90},
    )
    fig.add_annotation(
        text=source, xref="paper", yref="paper", x=0, y=-0.2, showarrow=False, font={"size": 11}
    )
    return fig


def save_figure(fig: go.Figure, name: str, figures_dir: Path) -> Path:
    """Export the chart as a PNG for the README.

    Parameters
    ----------
    fig : go.Figure
        Finished chart.
    name : str
        File name without extension, e.g. ``"01_prices"``.
    figures_dir : Path
        Destination folder (``results/figures``).

    Returns
    -------
    Path
        Path of the written PNG.

    Raises
    ------
    AssertionError
        If the file was not written.

    Notes
    -----
    The notebook keeps the interactive version; GitHub cannot render Plotly, so the
    README embeds these static copies (exported with kaleido).
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.write_image(path, scale=2)
    assert path.exists(), f"{path} not written"
    logger.info("Saved figure %s", path)
    return path

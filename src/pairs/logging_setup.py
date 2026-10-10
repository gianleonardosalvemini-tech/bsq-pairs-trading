"""Logging configuration shared by notebooks and command-line entry points."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def configure_logging(level: int = logging.INFO) -> None:
    """Route every ``logging.getLogger(__name__)`` logger through a Rich handler.

    Parameters
    ----------
    level : int, default logging.INFO
        Minimum level shown.

    Notes
    -----
    Modules never call ``print``: they log. Only entry points (notebooks, CLI) call
    this function, once, so that library code stays silent unless asked.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )

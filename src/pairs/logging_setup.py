"""Logging configuration shared by notebooks and command-line entry points."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Route every ``logging.getLogger(__name__)`` logger through a Rich handler.

    Parameters
    ----------
    level : int, default logging.INFO
        Minimum level shown for the ``pairs`` package.

    Returns
    -------
    None
        The root logger is configured in place.

    Raises
    ------
    ValueError
        If ``level`` is not a valid logging level (raised by ``logging``).

    Notes
    -----
    Modules never call ``print``: they log. Only entry points (notebooks, CLI) call
    this function, once, so that library code stays silent unless asked. ``level``
    applies to this project's loggers only; third-party libraries (e.g. kaleido, which
    logs every PNG export) are shown from WARNING up, to keep notebook output readable.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )
    logging.getLogger("pairs").setLevel(level)
    logger.debug("Logging configured at level %s", logging.getLevelName(level))

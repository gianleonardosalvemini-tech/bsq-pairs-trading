"""SQLite log of every backtest run, so the number of things tried can be reported."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL,
    phase TEXT NOT NULL,
    pair TEXT NOT NULL,
    sample TEXT NOT NULL,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    params TEXT NOT NULL,
    metrics TEXT NOT NULL,
    note TEXT NOT NULL
)
"""


def log_trial(
    db_path: Path,
    phase: str,
    pair: str,
    sample: str,
    period: tuple[date, date],
    params: Mapping[str, float | int | str],
    metrics: Mapping[str, float],
    note: str = "",
) -> int:
    """Append one backtest to the trial log.

    Parameters
    ----------
    db_path : Path
        SQLite file (created with its table if missing).
    phase : str
        e.g. ``"phase2-grid"``, ``"phase2-zoom"``.
    pair : str
        e.g. ``"KO/PEP"``.
    sample : str
        ``"in-sample"`` or ``"out-of-sample"``.
    period : tuple of date
        First and last day of the evaluated returns.
    params : Mapping
        Everything that defines the variant (threshold, window, beta, ...).
    metrics : Mapping
        Output of :func:`pairs.backtest.summarize`.
    note : str, default ""
        Free text.

    Returns
    -------
    int
        Id of the new row.

    Raises
    ------
    AssertionError
        If ``sample`` is not one of the two allowed values.

    Notes
    -----
    Every backtest in the project goes through here: the README reports the row count,
    because the brief asks how many things were tried. Tests point ``db_path`` to a
    temporary folder so they never inflate the real count.
    """
    assert sample in ("in-sample", "out-of-sample"), f"unknown sample {sample!r}"
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(_SCHEMA)
        cursor = conn.execute(
            "INSERT INTO trials (logged_at, phase, pair, sample, period_start, period_end,"
            " params, metrics, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                phase,
                pair,
                sample,
                period[0].isoformat(),
                period[1].isoformat(),
                json.dumps(dict(params)),
                json.dumps(dict(metrics)),
                note,
            ),
        )
        row_id = cursor.lastrowid
    assert row_id is not None
    logger.info("Logged trial %d (%s, %s, %s)", row_id, phase, sample, params)
    return row_id


def count_trials(db_path: Path, phase: str | None = None) -> int:
    """Count the logged backtests, optionally for one phase.

    Parameters
    ----------
    db_path : Path
        SQLite file written by :func:`log_trial`.
    phase : str or None, default None
        Count only this phase; ``None`` counts everything.

    Returns
    -------
    int
        Row count (0 if the file does not exist yet).

    Notes
    -----
    Reading never creates the file, so counting cannot change what is counted.
    """
    if not db_path.exists():
        return 0
    with closing(sqlite3.connect(db_path)) as conn:
        if phase is None:
            row = conn.execute("SELECT COUNT(*) FROM trials").fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM trials WHERE phase = ?", (phase,)).fetchone()
    return int(row[0])

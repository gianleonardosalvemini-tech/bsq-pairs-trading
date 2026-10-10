"""Structured configuration: environment settings and research parameters."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from pairs.errors import ConfigError

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Paths loaded from the environment (``.env``), validated at start-up.

    Attributes
    ----------
    data_dir : Path
        Folder for downloaded data and pickled checkpoints (versioned: Yahoo rewrites
        adjusted prices after each dividend, so a re-download would not reproduce results).
    results_dir : Path
        Folder for figures and the SQLite trial log (versioned).
    """

    data_dir: Path
    results_dir: Path

    @classmethod
    def from_env(cls) -> Settings:
        """Build the settings from environment variables, with project defaults.

        Parameters
        ----------
        None
            Reads ``PAIRS_DATA_DIR`` and ``PAIRS_RESULTS_DIR`` from the environment / ``.env``.

        Returns
        -------
        Settings
            Validated settings; the folders are created if missing.

        Raises
        ------
        ConfigError
            If a configured path exists but is not a directory.

        Notes
        -----
        No API key is needed: Yahoo Finance and the Nasdaq short-interest endpoint are
        public. ``.env`` is therefore optional and only overrides the default folders.
        """
        load_dotenv(PROJECT_ROOT / ".env")
        data_dir = Path(os.environ.get("PAIRS_DATA_DIR", PROJECT_ROOT / "data"))
        results_dir = Path(os.environ.get("PAIRS_RESULTS_DIR", PROJECT_ROOT / "results"))
        for folder in (data_dir, results_dir):
            if folder.exists() and not folder.is_dir():
                raise ConfigError(f"{folder} exists but is not a directory")
            folder.mkdir(parents=True, exist_ok=True)
        logger.debug("Settings: data_dir=%s results_dir=%s", data_dir, results_dir)
        return cls(data_dir=data_dir, results_dir=results_dir)


@dataclass(frozen=True)
class PairConfig:
    """The pair and the in-sample / out-of-sample split, fixed before any test.

    Attributes
    ----------
    ticker_a : str
        Dependent leg of the regression ``A = alpha + beta * B``.
    ticker_b : str
        Hedge leg.
    start : date
        First day of the sample.
    in_sample_end : date
        Last in-sample day. Every estimate (hedge ratio, thresholds, calibrations)
        uses ``as_of=in_sample_end``.
    end : date
        Last day of the sample (end of the out-of-sample period).

    Raises
    ------
    ConfigError
        If the tickers coincide or the dates are not strictly increasing.
    """

    ticker_a: str
    ticker_b: str
    start: date
    in_sample_end: date
    end: date

    def __post_init__(self) -> None:
        """Fail fast on an inconsistent configuration.

        Parameters
        ----------
        None
            Validates the fields set by the dataclass constructor.

        Returns
        -------
        None

        Raises
        ------
        ConfigError
            If the tickers coincide or the dates are not strictly increasing.

        Notes
        -----
        Validation happens at construction, so an invalid split can never reach a backtest.
        """
        if not self.ticker_a or not self.ticker_b or self.ticker_a == self.ticker_b:
            raise ConfigError(f"Invalid pair: {self.ticker_a!r} / {self.ticker_b!r}")
        if not self.start < self.in_sample_end < self.end:
            raise ConfigError(
                f"Dates must satisfy start < in_sample_end < end, got "
                f"{self.start} / {self.in_sample_end} / {self.end}"
            )


# Fixed at 10:55 on 10 Oct 2026, before downloading or looking at any price data.
# In-sample 2000-2018 (~19 years) for every estimate; out-of-sample 2019 to the last close
# before the project day (~7.8 years, ~29% of the sample), run once at the end.
PAIR = PairConfig(
    ticker_a="KO",
    ticker_b="PEP",
    start=date(2000, 1, 3),
    in_sample_end=date(2018, 12, 31),
    end=date(2026, 10, 9),
)

# Phase 1 describes the whole common history on Yahoo: 1972-06-01 is PepsiCo's first day there
# (Coca-Cola starts in 1962). Descriptive only: nothing is estimated before PAIR.start.
HISTORY_START = date(1972, 6, 1)


@dataclass(frozen=True)
class SignalGrid:
    """Phase 2 rules, written down before the first backtest.

    Attributes
    ----------
    entry_thresholds : tuple of float
        |z| needed to open a position. The interior values are the brief's own example
        ("1.5, 2, 2.5"); the two edges only measure sensitivity.
    windows : tuple of int
        Trading days of the rolling mean and standard deviation of the spread.
    exit_threshold : float
        z level at which a position is closed ("reverts back toward zero").
    min_trades : int
        Fewest in-sample trades for a cell to be selectable.
    threshold_offsets : tuple of float
        Small changes added to the chosen threshold in the sensitivity check.
    window_factors : tuple of float
        Small relative changes applied to the chosen window in the sensitivity check.

    Raises
    ------
    ConfigError
        If a grid axis has fewer than three values (no interior cell), is not strictly
        increasing, or the exit level is not below every entry threshold.

    Notes
    -----
    Selection: among interior cells with at least ``min_trades`` trades, the highest
    in-sample Sharpe averaged over the cell and its eight neighbours; ties go to fewer
    trades. Edge cells are never selected because their average uses fewer neighbours
    and is noisier. The sensitivity check (offsets x factors around the chosen cell) is
    a diagnostic only: it is logged as trials and never used to re-select.
    """

    entry_thresholds: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
    windows: tuple[int, ...] = (20, 60, 120, 190, 250)
    exit_threshold: float = 0.0
    min_trades: int = 20
    threshold_offsets: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)
    window_factors: tuple[float, ...] = (0.9, 0.95, 1.0, 1.05, 1.1)

    def __post_init__(self) -> None:
        """Fail fast on an inconsistent grid.

        Parameters
        ----------
        None
            Validates the fields set by the dataclass constructor.

        Returns
        -------
        None

        Raises
        ------
        ConfigError
            If an axis is too short or unsorted, or the exit is not below every entry.

        Notes
        -----
        Checked at construction, so a broken grid can never reach a backtest.
        """
        for name, axis in (("entry_thresholds", self.entry_thresholds), ("windows", self.windows)):
            if len(axis) < 3 or list(axis) != sorted(set(axis)):
                raise ConfigError(f"{name} needs >= 3 strictly increasing values, got {axis}")
        if not 0 <= self.exit_threshold < self.entry_thresholds[0]:
            raise ConfigError(f"exit {self.exit_threshold} must be in [0, lowest entry)")
        if self.windows[0] < 2 or self.min_trades < 1:
            raise ConfigError("windows must be >= 2 days and min_trades >= 1")


# Fixed at 12:50 on 10 Oct 2026 after the in-sample spread check (half-life ~192 days),
# before any backtest. 25 grid cells + 24 sensitivity variants = 49 declared trials.
SIGNAL_GRID = SignalGrid()


@dataclass(frozen=True)
class FrozenStrategy:
    """The Phase 2 variant chosen on the in-sample period, frozen before the out-of-sample run.

    Attributes
    ----------
    entry : float
        Entry threshold on |z|.
    window : int
        Rolling window of the z-score, in trading days.
    exit : float
        Exit level on z.
    beta : float
        OLS hedge ratio of KO on PEP estimated with ``as_of=PAIR.in_sample_end``.

    Raises
    ------
    ConfigError
        If the values are inconsistent (exit not below entry, window too short, beta <= 0).

    Notes
    -----
    Chosen by the pre-registered stable-zone rule of :class:`SignalGrid` (zone score
    0.361 among the nine interior cells). Nothing here may change after the out-of-sample
    run; later phases analyse this exact strategy.
    """

    entry: float
    window: int
    exit: float
    beta: float

    def __post_init__(self) -> None:
        """Fail fast on an inconsistent frozen strategy.

        Parameters
        ----------
        None
            Validates the fields set by the dataclass constructor.

        Returns
        -------
        None

        Raises
        ------
        ConfigError
            If exit is not in [0, entry), the window is shorter than 2 days or beta <= 0.

        Notes
        -----
        Same checks as :class:`SignalGrid`, applied to the single chosen variant.
        """
        if not 0 <= self.exit < self.entry or self.window < 2 or self.beta <= 0:
            raise ConfigError(f"Inconsistent frozen strategy: {self}")


# Frozen at 13:20 on 10 Oct 2026, after 49 in-sample trials and before any look at
# 2019-2026. The notebook re-estimates beta and asserts it matches this value.
PHASE2_STRATEGY = FrozenStrategy(entry=1.5, window=60, exit=0.0, beta=0.388372)


@dataclass(frozen=True)
class KalmanRules:
    """Extension rules: a hedge ratio that moves with a Kalman filter, fixed before computing.

    Attributes
    ----------
    delta : float
        Speed of the hedge ratio. Each day the intercept and the hedge ratio take a random
        step with variance ``delta / (1 - delta)``. 1e-4 is the textbook value (E. Chan,
        Algorithmic Trading, 2013), taken as is and never tuned.
    sensitivity_deltas : tuple of float
        Slower and faster filters, run only to show how much the result depends on ``delta``.
    init_end : date
        The filter starts from the OLS fit on the prices up to this day (the year 2000) and
        runs from the next trading day.

    Raises
    ------
    ConfigError
        If a delta is not in (0, 1) or the start-up period is outside the in-sample period.

    Notes
    -----
    The observation noise is the variance of the residuals of the Phase 2 OLS fit (as_of
    the in-sample end). The spread of day t uses only the intercept and hedge ratio
    estimated with data up to t-1. The z-score, thresholds and exit are the frozen Phase 2
    ones (window 60, entry 1.5, exit 0). The hedge ratio of a trade is the one of the day
    the trade opens and stays fixed until it closes. No constraint on its sign. No
    transaction costs, so the result compares with Phase 2. Decided after the single
    out-of-sample run of Phase 2: 2019-2026 is not a fresh out-of-sample period here.
    """

    delta: float = 1e-4
    sensitivity_deltas: tuple[float, ...] = (1e-5, 1e-3)
    init_end: date = date(2000, 12, 31)

    def __post_init__(self) -> None:
        """Fail fast on inconsistent rules.

        Parameters
        ----------
        None
            Validates the fields set by the dataclass constructor.

        Returns
        -------
        None

        Raises
        ------
        ConfigError
            If a delta is not in (0, 1) or ``init_end`` is not inside the in-sample period.

        Notes
        -----
        Checked at construction, like the other research parameters.
        """
        if not all(0 < d < 1 for d in (self.delta, *self.sensitivity_deltas)):
            raise ConfigError("every delta must be in (0, 1)")
        if not PAIR.start < self.init_end < PAIR.in_sample_end:
            raise ConfigError(f"init_end {self.init_end} must be inside the in-sample period")


# Fixed at 18:50 on 10 Oct 2026, before running any Kalman filter on the data. 1 main
# variant + 2 sensitivity deltas, each in-sample and out-of-sample = 6 declared trials.
KALMAN_RULES = KalmanRules()

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

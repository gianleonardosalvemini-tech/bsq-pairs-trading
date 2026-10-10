"""Phase 1 data pipeline: Yahoo download and raw / parsed / validated checkpoints."""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from pairs.config import PairConfig
from pairs.errors import DataFetchError, StaleDataError

logger = logging.getLogger(__name__)

RAW_FILE = "prices_raw.pkl"
PARSED_FILE = "prices_parsed.pkl"
VALIDATED_FILE = "prices_validated.pkl"
MAX_EDGE_GAP = timedelta(days=5)


def fetch_raw_prices(pair: PairConfig, data_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Download daily prices for both legs once and checkpoint them exactly as received.

    Parameters
    ----------
    pair : PairConfig
        Tickers and the full sample window (``start`` to ``end``, both inclusive).
    data_dir : Path
        Folder holding the pickled checkpoints.
    refresh : bool, default False
        Re-download even if the raw checkpoint exists.

    Returns
    -------
    pd.DataFrame
        Yahoo output untouched: columns are a (field, ticker) MultiIndex, index is the date.

    Raises
    ------
    DataFetchError
        If Yahoo returns no rows, or no Adj Close for either ticker (yfinance reports
        most failures by logging and returning an empty frame, not by raising).

    Notes
    -----
    The raw checkpoint is read back instead of re-downloading because Yahoo rewrites
    adjusted prices after every dividend: a new download would not reproduce our numbers.
    ``auto_adjust=False`` keeps both Close and Adj Close, so the choice is made (and
    documented) in the parsing step, not hidden inside the download. Yahoo's ``end`` is
    exclusive, hence the extra day.
    """
    path = data_dir / RAW_FILE
    if path.exists() and not refresh:
        logger.info("Loading raw checkpoint %s", path)
        cached: pd.DataFrame = pd.read_pickle(path)
        return cached
    tickers = [pair.ticker_a, pair.ticker_b]
    logger.info("Downloading %s from Yahoo, %s to %s", tickers, pair.start, pair.end)
    raw: pd.DataFrame | None = yf.download(  # yfinance is untyped: pin the type here
        tickers,
        start=pair.start.isoformat(),
        end=(pair.end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise DataFetchError(f"Yahoo returned no rows for {tickers}")
    for ticker in tickers:
        if raw["Adj Close"][ticker].dropna().empty:
            raise DataFetchError(f"No Adj Close data for {ticker}")
    raw.to_pickle(path)
    logger.info("Saved raw checkpoint %s (%d rows)", path, len(raw))
    return raw


def parse_prices(raw: pd.DataFrame, pair: PairConfig, data_dir: Path) -> pd.DataFrame:
    """Keep the adjusted close of both legs, one column per ticker, and checkpoint it.

    Parameters
    ----------
    raw : pd.DataFrame
        Output of :func:`fetch_raw_prices` (field, ticker) MultiIndex columns.
    pair : PairConfig
        Gives the column order: ``ticker_a`` first, then ``ticker_b``.
    data_dir : Path
        Folder holding the pickled checkpoints.

    Returns
    -------
    pd.DataFrame
        Columns ``[ticker_a, ticker_b]``, sorted tz-naive DatetimeIndex named ``date``.

    Raises
    ------
    KeyError
        If ``raw`` lacks the Adj Close column of either ticker (raised by pandas).

    Notes
    -----
    Adjusted close (dividends and splits) is used for both returns and the spread: a
    raw close drops on every ex-dividend date, which would show up as a fake spread move.
    """
    # With MultiIndex columns raw["Adj Close"] is a DataFrame; the stubs only know Series.
    prices = pd.DataFrame(raw["Adj Close"][[pair.ticker_a, pair.ticker_b]])
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    prices.index.name = "date"
    prices.columns.name = None
    prices = prices.sort_index()
    prices.to_pickle(data_dir / PARSED_FILE)
    logger.info("Saved parsed checkpoint (%d rows, Adj Close)", len(prices))
    return prices


def validate_prices(prices: pd.DataFrame, pair: PairConfig, data_dir: Path) -> pd.DataFrame:
    """Drop days missing either leg, check coverage and invariants, and checkpoint.

    Parameters
    ----------
    prices : pd.DataFrame
        Output of :func:`parse_prices`.
    pair : PairConfig
        Gives the required sample window.
    data_dir : Path
        Folder holding the pickled checkpoints.

    Returns
    -------
    pd.DataFrame
        Same layout, only days where both legs trade, every price strictly positive.

    Raises
    ------
    StaleDataError
        If the data starts or ends more than five days away from the configured window.

    Notes
    -----
    A pair needs both prices on the same day, so a day with one leg missing is dropped
    (and counted in the log) rather than forward-filled: filling would invent a zero
    return for that leg. Five days of tolerance covers holidays at the window edges.
    """
    n_before = len(prices)
    clean = prices.dropna(how="any")
    logger.info("Dropped %d day(s) with a missing leg", n_before - len(clean))
    if clean.empty or clean.index.min().date() > pair.start + MAX_EDGE_GAP:
        raise StaleDataError(f"Data does not start near {pair.start}")
    if clean.index.max().date() < pair.end - MAX_EDGE_GAP:
        raise StaleDataError(f"Data does not reach {pair.end}")
    assert list(clean.columns) == [pair.ticker_a, pair.ticker_b], "column order changed"
    assert clean.index.is_monotonic_increasing and clean.index.is_unique, "bad date index"
    assert (clean > 0).all().all(), "non-positive price"
    clean.to_pickle(data_dir / VALIDATED_FILE)
    logger.info("Saved validated checkpoint (%d rows)", len(clean))
    return clean

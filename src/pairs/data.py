"""Phase 1 data pipeline: Yahoo prices and corporate actions, pickled checkpoints."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from pairs.config import PairConfig
from pairs.errors import DataFetchError, StaleDataError

logger = logging.getLogger(__name__)

RAW_FILE = "prices_raw.pkl"
PARSED_FILE = "prices_parsed.pkl"
VALIDATED_FILE = "prices_validated.pkl"
ACTIONS_FILE = "actions_raw.pkl"
CONTEXT_FILE = "prices_context_raw.pkl"
CONTEXT_START = date(1962, 1, 2)  # Yahoo's first KO day; PEP starts on 1972-06-01
MAX_EDGE_GAP = timedelta(days=5)


@dataclass(frozen=True)
class DividendError:
    """A dividend that Yahoo reports, and bakes into Adj Close, but that was never paid.

    Attributes
    ----------
    ticker : str
        Stock concerned.
    ex_date : date
        Ex-date of the wrong record.
    amount : float
        Wrong amount, in dollars per share as Yahoo reports it.
    reason : str
        Why the record is wrong (shown in the notebook).
    """

    ticker: str
    ex_date: date
    amount: float
    reason: str


# Found on 10 Oct 2026 by listing every in-sample dividend date by date (Phase 1 notebook).
KNOWN_DIVIDEND_ERRORS = (
    DividendError(
        ticker="KO",
        ex_date=date(2001, 9, 10),
        amount=0.18,
        reason=(
            "Duplicate of the real 0.09 dividend of 2001-09-17 (ex-date moved by the market "
            "closure after 11 September), at its amount before the 2012 2-for-1 split"
        ),
    ),
)


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
    documented) in the parsing step, not hidden inside the download.
    """
    path = data_dir / RAW_FILE
    if path.exists() and not refresh:
        logger.info("Loading raw checkpoint %s", path)
        cached: pd.DataFrame = pd.read_pickle(path)
        return cached
    return _download([pair.ticker_a, pair.ticker_b], pair.start, pair.end, path)


def fetch_context_prices(pair: PairConfig, data_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Download the history *before* the project sample, used only as context in Phase 1.

    Parameters
    ----------
    pair : PairConfig
        Tickers; the history ends the day before ``pair.start``.
    data_dir : Path
        Folder holding the pickled checkpoints.
    refresh : bool, default False
        Re-download even if the checkpoint exists.

    Returns
    -------
    pd.DataFrame
        Yahoo output untouched, from ``CONTEXT_START`` (or the first available day) to the
        day before ``pair.start``; same layout as :func:`fetch_raw_prices`.

    Raises
    ------
    DataFetchError
        If Yahoo returns no rows, or no Adj Close for either ticker.

    Notes
    -----
    This history is never used to estimate or tune anything: it lies before the
    in-sample period, so looking at it cannot contaminate the out-of-sample test. It is
    given less weight because PepsiCo was a different business before 1997-1999
    (restaurants, bottlers) and prices were quoted in fractions of a dollar until 2001.
    """
    path = data_dir / CONTEXT_FILE
    if path.exists() and not refresh:
        logger.info("Loading context checkpoint %s", path)
        cached: pd.DataFrame = pd.read_pickle(path)
        return cached
    tickers = [pair.ticker_a, pair.ticker_b]
    return _download(tickers, CONTEXT_START, pair.start - timedelta(days=1), path)


def _download(tickers: list[str], start: date, end: date, path: Path) -> pd.DataFrame:
    """Download daily prices from Yahoo (``end`` inclusive), check them, and pickle them.

    Parameters
    ----------
    tickers : list of str
        Yahoo tickers.
    start, end : date
        First and last day requested, both inclusive.
    path : Path
        Checkpoint file to write.

    Returns
    -------
    pd.DataFrame
        Yahoo output untouched.

    Raises
    ------
    DataFetchError
        If Yahoo returns no rows, or no Adj Close for either ticker.

    Notes
    -----
    Yahoo's ``end`` is exclusive, hence the extra day.
    """
    logger.info("Downloading %s from Yahoo, %s to %s", tickers, start, end)
    raw: pd.DataFrame | None = yf.download(  # yfinance is untyped: pin the type here
        tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise DataFetchError(f"Yahoo returned no rows for {tickers}")
    for ticker in tickers:
        if raw["Adj Close"][ticker].dropna().empty:
            raise DataFetchError(f"No Adj Close data for {ticker}")
    raw.to_pickle(path)
    logger.info("Saved checkpoint %s (%d rows)", path, len(raw))
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


def fetch_corporate_actions(
    pair: PairConfig, data_dir: Path, refresh: bool = False
) -> pd.DataFrame:
    """Download the dividends and stock splits of both legs once, and checkpoint them.

    Parameters
    ----------
    pair : PairConfig
        Tickers whose corporate actions are needed.
    data_dir : Path
        Folder holding the pickled checkpoints.
    refresh : bool, default False
        Re-download even if the checkpoint exists.

    Returns
    -------
    pd.DataFrame
        One row per event: columns ``ticker``, ``dividend`` (dollars per share, 0 if none)
        and ``split`` (shares received per share held, 0 if none); tz-naive DatetimeIndex
        named ``date`` (the ex-date).

    Raises
    ------
    DataFetchError
        If Yahoo returns no dividend for either ticker (both pay quarterly dividends).

    Notes
    -----
    Needed to explain, and verify, the adjusted close: every dividend and split must be
    reported, because it changes how past prices read. Yahoo already divides dividend
    amounts paid before a split by the split ratio, like the prices.
    """
    path = data_dir / ACTIONS_FILE
    if path.exists() and not refresh:
        logger.info("Loading corporate actions checkpoint %s", path)
        cached: pd.DataFrame = pd.read_pickle(path)
        return cached
    frames = []
    for ticker in (pair.ticker_a, pair.ticker_b):
        actions: pd.DataFrame = yf.Ticker(ticker).actions
        if actions.empty or not (actions["Dividends"] > 0).any():
            raise DataFetchError(f"No dividends returned for {ticker}")
        frame = pd.DataFrame(
            {
                "ticker": ticker,
                "dividend": actions["Dividends"].to_numpy(),
                "split": actions["Stock Splits"].to_numpy(),
            },
            index=pd.DatetimeIndex(actions.index).tz_localize(None).normalize(),
        )
        frames.append(frame)
    events = pd.concat(frames).sort_index()
    events.index.name = "date"
    events.to_pickle(path)
    logger.info("Saved corporate actions checkpoint %s (%d events)", path, len(events))
    return events


def correct_dividend_errors(
    prices: pd.DataFrame, raw: pd.DataFrame, errors: tuple[DividendError, ...]
) -> pd.DataFrame:
    """Remove from the adjusted prices the effect of dividends that were never paid.

    Parameters
    ----------
    prices : pd.DataFrame
        Output of :func:`parse_prices` (adjusted close, one column per ticker).
    raw : pd.DataFrame
        Output of :func:`fetch_raw_prices`, for the unadjusted close.
    errors : tuple of DividendError
        The wrong dividend records, e.g. ``KNOWN_DIVIDEND_ERRORS``.

    Returns
    -------
    pd.DataFrame
        A corrected copy; ``prices`` itself is not modified.

    Raises
    ------
    AssertionError
        If, after the correction, the adjusted return on the ex-date still differs from
        the close return (which it must equal, since nothing was paid that day).

    Notes
    -----
    Yahoo multiplied every adjusted price before the ex-date by ``1 - D / C`` (``D`` the
    wrong dividend, ``C`` the close of the previous day); dividing by the same factor
    undoes exactly that and nothing else. The raw checkpoint stays as Yahoo sent it.
    """
    fixed = prices.copy()
    for error in errors:
        ex_day = pd.Timestamp(error.ex_date)
        if error.ticker not in fixed.columns or ex_day not in fixed.index:
            continue
        close = raw[("Close", error.ticker)]
        position = close.index.get_loc(ex_day)
        assert isinstance(position, int), "duplicate dates"
        factor = 1 - error.amount / float(close.iloc[position - 1])
        fixed.loc[fixed.index < ex_day, error.ticker] /= factor
        adj_return = fixed[error.ticker].pct_change().loc[ex_day]
        close_return = close.pct_change().loc[ex_day]
        assert abs(adj_return - close_return) < 1e-4, "correction did not remove the dividend"
        logger.info("Removed wrong %s dividend of %s (factor %.5f)", error.ticker, ex_day, factor)
    return fixed


def load_pair_prices(pair: PairConfig, data_dir: Path, last_day: date) -> pd.DataFrame:
    """Load the pair's adjusted close from ``pair.start`` to ``last_day``, known errors fixed.

    Parameters
    ----------
    pair : PairConfig
        The pair and its sample dates.
    data_dir : Path
        Folder with the raw checkpoint (see :func:`fetch_raw_prices`).
    last_day : date
        Last date kept (inclusive), e.g. ``pair.in_sample_end`` for in-sample work.

    Returns
    -------
    pd.DataFrame
        One column per ticker (``ticker_a``, ``ticker_b``), days with both prices only.

    Raises
    ------
    AssertionError
        If a date after ``last_day`` or before ``pair.start`` survives.

    Notes
    -----
    The single entry point of Phases 2-5: dates are cut first, then the wrong dividends of
    ``KNOWN_DIVIDEND_ERRORS`` are removed with :func:`correct_dividend_errors`.
    """
    raw = fetch_raw_prices(pair, data_dir)
    raw = raw.loc[(raw.index >= pd.Timestamp(pair.start)) & (raw.index <= pd.Timestamp(last_day))]
    adjusted = pd.DataFrame(raw["Adj Close"][[pair.ticker_a, pair.ticker_b]])
    prices = correct_dividend_errors(adjusted, raw, KNOWN_DIVIDEND_ERRORS).dropna()
    assert prices.index.min() >= pd.Timestamp(pair.start), "data before pair.start"
    assert prices.index.max() <= pd.Timestamp(last_day), "data after last_day"
    logger.info(
        "Loaded %d days of %s/%s up to %s", len(prices), pair.ticker_a, pair.ticker_b, last_day
    )
    return prices

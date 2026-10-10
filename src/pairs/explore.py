"""Phase 1 calculations: in-sample cut, daily returns, correlation by calendar year."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)


def select_until(prices: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Keep only the rows dated on or before ``as_of``.

    Parameters
    ----------
    prices : pd.DataFrame
        Any frame with a DatetimeIndex.
    as_of : date
        Last date allowed (inclusive).

    Returns
    -------
    pd.DataFrame
        The rows with ``date <= as_of``.

    Raises
    ------
    AssertionError
        If a row after ``as_of`` survives the filter.

    Notes
    -----
    This is the project's guard against look-ahead: every analysis or estimate on the
    in-sample period goes through it with ``as_of=PAIR.in_sample_end``, so data from the
    out-of-sample period cannot leak in by mistake.
    """
    kept = prices.loc[prices.index <= pd.Timestamp(as_of)]
    assert kept.index.max() <= pd.Timestamp(as_of), "data after as_of leaked through"
    logger.info("Kept %d of %d rows up to %s", len(kept), len(prices), as_of)
    return kept


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns, ``P_t / P_{t-1} - 1``, for every column.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted prices, one column per asset.

    Returns
    -------
    pd.DataFrame
        Same columns, one row fewer (the first day has no previous price).

    Raises
    ------
    AssertionError
        If a return is missing after dropping the first row.

    Notes
    -----
    Adjusted prices already include dividends, so these are total returns. Simple
    (not log) returns are used because they are what a reader intuitively expects.
    """
    returns = prices.pct_change().iloc[1:]
    assert not returns.isna().any().any(), "missing return"
    return returns


def yearly_correlation(returns: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
    """Correlation of the two assets' daily returns within each calendar year.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns with a DatetimeIndex.
    col_a, col_b : str
        The two columns to correlate.

    Returns
    -------
    pd.Series
        One correlation per year, indexed by the year (int).

    Raises
    ------
    KeyError
        If either column is missing (raised by pandas).

    Notes
    -----
    Calendar years give ~250 days per estimate, enough for a stable number, and are easy
    to read for a non-specialist. A rolling 60-120 day correlation is one of the Phase 3
    methods, so it is deliberately not used here.
    """
    years = pd.DatetimeIndex(returns.index).year
    by_year = {
        int(year): float(group[col_a].corr(group[col_b])) for year, group in returns.groupby(years)
    }
    return pd.Series(by_year, name="correlation")

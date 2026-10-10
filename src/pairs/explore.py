"""Phase 1 calculations: in-sample cut, data quality checks, dividend adjustment, returns."""

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


def longest_gaps(index: pd.DatetimeIndex, n: int) -> pd.DataFrame:
    """List the ``n`` longest pauses between two consecutive trading days.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Trading days, sorted.
    n : int
        How many gaps to return.

    Returns
    -------
    pd.DataFrame
        Columns ``from``, ``to`` (dates) and ``calendar_days``, longest first.

    Raises
    ------
    AssertionError
        If the index is not sorted.

    Notes
    -----
    A normal weekend is a 3-day gap. Anything longer must match a known market closure
    (holiday, exceptional event); otherwise it would be missing data.
    """
    assert index.is_monotonic_increasing, "index must be sorted"
    days = pd.Series(index)
    gap = days.diff().dt.days
    top = gap.nlargest(n).index
    return pd.DataFrame(
        {
            "from": days[top - 1].dt.date.to_numpy(),
            "to": days[top].dt.date.to_numpy(),
            "calendar_days": gap[top].astype(int).to_numpy(),
        }
    )


def largest_moves(close: pd.DataFrame, volume: pd.DataFrame, n: int) -> pd.DataFrame:
    """List the ``n`` largest one-day price moves, with evidence for or against a data error.

    Parameters
    ----------
    close : pd.DataFrame
        Prices, one column per ticker.
    volume : pd.DataFrame
        Traded shares, same layout as ``close``.
    n : int
        How many moves to return.

    Returns
    -------
    pd.DataFrame
        Columns ``date``, ``ticker``, ``return``, ``next_day_return`` and
        ``volume_vs_median`` (volume that day divided by the ticker's median volume).

    Raises
    ------
    KeyError
        If ``volume`` lacks a ticker of ``close`` (raised by pandas).

    Notes
    -----
    A bad price print typically jumps and fully reverses the next day on ordinary volume.
    A real event keeps the new level and trades on unusually high volume.
    """
    returns = close.pct_change()
    sizes = pd.concat({ticker: returns[ticker].abs() for ticker in close.columns})
    rows = []
    for ticker, day in sizes.nlargest(n).index:
        position = close.index.get_loc(day)
        assert isinstance(position, int), "duplicate dates"
        series = returns[ticker]
        next_day = series.iloc[position + 1] if position + 1 < len(close) else float("nan")
        rows.append(
            {
                "date": pd.Timestamp(day).date(),
                "ticker": ticker,
                "return": float(series.iloc[position]),
                "next_day_return": float(next_day),
                "volume_vs_median": float(volume[ticker].iloc[position] / volume[ticker].median()),
            }
        )
    return pd.DataFrame(rows)


def adjustment_from_dividends(close: pd.Series, dividends: pd.Series) -> pd.Series:
    """Rebuild the dividend adjustment factor from the dividends alone.

    Parameters
    ----------
    close : pd.Series
        Close prices (already split-adjusted), sorted DatetimeIndex.
    dividends : pd.Series
        Dividend per share, indexed by ex-date; only dates inside ``close`` are used.

    Returns
    -------
    pd.Series
        Factor by which each close must be multiplied to remove the dividends paid
        later in the series; equals 1 on the last day.

    Raises
    ------
    AssertionError
        If an ex-date is not a trading day of ``close``.

    Notes
    -----
    Standard (and Yahoo's) convention: every price before an ex-date is multiplied by
    ``1 - D / C``, with ``D`` the dividend and ``C`` the close of the day before the
    ex-date. Comparing this with Yahoo's ``Adj Close / Close`` checks that the adjusted
    prices are explained by the reported dividends and nothing else.
    """
    factor = pd.Series(1.0, index=close.index)
    first, last = close.index[0], close.index[-1]
    inside = dividends.loc[(dividends.index > first) & (dividends.index <= last)]
    for ex_date, amount in inside.items():
        assert ex_date in close.index, f"ex-date {ex_date} is not a trading day"
        position = close.index.get_loc(ex_date)
        assert isinstance(position, int), "duplicate dates"
        factor.iloc[:position] *= 1 - float(amount) / float(close.iloc[position - 1])
    return factor

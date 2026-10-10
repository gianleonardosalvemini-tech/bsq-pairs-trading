"""Phase 2 spread: OLS hedge ratio estimated up to ``as_of`` and the resulting spread."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from pairs.explore import select_until

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HedgeFit:
    """Result of the regression ``A = alpha + beta * B + residual``.

    Attributes
    ----------
    alpha : float
        Intercept, in dollars of A.
    beta : float
        Hedge ratio: dollars of A per dollar move of B.
    r_squared : float
        Share of the variance of A explained by B.
    as_of : date
        Last date used in the estimate.
    n_obs : int
        Number of trading days used.
    """

    alpha: float
    beta: float
    r_squared: float
    as_of: date
    n_obs: int


def fit_hedge_ratio(prices: pd.DataFrame, a: str, b: str, as_of: date) -> HedgeFit:
    """Estimate the hedge ratio by OLS of the price of ``a`` on the price of ``b``.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices with a DatetimeIndex and one column per ticker.
    a : str
        Dependent leg (the one bought when going long the spread).
    b : str
        Hedge leg.
    as_of : date
        Last date allowed in the estimate (inclusive).

    Returns
    -------
    HedgeFit
        Intercept, hedge ratio, R squared and the sample it comes from.

    Raises
    ------
    AssertionError
        If data after ``as_of`` reaches the regression or the sample has missing values.

    Notes
    -----
    The regression is on price levels, as the brief prescribes, not on log prices or
    returns. Filtering to ``as_of`` is the first operation, so calling it with
    ``as_of=PAIR.in_sample_end`` can never see out-of-sample data. Ordinary least squares
    is solved with ``numpy.linalg.lstsq`` (same estimate as statsmodels, fully typed).
    """
    sample = select_until(prices[[a, b]], as_of)
    assert sample.index.max() <= pd.Timestamp(as_of), "data after as_of in the fit"
    assert not sample.isna().any().any(), "missing prices in the fit sample"
    y = sample[a].to_numpy(dtype=float)
    x = sample[b].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    (alpha, beta), *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - (alpha + beta * x)
    r_squared = 1.0 - float(residual.var()) / float(y.var())
    fit = HedgeFit(float(alpha), float(beta), r_squared, as_of, len(sample))
    logger.info("Hedge fit %s on %s up to %s: %s", a, b, as_of, fit)
    return fit


def build_spread(prices: pd.DataFrame, a: str, b: str, beta: float) -> pd.Series:
    """Build the spread ``A - beta * B`` on every date of ``prices``.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices with a DatetimeIndex and one column per ticker.
    a : str
        Dependent leg.
    b : str
        Hedge leg.
    beta : float
        Hedge ratio, estimated beforehand with :func:`fit_hedge_ratio`.

    Returns
    -------
    pd.Series
        The spread in dollars, named ``"spread"``.

    Notes
    -----
    The intercept is left out: the z-score subtracts a rolling mean, so a constant shift
    does not change the signal. The spread can be built on dates after the fit (the
    out-of-sample) because ``beta`` is a number frozen at ``as_of``.
    """
    spread = (prices[a] - beta * prices[b]).rename("spread")
    return spread


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    """Measure the spread's distance from its rolling average, in rolling standard deviations.

    Parameters
    ----------
    spread : pd.Series
        Output of :func:`build_spread`.
    window : int
        Trading days in the rolling mean and standard deviation (today included).

    Returns
    -------
    pd.Series
        z-score named ``"z"``; NaN until the window is full.

    Raises
    ------
    AssertionError
        If ``window`` is shorter than two days.

    Notes
    -----
    ``z_t = (S_t - mean(S_{t-N+1..t})) / std(S_{t-N+1..t})``: only today and the past enter,
    so changing a future price cannot change today's z (tested). The brief's wording is
    "how many standard deviations away from its own rolling average it currently is".
    """
    assert window >= 2, "a rolling standard deviation needs at least two days"
    rolling = spread.rolling(window, min_periods=window)
    return ((spread - rolling.mean()) / rolling.std()).rename("z")


def mean_crossings_per_year(spread: pd.Series) -> pd.Series:
    """Count how many times per calendar year the spread crosses its own sample mean.

    Parameters
    ----------
    spread : pd.Series
        Spread on the period being checked (e.g. the in-sample period only).

    Returns
    -------
    pd.Series
        Number of crossings, indexed by year (years without crossings included).

    Notes
    -----
    The brief's basic check: a spread that "keeps crossing back over its own historical
    average" mean-reverts; one that stays on one side for years drifts. The mean is the
    one of the series passed in, so pass the in-sample spread only.
    """
    side = np.sign(spread - spread.mean())
    crossed = (side != side.shift(1)).iloc[1:]
    years = pd.DatetimeIndex(crossed.index).year
    counts: pd.Series = crossed.astype(int).groupby(years).sum()
    return counts.rename("crossings")


def half_life(spread: pd.Series) -> float:
    """Estimate in how many days a deviation of the spread halves, from an AR(1) fit.

    Parameters
    ----------
    spread : pd.Series
        Spread on the estimation period.

    Returns
    -------
    float
        Half-life in trading days (``inf`` if the spread does not pull back).

    Notes
    -----
    Regress the daily change on yesterday's level: ``dS_t = c + phi * S_{t-1} + e``.
    A negative ``phi`` means pull-back toward the mean; half-life = ``-ln 2 / phi``.
    A rough measure, used to choose the grid of rolling windows.
    """
    lagged = spread.shift(1).iloc[1:].to_numpy(dtype=float)
    change = spread.diff().iloc[1:].to_numpy(dtype=float)
    phi = float(np.polyfit(lagged, change, 1)[0])
    return -math.log(2) / phi if phi < 0 else math.inf

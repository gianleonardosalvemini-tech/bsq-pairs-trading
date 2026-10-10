"""Phase 2 spread: OLS hedge ratio estimated up to ``as_of`` and the resulting spread."""

from __future__ import annotations

import logging
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

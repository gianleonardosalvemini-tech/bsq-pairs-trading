"""Extension: a hedge ratio that moves day by day, estimated with a Kalman filter."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pairs.backtest import pair_shares, positions_from_zscore, run_backtest
from pairs.spread import rolling_zscore

logger = logging.getLogger(__name__)


def kalman_hedge(
    prices: pd.DataFrame,
    a: str,
    b: str,
    delta: float,
    obs_var: float,
    alpha0: float,
    beta0: float,
) -> pd.DataFrame:
    """Track the intercept and hedge ratio of ``A = alpha + beta * B`` day by day.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices from the first day the filter runs (no missing values).
    a, b : str
        Dependent leg and hedge leg.
    delta : float
        Each day alpha and beta take a random step with variance ``delta / (1 - delta)``.
    obs_var : float
        Variance of the noise of A around ``alpha + beta * B``, in dollars squared.
    alpha0, beta0 : float
        Starting values (an OLS fit on the days before ``prices`` starts).

    Returns
    -------
    pd.DataFrame
        Columns ``alpha`` and ``beta`` (estimates with data up to the day before),
        ``spread`` (``A - alpha - beta * B`` with those estimates) and ``spread_var``
        (its predicted variance).

    Raises
    ------
    AssertionError
        If a price is missing or a parameter is out of range.

    Notes
    -----
    The state (alpha, beta) is a random walk and the observation is the price of A, with
    ``[1, B]`` as the observation row. For each day the filter first predicts with the
    estimates of the day before (that prediction gives the spread used for trading), then
    updates them with the day's prices. So nothing on day t uses prices after t-1 except
    the day's own price in the spread. The starting uncertainty is zero: the first
    estimates are the OLS ones and move only through the daily steps.
    """
    assert 0 < delta < 1 and obs_var > 0, "delta in (0, 1), positive observation variance"
    assert not prices[[a, b]].isna().any().any(), "missing prices"
    y = prices[a].to_numpy(dtype=float)
    x = prices[b].to_numpy(dtype=float)
    state = np.array([alpha0, beta0])
    cov = np.zeros((2, 2))
    step = delta / (1 - delta) * np.eye(2)
    out = np.empty((len(y), 4))
    for t in range(len(y)):
        cov = cov + step  # prediction: the state keeps its value, its uncertainty grows
        h = np.array([1.0, x[t]])
        error = y[t] - h @ state
        var = float(h @ cov @ h) + obs_var
        out[t] = (state[0], state[1], error, var)
        gain = cov @ h / var
        state = state + gain * error
        cov = cov - np.outer(gain, h @ cov)
    logger.info("Kalman filter on %d days, last beta %.3f", len(y), state[1])
    return pd.DataFrame(out, index=prices.index, columns=["alpha", "beta", "spread", "spread_var"])


def trade_hedge(position: pd.Series, beta: pd.Series) -> pd.Series:
    """Hedge ratio of each trade: the one of the day the trade opens, kept until it closes.

    Parameters
    ----------
    position : pd.Series
        +1 / -1 / 0 positions from :func:`pairs.backtest.positions_from_zscore`.
    beta : pd.Series
        Hedge ratio available on each day (same index).

    Returns
    -------
    pd.Series
        Hedge ratio in use on each day (0 when flat).

    Notes
    -----
    A trade starts when the position becomes non-zero or changes sign (as in
    :func:`pairs.backtest.summarize`). Keeping the ratio fixed during a trade means the
    shares do not change every day, so there is nothing to rebalance.
    """
    assert position.index.equals(beta.index), "aligned series"
    starts = position.ne(0) & position.ne(position.shift(1).fillna(0.0))
    return beta.where(starts).ffill().where(position.ne(0), 0.0).fillna(0.0).rename("hedge")


def run_kalman_strategy(
    prices: pd.DataFrame,
    filt: pd.DataFrame,
    a: str,
    b: str,
    entry: float,
    window: int,
    start: pd.Timestamp,
    exit: float = 0.0,
) -> pd.DataFrame:
    """Phase 2 signal on the Kalman spread, with the hedge ratio fixed at each trade's entry.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices of the two legs, on the days to trade (index within ``filt``).
    filt : pd.DataFrame
        Output of :func:`kalman_hedge`.
    a, b : str
        Legs of the spread.
    entry, window, exit : float, int, float
        Frozen Phase 2 thresholds and z-score window.
    start : pd.Timestamp
        First day positions may be opened.

    Returns
    -------
    pd.DataFrame
        Columns ``spread``, ``z``, ``position``, ``hedge``, ``pnl``, ``gross``, ``ret``.

    Notes
    -----
    Same steps as :func:`pairs.backtest.run_zscore_strategy`, with the Kalman spread in
    place of ``A - beta * B`` and a hedge ratio that changes from trade to trade. The
    z-score may use the filter's spread before ``start`` (the past), positions may not.
    """
    filt = filt.loc[prices.index[0] : prices.index[-1]]
    assert filt.index.equals(prices.index), "prices must cover the filter's days"
    z = rolling_zscore(filt["spread"], window)
    assert z.loc[z.index >= start].notna().all(), "z must be defined from start onwards"
    position = positions_from_zscore(z.where(z.index >= start), entry, exit)
    hedge = trade_hedge(position, filt["beta"])
    result = run_backtest(prices[[a, b]], pair_shares(position, a, b, hedge))
    return pd.concat([filt["spread"], z, position, hedge, result], axis=1)

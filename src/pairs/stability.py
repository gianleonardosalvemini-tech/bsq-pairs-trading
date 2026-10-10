"""Phase 3 stability checks: rolling hedge ratio, rolling ADF test and weak-relationship labels."""

from __future__ import annotations

import logging
import math
from datetime import date

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from pairs.explore import select_until

logger = logging.getLogger(__name__)


def rolling_hedge_ratio(prices: pd.DataFrame, a: str, b: str, window: int) -> pd.Series:
    """Re-estimate the OLS hedge ratio of ``a`` on ``b`` every day on the last ``window`` days.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices with a DatetimeIndex and one column per ticker.
    a : str
        Dependent leg.
    b : str
        Hedge leg.
    window : int
        Trading days in each regression (today included).

    Returns
    -------
    pd.Series
        Hedge ratio named ``"beta"``; NaN until the window is full.

    Raises
    ------
    AssertionError
        If ``window`` is shorter than three days.

    Notes
    -----
    OLS with an intercept has slope ``cov(A, B) / var(B)``, so the rolling slope is the
    rolling covariance over the rolling variance: the same number as
    :func:`pairs.spread.fit_hedge_ratio` on each window, without a Python loop. Only today
    and the past enter. Descriptive: it shows how far the frozen hedge ratio is from the
    one each period would have chosen; it is never used to trade.
    """
    assert window >= 3, "a regression needs at least three days"
    cov = prices[a].rolling(window, min_periods=window).cov(prices[b])
    var = prices[b].rolling(window, min_periods=window).var()
    return (cov / var).rename("beta")


def rolling_adf_pvalue(spread: pd.Series, window: int, step: int = 5) -> pd.Series:
    """Augmented Dickey-Fuller p-value of the spread on the last ``window`` days.

    Parameters
    ----------
    spread : pd.Series
        Spread with a frozen hedge ratio (see :func:`pairs.spread.build_spread`).
    window : int
        Trading days in each test (the day of the estimate included).
    step : int, default 5
        Re-run the test every ``step`` trading days; days in between carry the last value.

    Returns
    -------
    pd.Series
        p-value named ``"adf_pvalue"``, on the index of ``spread``; NaN until the first
        full window. Low = the window looks mean-reverting; high = it looks like a drift.

    Raises
    ------
    AssertionError
        If the window is too short for the test or ``step`` is not positive.

    Notes
    -----
    Test with a constant and the lag length chosen by AIC (statsmodels defaults). The value
    on day t uses days ``t-window+1 .. t`` only. Running it weekly (``step=5``) instead of
    daily cuts the cost five times; carrying the last value forward only uses the past.
    The p-values are the plain Dickey-Fuller ones: the frozen hedge ratio was fitted on the
    in-sample period, so in-sample windows look somewhat more stationary than they are.
    """
    assert window >= 50 and step >= 1, "window too short for an ADF test, or bad step"
    values = spread.to_numpy(dtype=float)
    assert not np.isnan(values).any(), "missing values in the spread"
    ends = list(range(window - 1, len(values), step))
    if ends and ends[-1] != len(values) - 1:
        ends.append(len(values) - 1)
    pvalues = [
        float(adfuller(values[end - window + 1 : end + 1], autolag="AIC", result_object=False)[1])
        for end in ends
    ]
    computed = pd.Series(pvalues, index=spread.index[ends])
    logger.info("Rolling ADF: %d tests, window %d, step %d", len(ends), window, step)
    return computed.reindex(spread.index).ffill().rename("adf_pvalue")


def pvalue_cutoff(pvalues: pd.Series, as_of: date, share: float) -> float:
    """p-value above which the ``share`` highest in-sample values lie.

    Parameters
    ----------
    pvalues : pd.Series
        Output of :func:`rolling_adf_pvalue`.
    as_of : date
        Last date allowed in the calibration (the end of the in-sample period).
    share : float
        Share of in-sample days to label as weak, in ``(0, 1)``.

    Returns
    -------
    float
        The ``1 - share`` quantile of the p-values up to ``as_of``.

    Raises
    ------
    AssertionError
        If data after ``as_of`` reaches the calibration or ``share`` is outside ``(0, 1)``.

    Notes
    -----
    A relative threshold ("less mean-reverting than usual") instead of the absolute 0.10
    ("not mean-reverting"), like the brief's simple method ("well below its historical
    norm"). The cutoff is frozen at ``as_of`` and reused unchanged after it.
    """
    assert 0 < share < 1, "share must be in (0, 1)"
    sample = select_until(pvalues.dropna().to_frame(), as_of)["adf_pvalue"]
    assert sample.index.max() <= pd.Timestamp(as_of), "data after as_of in the cutoff"
    return float(sample.quantile(1 - share))


def weak_labels(pvalues: pd.Series, cutoff: float) -> pd.Series:
    """Label each day weak (1) or normal (0), using only information up to the day before.

    Parameters
    ----------
    pvalues : pd.Series
        Output of :func:`rolling_adf_pvalue`.
    cutoff : float
        A p-value above it means "weak relationship".

    Returns
    -------
    pd.Series
        1.0 weak, 0.0 normal, NaN while no p-value is available; named ``"weak"``.

    Notes
    -----
    The p-value is shifted by one day: the label of day t is known at the close of t-1,
    before the signal of day t is computed, so it could have been used to decide a trade.
    """
    weak = pvalues.gt(cutoff).astype(float).where(pvalues.notna())
    return weak.shift(1).rename("weak")


def trade_returns(run: pd.DataFrame) -> pd.DataFrame:
    """One row per trade of a backtest: entry day, holding days, direction and return.

    Parameters
    ----------
    run : pd.DataFrame
        Output of :func:`pairs.backtest.run_zscore_strategy`, cut to the evaluation period.

    Returns
    -------
    pd.DataFrame
        Columns ``entry`` (close at which the trade is decided), ``days`` (days held),
        ``direction`` (+1 long, -1 short the spread) and ``ret`` (compounded return).

    Raises
    ------
    AssertionError
        If the trade returns do not compound back to the backtest's total return.

    Notes
    -----
    A trade starts on each day the position becomes non-zero or changes sign (the same rule
    as :func:`pairs.backtest.summarize`). Its return compounds the daily returns earned
    while it is held, from the day after the entry to the exit day. A trade still open on
    the last day is counted with the return earned so far.
    """
    position = run["position"]
    starts = position.ne(0) & position.ne(position.shift(1).fillna(0.0))
    trade_id = starts.cumsum().where(position.ne(0))
    held_id = trade_id.shift(1)
    held = held_id.notna()
    ret = (1.0 + run["ret"][held]).groupby(held_id[held]).prod() - 1.0
    days = held_id[held].groupby(held_id[held]).size()
    entries = position.index[starts]
    table = pd.DataFrame(
        {
            "entry": entries,
            "days": days.reindex(range(1, len(entries) + 1)).fillna(0).to_numpy(dtype=int),
            "direction": position[starts].to_numpy(dtype=int),
            "ret": ret.reindex(range(1, len(entries) + 1)).fillna(0.0).to_numpy(dtype=float),
        }
    )
    total = float(np.prod(1.0 + run["ret"].to_numpy(dtype=float)))
    assert math.isclose(
        float(np.prod(1.0 + table["ret"].to_numpy(dtype=float))), total, rel_tol=1e-9
    ), "trades miss P&L"
    return table


def compare_groups(values: pd.Series, weak: pd.Series) -> dict[str, float]:
    """Compare the mean of ``values`` on weak and normal observations.

    Parameters
    ----------
    values : pd.Series
        Trade or daily returns.
    weak : pd.Series
        Label aligned with ``values``: 1 weak, 0 normal, NaN dropped.

    Returns
    -------
    dict of str to float
        ``n_weak``, ``n_normal``, ``mean_weak``, ``mean_normal``, ``win_weak``,
        ``win_normal`` (share of positive values) and ``welch_t`` of the difference
        weak minus normal (NaN if a group has fewer than two observations).

    Raises
    ------
    AssertionError
        If the two series are not aligned.

    Notes
    -----
    Welch's t allows different variances in the two groups. A negative ``welch_t`` means
    the weak observations earn less, which is the brief's hypothesis. Daily returns are
    autocorrelated within a trade, so on daily data the t overstates the evidence.
    """
    assert values.index.equals(weak.index), "values and labels must be aligned"
    keep = weak.notna()
    group_weak = values[keep & weak.eq(1)]
    group_normal = values[keep & weak.eq(0)]
    n_w, n_n = len(group_weak), len(group_normal)
    welch = math.nan
    if n_w > 1 and n_n > 1:
        se = math.sqrt(
            float(np.var(group_weak.to_numpy(dtype=float), ddof=1)) / n_w
            + float(np.var(group_normal.to_numpy(dtype=float), ddof=1)) / n_n
        )
        welch = (float(group_weak.mean()) - float(group_normal.mean())) / se
    return {
        "n_weak": float(n_w),
        "n_normal": float(n_n),
        "mean_weak": float(group_weak.mean()) if n_w else math.nan,
        "mean_normal": float(group_normal.mean()) if n_n else math.nan,
        "win_weak": float(group_weak.gt(0).mean()) if n_w else math.nan,
        "win_normal": float(group_normal.gt(0).mean()) if n_n else math.nan,
        "welch_t": welch,
    }

"""Phase 2 backtest: z-score positions, a generic multi-leg P&L, and summary metrics."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def positions_from_zscore(z: pd.Series, entry: float, exit: float = 0.0) -> pd.Series:
    """Turn a z-score into a spread position: +1 long, -1 short, 0 flat.

    Parameters
    ----------
    z : pd.Series
        Output of :func:`pairs.spread.rolling_zscore`; NaN days keep the current state.
    entry : float
        Open long when ``z < -entry``, short when ``z > entry``.
    exit : float, default 0.0
        Close a long when ``z >= -exit`` and a short when ``z <= exit``.

    Returns
    -------
    pd.Series
        Position decided at the close of each day, named ``"position"``.

    Raises
    ------
    AssertionError
        If ``exit`` is not in ``[0, entry)``.

    Notes
    -----
    An explicit day-by-day state machine, to read like the brief: long the spread when z
    is very negative, short when very positive, close when it reverts back toward zero.
    If z jumps past the opposite threshold on the closing day, the position flips.
    """
    assert 0 <= exit < entry, "exit must lie in [0, entry)"
    state = 0
    out = np.zeros(len(z))
    for i, value in enumerate(z.to_numpy(dtype=float)):
        if not math.isnan(value):
            if (state == 1 and value >= -exit) or (state == -1 and value <= exit):
                state = 0
            if state == 0 and value < -entry:
                state = 1
            elif state == 0 and value > entry:
                state = -1
        out[i] = state
    return pd.Series(out, index=z.index, name="position")


def pair_shares(position: pd.Series, a: str, b: str, beta: float) -> pd.DataFrame:
    """Shares held in each leg for a spread position: ``+1 A, -beta B`` per unit.

    Parameters
    ----------
    position : pd.Series
        Output of :func:`positions_from_zscore`.
    a : str
        Leg bought when long the spread.
    b : str
        Leg sold (``beta`` shares) when long the spread.
    beta : float
        Hedge ratio, a number or a series aligned with ``position``.

    Returns
    -------
    pd.DataFrame
        One column per leg, shares decided at each close.

    Notes
    -----
    Kept separate from :func:`run_backtest` so that the same P&L code serves a moving
    hedge ratio or a basket: only the share table changes.
    """
    return pd.DataFrame({a: position, b: -beta * position}, index=position.index)


def run_backtest(prices: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Daily P&L and return of holding ``shares`` decided at the previous close.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, one column per leg.
    shares : pd.DataFrame
        Shares per leg decided at each close (same index and columns as ``prices``).

    Returns
    -------
    pd.DataFrame
        Columns ``pnl`` (dollars), ``gross`` (dollars at work) and ``ret`` (daily return).

    Raises
    ------
    AssertionError
        If the indexes or columns differ, or a held position meets a missing price.

    Notes
    -----
    Shares decided at the close of t are held from t to t+1 (``shift(1)``), so a signal
    never earns the move of the day it was computed on. Return = P&L / gross exposure at
    the previous close (sum of absolute leg values); flat days return 0. Adjusted prices
    include dividends, so the short leg pays them. No transaction costs (not part of the
    brief's Phase 2).
    """
    assert prices.index.equals(shares.index), "prices and shares must share the index"
    assert list(prices.columns) == list(shares.columns), "same legs in the same order"
    held = shares.shift(1).fillna(0.0)
    pnl = (held * prices.diff()).sum(axis=1, min_count=1).fillna(0.0)
    gross = (held.abs() * prices.shift(1)).sum(axis=1).fillna(0.0)
    assert not (held.ne(0) & prices.isna()).any().any(), "position held on a missing price"
    ret = pnl.where(gross > 0, 0.0) / gross.where(gross > 0, 1.0)
    return pd.DataFrame({"pnl": pnl, "gross": gross, "ret": ret})


def summarize(ret: pd.Series, position: pd.Series) -> dict[str, float]:
    """Compute the performance metrics of a daily return series.

    Parameters
    ----------
    ret : pd.Series
        Daily returns from :func:`run_backtest`, already cut to the evaluation period.
    position : pd.Series
        Positions on the same days, for the trade statistics.

    Returns
    -------
    dict of str to float
        ``ann_return``, ``ann_vol``, ``sharpe``, ``t_stat``, ``max_drawdown``,
        ``n_trades``, ``pct_in_market``, ``avg_days_per_trade``.

    Raises
    ------
    AssertionError
        If the two series are not aligned or are empty.

    Notes
    -----
    Every day counts, flat days included: idle capital is still committed, and
    dropping flat days would inflate the Sharpe. Sharpe = mean / std * sqrt(252) with no
    risk-free rate (a long-short spread is self-financing). ``t_stat`` tests the mean
    daily return against zero. A trade starts on each day the position becomes non-zero
    or changes sign. Drawdown is on the compounded equity curve.
    """
    assert ret.index.equals(position.index) and len(ret) > 1, "aligned, non-empty series"
    mean, std = float(ret.mean()), float(ret.std())
    equity = (1.0 + ret).cumprod()
    drawdown = float((equity / equity.cummax() - 1.0).min())
    starts = position.ne(0) & position.ne(position.shift(1).fillna(0.0))
    n_trades = int(starts.sum())
    days_in = int(position.ne(0).sum())
    return {
        "ann_return": mean * TRADING_DAYS,
        "ann_vol": std * math.sqrt(TRADING_DAYS),
        "sharpe": mean / std * math.sqrt(TRADING_DAYS) if std > 0 else 0.0,
        "t_stat": mean / (std / math.sqrt(len(ret))) if std > 0 else 0.0,
        "max_drawdown": drawdown,
        "n_trades": float(n_trades),
        "pct_in_market": days_in / len(position),
        "avg_days_per_trade": days_in / n_trades if n_trades else 0.0,
    }

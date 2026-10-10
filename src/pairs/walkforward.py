"""Extension: transaction costs and a rolling walk-forward of the Phase 2 procedure."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pairs.backtest import (
    pair_shares,
    positions_from_zscore,
    run_backtest,
    run_zscore_strategy,
    select_stable_cell,
    summarize,
)
from pairs.config import SignalGrid
from pairs.spread import build_spread, fit_hedge_ratio, rolling_zscore

logger = logging.getLogger(__name__)

BPS = 1e-4


def cost_returns(prices: pd.DataFrame, shares: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Daily cost of trading, as a fraction of the capital at work.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, one column per leg.
    shares : pd.DataFrame
        Shares per leg decided at each close (same index and columns as ``prices``).
    cost_bps : float
        One-way cost per leg, in basis points of the dollar value traded.

    Returns
    -------
    pd.Series
        Cost on each day, to subtract from that day's return (0 on days without trades).

    Raises
    ------
    AssertionError
        If the tables are not aligned or the cost is negative.

    Notes
    -----
    A trade happens at the close of day t when the shares change. Its cost is
    ``cost_bps`` of the dollar value traded on each leg (|change in shares| x price at t),
    divided by the larger of the position values before and after the trade at the same
    price, so opening or closing a position costs about ``cost_bps`` of the capital on
    each leg and a round trip about twice that. A flip from long to short trades twice
    the shares and pays twice. The cost is charged on day t itself.
    """
    assert prices.index.equals(shares.index), "prices and shares must share the index"
    assert list(prices.columns) == list(shares.columns), "same legs in the same order"
    assert cost_bps >= 0, "cost must be non-negative"
    traded = (shares.diff().fillna(shares).abs() * prices).sum(axis=1)
    before = (shares.shift(1).fillna(0.0).abs() * prices).sum(axis=1)
    after = (shares.abs() * prices).sum(axis=1)
    reference = pd.concat([before, after], axis=1).max(axis=1)
    cost = (cost_bps * BPS * traded).where(reference > 0, 0.0) / reference.where(reference > 0, 1.0)
    return cost.rename("cost")


def run_with_costs(
    prices: pd.DataFrame,
    a: str,
    b: str,
    beta: float,
    entry: float,
    window: int,
    start: pd.Timestamp,
    cost_bps: float,
    exit: float = 0.0,
) -> pd.DataFrame:
    """Phase 2 strategy with transaction costs subtracted from the daily return.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices of the two legs.
    a, b : str
        Legs of the spread ``A - beta * B``.
    beta : float
        Hedge ratio.
    entry : float
        Entry threshold on |z|.
    window : int
        Rolling window of the z-score, in trading days.
    start : pd.Timestamp
        First day positions may be opened.
    cost_bps : float
        One-way cost per leg, in basis points (see :func:`cost_returns`).
    exit : float, default 0.0
        Exit level.

    Returns
    -------
    pd.DataFrame
        The columns of :func:`pairs.backtest.run_zscore_strategy` plus ``cost`` and
        ``ret_gross``; ``ret`` is net of costs.

    Notes
    -----
    With ``cost_bps = 0`` the result equals :func:`pairs.backtest.run_zscore_strategy`.
    """
    out = run_zscore_strategy(prices, a, b, beta, entry, window, start, exit)
    shares = pair_shares(out["position"], a, b, beta)
    cost = cost_returns(prices[[a, b]], shares, cost_bps)
    out["ret_gross"] = out["ret"]
    out["cost"] = cost
    out["ret"] = out["ret_gross"] - cost
    return out


@dataclass(frozen=True)
class WalkForwardYear:
    """What the walk-forward chose and earned in one test year."""

    year: int
    beta: float
    entry: float
    window: int
    sharpe: float
    ret: float


def walk_forward(
    prices: pd.DataFrame,
    a: str,
    b: str,
    grid: SignalGrid,
    first_test_year: int,
    last_test_year: int,
    train_years: int,
    min_trades: int,
    cost_bps: float,
) -> tuple[pd.DataFrame, list[WalkForwardYear]]:
    """Re-run the whole Phase 2 procedure every year on the past, then trade the next year.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices of the two legs, covering every training and test year.
    a, b : str
        Legs of the spread ``A - beta * B``.
    grid : SignalGrid
        Thresholds, windows and selection rule of Phase 2 (unchanged).
    first_test_year, last_test_year : int
        Calendar years traded out-of-sample, one at a time.
    train_years : int
        Length of the training window: the ``train_years`` calendar years before each
        test year.
    min_trades : int
        Fewest training trades for a cell to be selectable (Phase 2 used 20 over 18
        years; the training window is shorter).
    cost_bps : float
        One-way cost per leg, in basis points, in training and in test.

    Returns
    -------
    tuple of (pd.DataFrame, list of WalkForwardYear)
        Daily ``ret`` (net), ``cost`` and ``position`` of the stitched test years, and
        what was chosen each year.

    Raises
    ------
    AssertionError
        If a training window is too short for the longest rolling window.

    Notes
    -----
    Each step uses only data up to 31 December of the year before the test year: the
    hedge ratio is the OLS of Phase 2 with ``as_of`` that day, the 25 cells are
    backtested on the training years and the stable-zone rule picks one. The test year
    starts flat; its z-score may use the last days of the previous year (the past). Any
    position still open on the last day of the test year is closed there and pays its
    cost, so no trade spills into the next year.
    """
    days: list[pd.DataFrame] = []
    chosen: list[WalkForwardYear] = []
    for year in range(first_test_year, last_test_year + 1):
        as_of = pd.Timestamp(f"{year - 1}-12-31")
        train = prices.loc[f"{year - train_years}" : as_of]
        assert len(train) > max(grid.windows) + 250, "training window too short"
        beta = fit_hedge_ratio(train, a, b, as_of.date()).beta
        train_start = train.index[max(grid.windows) - 1]
        rows = []
        for entry in grid.entry_thresholds:
            for window in grid.windows:
                run = run_with_costs(
                    train, a, b, beta, entry, window, train_start, cost_bps, grid.exit_threshold
                ).loc[train_start:]
                stats = summarize(run["ret"], run["position"])
                rows.append(
                    {
                        "entry": entry,
                        "window": window,
                        "sharpe": stats["sharpe"],
                        "n_trades": stats["n_trades"],
                    }
                )
        table = pd.DataFrame(rows)
        entry, window = select_stable_cell(
            table.pivot(index="entry", columns="window", values="sharpe"),
            table.pivot(index="entry", columns="window", values="n_trades"),
            min_trades,
        )
        history = prices.loc[f"{year - train_years}" : f"{year}-12-31"]
        test_start = history.index[pd.DatetimeIndex(history.index).year == year][0]
        z = rolling_zscore(build_spread(history, a, b, beta), window)
        position = positions_from_zscore(z.where(z.index >= test_start), entry, grid.exit_threshold)
        position.iloc[-1] = 0.0  # close at the end of the test year
        shares = pair_shares(position, a, b, beta)
        result = run_backtest(history[[a, b]], shares)
        cost = cost_returns(history[[a, b]], shares, cost_bps)
        test = pd.DataFrame({"ret": result["ret"] - cost, "cost": cost, "position": position}).loc[
            test_start:
        ]
        days.append(test)
        stats = summarize(test["ret"], test["position"])
        year_ret = float(np.prod(1.0 + test["ret"].to_numpy()) - 1.0)
        chosen.append(WalkForwardYear(year, beta, entry, window, stats["sharpe"], year_ret))
        logger.info("Walk-forward %s: beta %.3f, entry %s, window %s", year, beta, entry, window)
    return pd.concat(days), chosen

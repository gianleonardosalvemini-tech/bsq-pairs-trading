"""Phase 5 simulation: joint block bootstrap of both legs and an Ornstein-Uhlenbeck spread."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from pairs.backtest import (
    TRADING_DAYS,
    positions_from_zscore,
    run_zscore_strategy,
    select_stable_cell,
    summarize,
)
from pairs.config import SignalGrid
from pairs.explore import select_until
from pairs.spread import fit_hedge_ratio, rolling_zscore

logger = logging.getLogger(__name__)


def block_bootstrap_indices(
    n_obs: int, block: int, n_out: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw row positions for a moving block bootstrap with replacement.

    Parameters
    ----------
    n_obs : int
        Number of historical rows to resample from.
    block : int
        Consecutive rows per block.
    n_out : int
        Length of the synthetic series.
    rng : np.random.Generator
        Source of randomness (seeded by the caller, for reproducibility).

    Returns
    -------
    np.ndarray
        ``n_out`` integer positions in ``[0, n_obs)``.

    Raises
    ------
    AssertionError
        If a block does not fit in the history or the lengths are not positive.

    Notes
    -----
    Each block starts at a uniformly drawn row and takes the next ``block`` rows; blocks
    are stitched together and the last one is cut to reach ``n_out``. The same positions
    are used for every asset, which is what keeps the assets' co-movement.
    """
    assert 1 <= block <= n_obs and n_out >= 1, "block must fit in the history"
    n_blocks = math.ceil(n_out / block)
    starts = rng.integers(0, n_obs - block + 1, size=n_blocks)
    positions = (starts[:, None] + np.arange(block)[None, :]).ravel()
    return positions[:n_out]


def simulate_joint_prices(
    prices: pd.DataFrame, block: int, rng: np.random.Generator, as_of: date
) -> pd.DataFrame:
    """Build one synthetic joint price path from the historical daily returns up to ``as_of``.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, one column per leg, no missing values.
    block : int
        Block length in trading days.
    rng : np.random.Generator
        Source of randomness.
    as_of : date
        Last historical date whose returns may be resampled (inclusive).

    Returns
    -------
    pd.DataFrame
        Synthetic prices on the same calendar and with the same first-day prices as
        ``prices`` cut at ``as_of``.

    Raises
    ------
    AssertionError
        If data after ``as_of`` survives the filter or prices are missing.

    Notes
    -----
    Simple daily returns of all legs are resampled with the same blocks (joint bootstrap)
    and compounded from the real first-day prices. The path keeps each day's
    cross-sectional co-movement and short-run patterns inside a block (volatility
    clustering, short reversals), but blocks are drawn independently, so any pull-back of
    the spread on scales longer than a block is destroyed.
    """
    history = select_until(prices, as_of)
    assert history.index.max() <= pd.Timestamp(as_of), "data after as_of in the bootstrap"
    assert not history.isna().any().any(), "missing prices in the bootstrap sample"
    returns = history.pct_change().iloc[1:].to_numpy(dtype=float)
    rows = block_bootstrap_indices(len(returns), block, len(returns), rng)
    growth = np.vstack([np.ones(history.shape[1]), np.cumprod(1.0 + returns[rows], axis=0)])
    return pd.DataFrame(
        history.iloc[0].to_numpy(dtype=float) * growth,
        index=history.index,
        columns=history.columns,
    )


def frozen_rules_on_path(
    prices: pd.DataFrame, a: str, b: str, entry: float, window: int, start: int, as_of: date
) -> dict[str, float]:
    """Re-run the Phase 2 procedure with frozen rules on one (synthetic) price path.

    Parameters
    ----------
    prices : pd.DataFrame
        Prices of the two legs (one simulated path).
    a, b : str
        Legs of the spread ``A - beta * B``.
    entry : float
        Entry threshold on |z|.
    window : int
        z-score window in trading days.
    start : int
        Position of the first evaluated day (Phase 2: day 250, when the longest window fills).
    as_of : date
        Last date of the hedge-ratio fit, as in Phase 2.

    Returns
    -------
    dict of str to float
        Output of :func:`pairs.backtest.summarize` plus the path's ``beta``.

    Notes
    -----
    The hedge ratio is re-estimated by OLS on the path, exactly as Phase 2 did on the real
    in-sample: the procedure is repeated, not the number 0.388 (which has no meaning on
    other prices).
    """
    beta = fit_hedge_ratio(prices, a, b, as_of).beta
    first = prices.index[start]
    out = run_zscore_strategy(prices, a, b, beta, entry, window, first).loc[first:]
    return {**summarize(out["ret"], out["position"]), "beta": beta}


def selected_rules_on_path(
    prices: pd.DataFrame, a: str, b: str, grid: SignalGrid, as_of: date
) -> dict[str, float]:
    """Repeat the whole Phase 2 tuning on one path: 25-cell grid, then the stable-zone rule.

    Parameters
    ----------
    prices : pd.DataFrame
        Prices of the two legs (one simulated path).
    a, b : str
        Legs of the spread.
    grid : SignalGrid
        The pre-registered grid and selection rule of Phase 2.
    as_of : date
        Last date of the hedge-ratio fit.

    Returns
    -------
    dict of str to float
        ``sharpe`` and ``n_trades`` of the selected cell, its ``entry`` and ``window``
        (``sharpe`` is NaN if no interior cell reaches ``grid.min_trades``).

    Notes
    -----
    The real 0.35 was the winner of a selection among 25 cells; comparing it with the
    winner of the same selection on each path accounts for that choice.
    """
    beta = fit_hedge_ratio(prices, a, b, as_of).beta
    first = prices.index[max(grid.windows) - 1]
    sharpe = pd.DataFrame(
        index=list(grid.entry_thresholds), columns=list(grid.windows), dtype=float
    )
    trades = sharpe.copy()
    for entry in grid.entry_thresholds:
        for window in grid.windows:
            out = run_zscore_strategy(prices, a, b, beta, entry, window, first, grid.exit_threshold)
            stats = summarize(out.loc[first:, "ret"], out.loc[first:, "position"])
            sharpe.loc[entry, window] = stats["sharpe"]
            trades.loc[entry, window] = stats["n_trades"]
    if not (trades.iloc[1:-1, 1:-1] >= grid.min_trades).any().any():
        return {"sharpe": math.nan, "n_trades": math.nan, "entry": math.nan, "window": math.nan}
    entry, window = select_stable_cell(sharpe, trades, grid.min_trades)
    row = list(sharpe.index).index(entry)
    col = [int(w) for w in sharpe.columns].index(window)
    return {
        "sharpe": float(sharpe.to_numpy(dtype=float)[row, col]),
        "n_trades": float(trades.to_numpy(dtype=float)[row, col]),
        "entry": entry,
        "window": float(window),
    }


@dataclass(frozen=True)
class OUFit:
    """Ornstein-Uhlenbeck parameters of a spread, from its daily AR(1) regression.

    Attributes
    ----------
    mu : float
        Long-run mean of the spread (dollars).
    phi : float
        Daily persistence ``exp(-kappa)``; the pull-back speed is ``kappa = -ln(phi)``.
    sigma : float
        Standard deviation of the daily shock (dollars).
    half_life : float
        Trading days for a deviation to halve, ``ln 2 / kappa``.
    as_of : date
        Last date used in the estimate.
    n_obs : int
        Number of trading days used.
    """

    mu: float
    phi: float
    sigma: float
    half_life: float
    as_of: date
    n_obs: int


def fit_ou(spread: pd.Series, as_of: date) -> OUFit:
    """Calibrate an Ornstein-Uhlenbeck process to a spread observed daily up to ``as_of``.

    Parameters
    ----------
    spread : pd.Series
        Spread in dollars with a DatetimeIndex.
    as_of : date
        Last date allowed in the estimate (inclusive).

    Returns
    -------
    OUFit
        Mean, daily persistence, shock size and half-life.

    Raises
    ------
    AssertionError
        If data after ``as_of`` survives the filter or the spread does not pull back.

    Notes
    -----
    Sampled daily, an OU process is exactly an AR(1): ``S_t = c + phi * S_{t-1} + e_t`` with
    ``mu = c / (1 - phi)``. OLS of today's level on yesterday's gives ``phi`` and the shock
    standard deviation. The same regression as :func:`pairs.spread.half_life`, so the
    half-life matches Phase 2 (192 days on KO/PEP).
    """
    sample = spread.loc[spread.index <= pd.Timestamp(as_of)].dropna()
    assert sample.index.max() <= pd.Timestamp(as_of), "data after as_of in the OU fit"
    lagged = sample.shift(1).iloc[1:].to_numpy(dtype=float)
    today = sample.iloc[1:].to_numpy(dtype=float)
    phi, c = np.polyfit(lagged, today, 1)
    assert 0 < phi < 1, "the spread does not pull back: no OU fit"
    sigma = float(np.std(today - (c + phi * lagged), ddof=2))
    fit = OUFit(
        mu=float(c / (1.0 - phi)),
        phi=float(phi),
        sigma=sigma,
        half_life=math.log(2) / -math.log(phi),
        as_of=as_of,
        n_obs=len(sample),
    )
    logger.info("OU fit up to %s: %s", as_of, fit)
    return fit


def simulate_ou(
    fit: OUFit, start_value: float, n_days: int, rng: np.random.Generator
) -> np.ndarray:
    """Simulate one Ornstein-Uhlenbeck spread path with the exact daily transition.

    Parameters
    ----------
    fit : OUFit
        Calibrated parameters.
    start_value : float
        Spread on the first day.
    n_days : int
        Length of the path, first day included.
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    np.ndarray
        Simulated spread, ``n_days`` values.

    Notes
    -----
    ``S_t = mu + phi * (S_{t-1} - mu) + sigma * eps_t`` with Gaussian ``eps``: the AR(1) the
    parameters were estimated from, so no discretisation error. Constant volatility and
    normal shocks are simplifications the bootstrap does not make.
    """
    shocks = rng.normal(0.0, fit.sigma, n_days - 1)
    path = np.empty(n_days)
    path[0] = start_value
    for t in range(1, n_days):
        path[t] = fit.mu + fit.phi * (path[t - 1] - fit.mu) + shocks[t - 1]
    return path


def spread_pnl_sharpe(spread: pd.Series, entry: float, window: int, start: int) -> float:
    """Annualised Sharpe of the daily dollar P&L of trading one unit of the spread.

    Parameters
    ----------
    spread : pd.Series
        Spread in dollars (real or simulated).
    entry : float
        Entry threshold on |z|.
    window : int
        z-score window in trading days.
    start : int
        Position of the first evaluated day.

    Returns
    -------
    float
        ``mean / std * sqrt(252)`` of the daily P&L (0 if the P&L never moves).

    Notes
    -----
    An OU path gives the spread only, not the two prices, so returns on capital cannot be
    computed. Holding one unit of the spread is P&L at constant size; the same measure on
    the real spread makes the comparison like for like. Position at the close of t earns
    the move from t to t+1.
    """
    z = rolling_zscore(spread, window)
    position = positions_from_zscore(z.where(z.index >= spread.index[start]), entry)
    pnl = (position.shift(1) * spread.diff()).iloc[start:]
    std = float(pnl.std())
    return float(pnl.mean()) / std * math.sqrt(TRADING_DAYS) if std > 0 else 0.0

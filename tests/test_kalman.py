from __future__ import annotations

import numpy as np
import pandas as pd

from pairs.kalman import kalman_hedge, run_kalman_strategy, trade_hedge


def _prices(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2001-01-01", periods=n)
    b = 50 + np.cumsum(rng.normal(0, 0.5, n))
    beta = np.linspace(0.4, 0.9, n)  # the true hedge ratio drifts
    a = 5 + beta * b + rng.normal(0, 0.3, n)
    return pd.DataFrame({"A": a, "B": b}, index=days)


def test_filter_uses_only_the_past() -> None:
    prices = _prices()
    full = kalman_hedge(prices, "A", "B", 1e-4, 0.09, 5.0, 0.4)
    short = kalman_hedge(prices.iloc[:300], "A", "B", 1e-4, 0.09, 5.0, 0.4)
    pd.testing.assert_frame_equal(full.iloc[:300], short)
    # the estimates of day t do not move if the price of day t changes
    values = prices.to_numpy().copy()
    values[299, 0] += 100.0
    bumped = pd.DataFrame(values, index=prices.index, columns=prices.columns)
    moved = kalman_hedge(bumped, "A", "B", 1e-4, 0.09, 5.0, 0.4)
    assert moved["beta"].iloc[299] == full["beta"].iloc[299]
    assert moved["beta"].iloc[300] != full["beta"].iloc[300]


def test_filter_tracks_a_drifting_hedge_ratio() -> None:
    filt = kalman_hedge(_prices(), "A", "B", 1e-4, 0.09, 5.0, 0.4)
    assert abs(filt["beta"].iloc[-1] - 0.9) < 0.1


def test_tiny_delta_keeps_the_starting_fit() -> None:
    prices = _prices()
    filt = kalman_hedge(prices, "A", "B", 1e-16, 0.09, 5.0, 0.4)
    assert np.allclose(filt["beta"], 0.4, atol=1e-4) and np.allclose(filt["alpha"], 5.0, atol=1e-4)
    assert np.allclose(filt["spread"], prices["A"] - 5.0 - 0.4 * prices["B"], atol=1e-2)


def test_hedge_is_fixed_during_a_trade() -> None:
    days = pd.bdate_range("2001-01-01", periods=7)
    position = pd.Series([0, 1, 1, 0, -1, 1, 1], index=days, dtype=float)
    beta = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], index=days)
    assert trade_hedge(position, beta).tolist() == [0.0, 0.2, 0.2, 0.0, 0.5, 0.6, 0.6]


def test_strategy_starts_flat_and_only_trades_from_start() -> None:
    prices = _prices()
    filt = kalman_hedge(prices, "A", "B", 1e-4, 0.09, 5.0, 0.4)
    start = prices.index[100]
    run = run_kalman_strategy(prices, filt, "A", "B", 1.5, 60, start)
    assert (run.loc[: prices.index[99], "position"] == 0).all()
    assert run["position"].ne(0).any()
    held = run["hedge"].where(run["position"].ne(0))
    assert held.dropna().isin(filt["beta"].to_numpy()).all()

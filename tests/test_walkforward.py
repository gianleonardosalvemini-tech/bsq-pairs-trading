from __future__ import annotations

import numpy as np
import pandas as pd

from pairs.backtest import run_zscore_strategy
from pairs.config import SignalGrid
from pairs.walkforward import cost_returns, run_with_costs, walk_forward


def _prices(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2000-01-03", "2007-12-31")
    n = len(days)
    b = 50 + np.cumsum(rng.normal(0, 0.4, n))
    noise = np.zeros(n)
    for t in range(1, n):  # mean-reverting spread, half-life about 20 days
        noise[t] = 0.966 * noise[t - 1] + rng.normal(0, 0.3)
    return pd.DataFrame({"A": 10 + 0.5 * b + noise, "B": b}, index=days)


def test_cost_of_opening_closing_and_flipping() -> None:
    days = pd.bdate_range("2001-01-01", periods=5)
    prices = pd.DataFrame({"A": 100.0, "B": 50.0}, index=days)
    shares = pd.DataFrame({"A": [0, 1, 1, -1, 0], "B": [0, -2, -2, 2, 0]}, index=days, dtype=float)
    cost = cost_returns(prices, shares, 10.0)
    assert np.allclose(cost.to_numpy(), [0.0, 0.001, 0.0, 0.002, 0.001])


def test_zero_cost_gives_phase2_exactly() -> None:
    prices = _prices()
    start = prices.index[249]
    plain = run_zscore_strategy(prices, "A", "B", 0.5, 1.5, 60, start)
    costed = run_with_costs(prices, "A", "B", 0.5, 1.5, 60, start, 0.0)
    pd.testing.assert_series_equal(plain["ret"], costed["ret"])
    assert (costed["cost"] == 0).all()
    assert (
        run_with_costs(prices, "A", "B", 0.5, 1.5, 60, start, 5.0)["ret"].sum() < plain["ret"].sum()
    )


def test_walk_forward_uses_only_the_past_and_ends_each_year_flat() -> None:
    prices = _prices()
    days, chosen = walk_forward(prices, "A", "B", SignalGrid(), 2004, 2007, 3, 3, 5.0)
    assert [c.year for c in chosen] == [2004, 2005, 2006, 2007]
    assert days.index[0].year == 2004 and days.index.is_unique
    for year in (2004, 2005, 2006):
        assert days["position"].loc[str(year)].iloc[-1] == 0
    # changing the prices of 2007 cannot change what happened before 2007
    in_2007 = np.where(pd.DatetimeIndex(prices.index).year == 2007, 5.0, 0.0)
    bumped = prices.assign(A=prices["A"] + in_2007)
    days2, chosen2 = walk_forward(bumped, "A", "B", SignalGrid(), 2004, 2007, 3, 3, 5.0)
    assert chosen2[:3] == chosen[:3]
    pd.testing.assert_frame_equal(days2.loc[:"2006"], days.loc[:"2006"])

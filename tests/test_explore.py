from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pairs.explore import daily_returns, select_until, yearly_correlation


def make_prices() -> pd.DataFrame:
    days = pd.bdate_range("2017-01-02", "2019-12-31")
    rng = np.random.default_rng(0)
    a = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(days)))
    return pd.DataFrame({"KO": a, "PEP": 2 * a}, index=days)


def test_select_until_drops_every_later_date() -> None:
    kept = select_until(make_prices(), date(2018, 12, 31))
    assert kept.index.max() <= pd.Timestamp("2018-12-31")
    assert kept.index.min() == pd.Timestamp("2017-01-02")


def test_daily_returns_known_values() -> None:
    days = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"X": [100.0, 110.0, 99.0]}, index=days)
    returns = daily_returns(prices)
    assert len(returns) == 2
    assert returns["X"].tolist() == pytest.approx([0.10, -0.10])


def test_yearly_correlation_is_one_for_proportional_prices() -> None:
    corr = yearly_correlation(daily_returns(make_prices()), "KO", "PEP")
    assert corr.index.tolist() == [2017, 2018, 2019]
    assert corr.to_numpy() == pytest.approx(1.0)

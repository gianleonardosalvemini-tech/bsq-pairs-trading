from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pairs.spread import build_spread, fit_hedge_ratio, rolling_zscore


def make_prices() -> pd.DataFrame:
    days = pd.bdate_range("2017-01-02", "2019-12-31")
    rng = np.random.default_rng(1)
    b = 50 * np.cumprod(1 + rng.normal(0, 0.01, len(days)))
    a = 2 + 3 * b + rng.normal(0, 0.5, len(days))
    return pd.DataFrame({"KO": a, "PEP": b}, index=days)


def test_fit_recovers_known_line() -> None:
    days = pd.bdate_range("2020-01-01", periods=50)
    b = np.linspace(10.0, 60.0, 50)
    prices = pd.DataFrame({"KO": 2 + 3 * b, "PEP": b}, index=days)
    fit = fit_hedge_ratio(prices, "KO", "PEP", date(2020, 12, 31))
    assert fit.beta == pytest.approx(3.0)
    assert fit.alpha == pytest.approx(2.0)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.n_obs == 50


def test_fit_ignores_data_after_as_of() -> None:
    prices = make_prices()
    as_of = date(2018, 12, 31)
    before = fit_hedge_ratio(prices, "KO", "PEP", as_of)
    changed = prices.copy()
    changed.loc[changed.index > pd.Timestamp(as_of), "KO"] *= 10
    after = fit_hedge_ratio(changed, "KO", "PEP", as_of)
    assert after == before


def test_build_spread_known_values() -> None:
    days = pd.bdate_range("2020-01-01", periods=2)
    prices = pd.DataFrame({"KO": [10.0, 12.0], "PEP": [4.0, 5.0]}, index=days)
    spread = build_spread(prices, "KO", "PEP", beta=2.0)
    assert spread.tolist() == pytest.approx([2.0, 2.0])
    assert spread.name == "spread"


def test_zscore_ignores_the_future() -> None:
    spread = make_prices()["KO"].rename("spread")
    z = rolling_zscore(spread, 60)
    changed = spread.copy()
    changed.iloc[500:] += 100.0
    z_changed = rolling_zscore(changed, 60)
    pd.testing.assert_series_equal(z.iloc[:500], z_changed.iloc[:500])
    assert z.iloc[:59].isna().all()
    assert z.iloc[59:].notna().all()


def test_zscore_known_value() -> None:
    spread = pd.Series([1.0, 2.0, 3.0, 6.0])
    z = rolling_zscore(spread, 3)
    assert z.iloc[3] == pytest.approx((6.0 - 11 / 3) / np.std([2.0, 3.0, 6.0], ddof=1))

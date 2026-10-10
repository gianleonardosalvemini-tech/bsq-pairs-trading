from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pairs.spread import fit_hedge_ratio
from pairs.stability import (
    compare_groups,
    pvalue_cutoff,
    rolling_adf_pvalue,
    rolling_hedge_ratio,
    trade_returns,
    weak_labels,
)


def make_prices() -> pd.DataFrame:
    days = pd.bdate_range("2017-01-02", "2018-12-31")
    rng = np.random.default_rng(3)
    b = 50 * np.cumprod(1 + rng.normal(0, 0.01, len(days)))
    a = 2 + 3 * b + rng.normal(0, 0.5, len(days))
    return pd.DataFrame({"KO": a, "PEP": b}, index=days)


def test_rolling_beta_matches_ols_on_the_window() -> None:
    prices = make_prices()
    beta = rolling_hedge_ratio(prices, "KO", "PEP", 100)
    day = prices.index[300]
    window = prices.loc[:day].iloc[-100:]
    fit = fit_hedge_ratio(window, "KO", "PEP", day.date())
    assert beta.loc[day] == pytest.approx(fit.beta)
    assert beta.iloc[:99].isna().all() and beta.iloc[99:].notna().all()


def test_adf_separates_mean_reversion_from_a_random_walk() -> None:
    days = pd.bdate_range("2015-01-01", periods=600)
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 1, len(days))
    reverting = pd.Series(noise, index=days)  # white noise: as mean-reverting as it gets
    walk = pd.Series(np.cumsum(noise), index=days)
    assert rolling_adf_pvalue(reverting, 250).dropna().max() < 0.01
    assert rolling_adf_pvalue(walk, 250).dropna().median() > 0.10


def test_adf_pvalue_uses_only_the_past() -> None:
    days = pd.bdate_range("2015-01-01", periods=400)
    spread = pd.Series(np.random.default_rng(5).normal(0, 1, len(days)).cumsum(), index=days)
    before = rolling_adf_pvalue(spread, 100, step=5)
    changed = spread.copy()
    changed.iloc[300:] += 50.0
    after = rolling_adf_pvalue(changed, 100, step=5)
    pd.testing.assert_series_equal(before.iloc[:300], after.iloc[:300])


def test_labels_are_known_the_day_before() -> None:
    days = pd.bdate_range("2020-01-01", periods=5)
    pvalues = pd.Series([np.nan, 0.5, 0.01, 0.5, 0.01], index=days, name="adf_pvalue")
    weak = weak_labels(pvalues, cutoff=0.10)
    assert weak.iloc[:2].isna().all()
    assert weak.iloc[2:].tolist() == [1.0, 0.0, 1.0]


def test_cutoff_ignores_data_after_as_of() -> None:
    days = pd.bdate_range("2018-12-24", periods=10)
    pvalues = pd.Series(np.linspace(0.0, 0.9, 10), index=days, name="adf_pvalue")
    cutoff = pvalue_cutoff(pvalues, date(2018, 12, 31), share=0.5)
    in_sample = pvalues.loc[: pd.Timestamp("2018-12-31")]
    assert cutoff == pytest.approx(in_sample.quantile(0.5))
    changed = pvalues.copy()
    changed.iloc[-3:] = 1.0
    assert pvalue_cutoff(changed, date(2018, 12, 31), share=0.5) == pytest.approx(cutoff)


def test_trade_returns_compound_to_the_total() -> None:
    days = pd.bdate_range("2020-01-01", periods=8)
    position = pd.Series([0, 1, 1, 0, -1, 1, 1, 0], index=days, dtype=float)
    ret = pd.Series([0, 0, 0.01, 0.02, 0, -0.01, 0.03, 0.01], index=days)
    trades = trade_returns(pd.DataFrame({"position": position, "ret": ret}))
    assert trades["entry"].tolist() == [days[1], days[4], days[5]]
    assert trades["direction"].tolist() == [1, -1, 1]
    assert trades["days"].tolist() == [2, 1, 2]
    assert trades["ret"].iloc[0] == pytest.approx(1.01 * 1.02 - 1)
    assert trades["ret"].iloc[2] == pytest.approx(1.03 * 1.01 - 1)


def test_compare_groups() -> None:
    days = pd.bdate_range("2020-01-01", periods=6)
    values = pd.Series([1.0, 3.0, -1.0, -3.0, 5.0, 0.0], index=days)
    weak = pd.Series([0, 0, 1, 1, np.nan, 1], index=days, dtype=float)
    out = compare_groups(values, weak)
    assert out["n_weak"] == 3 and out["n_normal"] == 2
    assert out["mean_normal"] == pytest.approx(2.0)
    assert out["mean_weak"] == pytest.approx(-4.0 / 3)
    assert out["welch_t"] < 0
    assert math.isnan(compare_groups(values, weak.where(weak.eq(1)))["welch_t"])

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pairs.simulate import (
    OUFit,
    block_bootstrap_indices,
    fit_ou,
    simulate_joint_prices,
    simulate_ou,
    spread_pnl_sharpe,
)


def make_prices() -> pd.DataFrame:
    days = pd.bdate_range("2017-01-02", periods=300)
    step = np.arange(1, len(days)) * 1e-5  # every historical day has its own return
    ko = 40 * np.concatenate([[1.0], np.cumprod(1 + step)])
    pep = 90 * np.concatenate([[1.0], np.cumprod(1 - step)])
    return pd.DataFrame({"KO": ko, "PEP": pep}, index=days)


def test_blocks_are_consecutive_and_cut_to_length() -> None:
    rows = block_bootstrap_indices(100, 10, 35, np.random.default_rng(0))
    assert len(rows) == 35 and rows.min() >= 0 and rows.max() < 100
    for k in range(3):  # three full blocks, then a cut one
        block = rows[10 * k : 10 * k + 10]
        np.testing.assert_array_equal(np.diff(block), np.ones(9))


def test_both_legs_get_the_same_days() -> None:
    path = simulate_joint_prices(make_prices(), 10, np.random.default_rng(1), date(2018, 12, 31))
    ret = path.pct_change().iloc[1:]
    # KO's return on a historical day is minus PEP's: kept only if both legs draw the same day
    np.testing.assert_allclose(ret["KO"], -ret["PEP"], atol=1e-12)


def test_path_starts_at_real_prices_and_is_reproducible() -> None:
    prices = make_prices()
    one = simulate_joint_prices(prices, 5, np.random.default_rng(2), date(2018, 12, 31))
    two = simulate_joint_prices(prices, 5, np.random.default_rng(2), date(2018, 12, 31))
    pd.testing.assert_frame_equal(one, two)
    pd.testing.assert_series_equal(one.iloc[0], prices.iloc[0])


def test_data_after_as_of_is_never_resampled() -> None:
    prices = make_prices()
    as_of = prices.index[200].date()
    before = simulate_joint_prices(prices, 10, np.random.default_rng(3), as_of)
    changed = prices.copy()
    changed.iloc[201:] *= 3.0
    after = simulate_joint_prices(changed, 10, np.random.default_rng(3), as_of)
    pd.testing.assert_frame_equal(before, after)
    assert after.index.max() == pd.Timestamp(as_of)


def test_ou_fit_recovers_the_simulated_parameters() -> None:
    days = pd.bdate_range("1990-01-01", periods=20_000)
    truth = OUFit(mu=5.0, phi=0.99, sigma=0.4, half_life=68.97, as_of=date(1990, 1, 1), n_obs=0)
    rng = np.random.default_rng(4)
    path = pd.Series(simulate_ou(truth, truth.mu, len(days), rng), index=days)
    fit = fit_ou(path, days[-1].date())
    assert fit.phi == pytest.approx(truth.phi, abs=2e-3)
    assert fit.sigma == pytest.approx(truth.sigma, rel=0.05)


def test_spread_pnl_earns_from_the_next_day() -> None:
    days = pd.bdate_range("2015-01-01", periods=200)
    values = np.zeros(len(days))
    values[150] = -10.0  # one deep dip: long at its close, earns only from day 151
    values[151:] = 0.0
    spread = pd.Series(values, index=days) + np.linspace(0, 1e-6, len(days))
    sharpe = spread_pnl_sharpe(spread, 1.5, 60, 100)
    assert sharpe > 0  # the +10 rebound on day 151 is earned, the -10 drop on day 150 is not

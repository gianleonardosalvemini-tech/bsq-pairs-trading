from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pairs.explore import (
    adjustment_from_dividends,
    daily_returns,
    largest_moves,
    longest_gaps,
    select_until,
    yearly_correlation,
)


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


def test_longest_gaps_finds_the_missing_week() -> None:
    days = pd.bdate_range("2020-01-01", "2020-03-31").delete(list(range(20, 25)))
    gaps = longest_gaps(pd.DatetimeIndex(days), 1)
    assert gaps["calendar_days"].iloc[0] == 8


def test_largest_moves_flags_a_reversed_spike() -> None:
    days = pd.bdate_range("2020-01-01", periods=6)
    close = pd.DataFrame({"KO": [10.0, 10.0, 15.0, 10.0, 10.0, 10.0]}, index=days)
    volume = pd.DataFrame({"KO": [100.0] * 6}, index=days)
    moves = largest_moves(close, volume, 1)
    assert moves["date"].iloc[0] == days[2].date()
    assert moves["return"].iloc[0] == pytest.approx(0.5)
    assert moves["next_day_return"].iloc[0] == pytest.approx(-1 / 3)
    assert moves["volume_vs_median"].iloc[0] == pytest.approx(1.0)


def test_adjustment_from_dividends_known_value() -> None:
    days = pd.bdate_range("2020-01-01", periods=4)
    close = pd.Series([100.0, 100.0, 99.0, 99.0], index=days)
    dividends = pd.Series([1.0], index=[days[2]])  # $1 paid, price was 100 the day before
    factor = adjustment_from_dividends(close, dividends)
    assert factor.tolist() == pytest.approx([0.99, 0.99, 1.0, 1.0])

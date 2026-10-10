from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pairs.config import PairConfig
from pairs.data import (
    PARSED_FILE,
    VALIDATED_FILE,
    DividendError,
    correct_dividend_errors,
    parse_prices,
    validate_prices,
)
from pairs.errors import StaleDataError

PAIR = PairConfig("KO", "PEP", date(2020, 1, 1), date(2020, 1, 10), date(2020, 1, 31))


def make_raw(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Yahoo-shaped frame: (field, ticker) columns, PEP before KO to test reordering."""
    columns = pd.MultiIndex.from_product([["Adj Close", "Close"], ["PEP", "KO"]])
    values = np.tile([100.0, 50.0, 110.0, 55.0], (len(days), 1))
    return pd.DataFrame(values, index=days, columns=columns)


def test_parse_keeps_adjusted_close_in_pair_order(tmp_path: Path) -> None:
    raw = make_raw(pd.bdate_range("2020-01-01", "2020-01-31"))
    prices = parse_prices(raw, PAIR, tmp_path)
    assert list(prices.columns) == ["KO", "PEP"]
    assert prices["KO"].iloc[0] == 50.0  # Adj Close, not Close (55)
    assert prices.index.name == "date"
    assert (tmp_path / PARSED_FILE).exists()


def test_validate_drops_days_with_a_missing_leg(tmp_path: Path) -> None:
    prices = parse_prices(make_raw(pd.bdate_range("2020-01-01", "2020-01-31")), PAIR, tmp_path)
    prices.loc[prices.index[3], "PEP"] = np.nan
    clean = validate_prices(prices, PAIR, tmp_path)
    assert len(clean) == len(prices) - 1
    assert not clean.isna().any().any()
    assert (tmp_path / VALIDATED_FILE).exists()


def test_validate_rejects_data_that_ends_too_early(tmp_path: Path) -> None:
    prices = parse_prices(make_raw(pd.bdate_range("2020-01-01", "2020-01-15")), PAIR, tmp_path)
    with pytest.raises(StaleDataError):
        validate_prices(prices, PAIR, tmp_path)


def test_validate_rejects_non_positive_price(tmp_path: Path) -> None:
    prices = parse_prices(make_raw(pd.bdate_range("2020-01-01", "2020-01-31")), PAIR, tmp_path)
    prices.loc[prices.index[5], "KO"] = 0.0
    with pytest.raises(AssertionError):
        validate_prices(prices, PAIR, tmp_path)


def test_correct_dividend_errors_undoes_a_fake_dividend(tmp_path: Path) -> None:
    days = pd.bdate_range("2020-01-01", periods=4)
    columns = pd.MultiIndex.from_product([["Adj Close", "Close"], ["KO", "PEP"]])
    # Close is flat at 100; Yahoo applied a fake $1 dividend on day 2 to Adj Close
    values = [[99.0, 50.0, 100.0, 50.0], [99.0, 50.0, 100.0, 50.0]]
    values += [[100.0, 50.0, 100.0, 50.0], [100.0, 50.0, 100.0, 50.0]]
    raw = pd.DataFrame(values, index=days, columns=columns)
    prices = parse_prices(raw, PAIR, tmp_path)
    error = DividendError("KO", days[2].date(), 1.0, "test")
    fixed = correct_dividend_errors(prices, raw, (error,))
    assert fixed["KO"].tolist() == pytest.approx([100.0] * 4)
    assert fixed["PEP"].equals(prices["PEP"])
    assert prices["KO"].iloc[0] == 99.0  # input untouched


def test_load_pair_prices_cuts_dates_and_fixes_the_known_dividend() -> None:
    from pairs.config import PAIR as REAL_PAIR
    from pairs.config import Settings
    from pairs.data import load_pair_prices

    prices = load_pair_prices(REAL_PAIR, Settings.from_env().data_dir, REAL_PAIR.in_sample_end)
    assert list(prices.columns) == [REAL_PAIR.ticker_a, REAL_PAIR.ticker_b]
    assert prices.index.min() >= pd.Timestamp(REAL_PAIR.start)
    assert prices.index.max() == pd.Timestamp(REAL_PAIR.in_sample_end)
    assert not prices.isna().any().any()
    ex_day = pd.Timestamp("2001-09-10")  # the fake KO dividend must no longer show as a +0.73% jump
    assert abs(prices[REAL_PAIR.ticker_a].pct_change().loc[ex_day]) < 0.005

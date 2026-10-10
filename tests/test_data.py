from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pairs.config import PairConfig
from pairs.data import PARSED_FILE, VALIDATED_FILE, parse_prices, validate_prices
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

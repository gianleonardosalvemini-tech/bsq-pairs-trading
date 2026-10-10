from __future__ import annotations

from datetime import date

import pytest

from pairs.config import HISTORY_START, PAIR, PairConfig
from pairs.errors import ConfigError


def test_valid_config() -> None:
    cfg = PairConfig("KO", "PEP", date(2005, 1, 1), date(2018, 12, 31), date(2026, 9, 30))
    assert cfg.in_sample_end < cfg.end


def test_same_ticker_fails_fast() -> None:
    with pytest.raises(ConfigError):
        PairConfig("KO", "KO", date(2005, 1, 1), date(2018, 12, 31), date(2026, 9, 30))


def test_split_after_end_fails_fast() -> None:
    with pytest.raises(ConfigError):
        PairConfig("KO", "PEP", date(2005, 1, 1), date(2027, 1, 1), date(2026, 9, 30))


def test_project_pair_split_is_fixed() -> None:
    assert (PAIR.ticker_a, PAIR.ticker_b) == ("KO", "PEP")
    assert PAIR.in_sample_end == date(2018, 12, 31)


def test_history_start_is_only_descriptive() -> None:
    # Phase 1 history starts on PepsiCo's first Yahoo day, before the strategy's in-sample
    assert date(1972, 6, 1) == HISTORY_START < PAIR.start

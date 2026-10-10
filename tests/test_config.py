from __future__ import annotations

from datetime import date

import pytest

from pairs.config import PairConfig
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

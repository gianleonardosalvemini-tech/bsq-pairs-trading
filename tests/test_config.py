from __future__ import annotations

from datetime import date

import pytest

from pairs.config import (
    HISTORY_START,
    PAIR,
    PHASE2_STRATEGY,
    SIGNAL_GRID,
    FrozenStrategy,
    PairConfig,
    SignalGrid,
)
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


def test_signal_grid_is_preregistered() -> None:
    assert SIGNAL_GRID.entry_thresholds == (1.0, 1.5, 2.0, 2.5, 3.0)
    assert SIGNAL_GRID.windows == (20, 60, 120, 190, 250)
    assert SIGNAL_GRID.exit_threshold == 0.0


def test_signal_grid_without_interior_fails_fast() -> None:
    with pytest.raises(ConfigError):
        SignalGrid(entry_thresholds=(1.0, 2.0))


def test_signal_grid_exit_above_entry_fails_fast() -> None:
    with pytest.raises(ConfigError):
        SignalGrid(exit_threshold=1.5)


def test_phase2_strategy_is_frozen() -> None:
    assert (PHASE2_STRATEGY.entry, PHASE2_STRATEGY.window) == (1.5, 60)
    assert PHASE2_STRATEGY.entry in SIGNAL_GRID.entry_thresholds[1:-1]
    assert PHASE2_STRATEGY.window in SIGNAL_GRID.windows[1:-1]
    with pytest.raises(ConfigError):
        FrozenStrategy(entry=1.5, window=60, exit=2.0, beta=0.4)

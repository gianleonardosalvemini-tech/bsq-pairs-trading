from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pairs.backtest import pair_shares, positions_from_zscore, run_backtest, summarize
from pairs.trials import count_trials, log_trial


def test_state_machine_by_hand() -> None:
    z = pd.Series([np.nan, -1.0, -2.5, -1.0, 0.1, 2.6, 0.5, -0.2, 3.0, -2.1])
    position = positions_from_zscore(z, entry=2.0)
    assert position.tolist() == [0, 0, 1, 1, 0, -1, -1, 0, -1, 1]


def test_position_earns_only_from_next_day() -> None:
    days = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"A": [10.0, 20.0, 40.0, 40.0]}, index=days)
    shares = pd.DataFrame({"A": [0.0, 1.0, 0.0, 0.0]}, index=days)
    result = run_backtest(prices, shares)
    # bought at the close of day 1 (20): day 1 itself earns nothing, day 2 earns 20.
    assert result["pnl"].tolist() == pytest.approx([0.0, 0.0, 20.0, 0.0])
    assert result["ret"].tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0])


def test_two_leg_pnl_by_hand() -> None:
    days = pd.bdate_range("2020-01-01", periods=2)
    prices = pd.DataFrame({"KO": [50.0, 51.0], "PEP": [100.0, 98.0]}, index=days)
    position = pd.Series([1.0, 1.0], index=days)
    shares = pair_shares(position, "KO", "PEP", beta=0.5)
    result = run_backtest(prices, shares)
    # long 1 KO (+1 $) and short 0.5 PEP (+1 $) on 50 + 50 = 100 $ of gross exposure.
    assert result["pnl"].iloc[1] == pytest.approx(2.0)
    assert result["gross"].iloc[1] == pytest.approx(100.0)
    assert result["ret"].iloc[1] == pytest.approx(0.02)


def test_summarize_known_numbers() -> None:
    ret = pd.Series([0.1, -0.5, 0.2, 0.0])
    position = pd.Series([1.0, 1.0, 0.0, -1.0])
    stats = summarize(ret, position)
    assert stats["max_drawdown"] == pytest.approx(-0.5)
    assert stats["n_trades"] == 2
    assert stats["pct_in_market"] == pytest.approx(0.75)
    assert stats["sharpe"] == pytest.approx(ret.mean() / ret.std() * np.sqrt(252))


def test_trial_log_counts_rows(tmp_path: Path) -> None:
    db = tmp_path / "trials.sqlite"
    assert count_trials(db) == 0
    period = (date(2001, 1, 2), date(2018, 12, 31))
    for window in (20, 60):
        params = {"window": window}
        log_trial(db, "phase2-grid", "KO/PEP", "in-sample", period, params, {"sharpe": 0.1})
    log_trial(db, "phase2-zoom", "KO/PEP", "in-sample", period, {"window": 66}, {"sharpe": 0.2})
    assert count_trials(db) == 3
    assert count_trials(db, "phase2-grid") == 2

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs.errors import DataFetchError, ShortInterestUnavailableError
from pairs.short_interest import interval_table, parse_finra, parse_nasdaq, spearman

NASDAQ_PEP: dict[str, object] = {
    "data": {
        "shortInterestTable": {
            "rows": [
                {
                    "settlementDate": "09/30/2026",
                    "interest": "23,293,009",
                    "avgDailyShareVolume": "9,951,472",
                    "daysToCover": 2.34066,
                },
                {
                    "settlementDate": "09/15/2026",
                    "interest": "25,853,939",
                    "avgDailyShareVolume": "8,064,282",
                    "daysToCover": 3.205982,
                },
            ]
        }
    }
}


def test_parse_nasdaq_reads_numbers_with_separators_and_sorts() -> None:
    table = parse_nasdaq(NASDAQ_PEP, "PEP")
    assert list(table.index) == [pd.Timestamp("2026-09-15"), pd.Timestamp("2026-09-30")]
    assert table.loc["2026-09-30", "short_interest"] == 23_293_009


def test_parse_nasdaq_reports_nyse_tickers_as_unavailable() -> None:
    refusal: dict[str, object] = {
        "data": None,
        "message": "Short interest is only supported for Nasdaq Listed stocks",
    }
    with pytest.raises(ShortInterestUnavailableError, match="Nasdaq Listed"):
        parse_nasdaq(refusal, "KO")


def test_parse_finra_rejects_another_ticker() -> None:
    record: dict[str, object] = {
        "symbolCode": "KO",
        "settlementDate": "2026-09-15",
        "currentShortPositionQuantity": 40_584_069,
        "averageDailyVolumeQuantity": 14_540_305,
        "daysToCoverQuantity": 2.79,
    }
    assert parse_finra([record], "KO").iloc[0]["short_interest"] == 40_584_069
    with pytest.raises(DataFetchError):
        parse_finra([record], "PEP")


def test_interval_table_by_hand() -> None:
    days = pd.bdate_range("2026-01-01", periods=10)
    spread = pd.Series(np.arange(10.0), index=days)  # +1 dollar per day
    dates = pd.DatetimeIndex([days[0], days[3]])
    si_a = pd.Series([100.0, 200.0], index=dates)
    si_b = pd.Series([100.0, 100.0], index=dates)
    row = interval_table(si_a, si_b, spread, publication_lag=2, horizon=3).iloc[0]
    assert row["rel_shorting"] == pytest.approx(np.log(2))
    assert row["spread_change"] == 3.0  # day 0 -> day 3
    assert row["next_spread_change"] == 3.0  # day 5 -> day 8, from publication


def test_interval_table_next_change_is_missing_when_prices_end() -> None:
    days = pd.bdate_range("2026-01-01", periods=5)
    spread = pd.Series(np.arange(5.0), index=days)
    dates = pd.DatetimeIndex([days[0], days[3]])
    si = pd.Series([1.0, 2.0], index=dates)
    assert np.isnan(interval_table(si, si, spread, 2, 3).iloc[0]["next_spread_change"])


def test_spearman_is_one_for_monotone_data() -> None:
    x = pd.Series(np.arange(20.0))
    assert spearman(x, x**3)["r"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        spearman(x.iloc[:5], x.iloc[:5])

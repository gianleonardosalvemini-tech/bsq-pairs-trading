"""Phase 4 data: short interest from Nasdaq (the brief's source) and FINRA (both legs)."""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pairs.errors import DataFetchError, ShortInterestUnavailableError

logger = logging.getLogger(__name__)

NASDAQ_URL = "https://api.nasdaq.com/api/quote/{ticker}/short-interest?assetClass=stocks"
FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
NASDAQ_RAW_FILE = "short_interest_nasdaq_raw.pkl"
FINRA_RAW_FILE = "short_interest_finra_raw.pkl"
COLUMNS = ["short_interest", "avg_daily_volume", "days_to_cover"]
TIMEOUT_S = 60


def fetch_nasdaq_raw(
    tickers: list[str], data_dir: Path, refresh: bool = False
) -> dict[str, dict[str, object]]:
    """Download Nasdaq's short-interest JSON for each ticker once, and checkpoint it as received.

    Parameters
    ----------
    tickers : list of str
        Tickers to query (NYSE tickers are queried too, to record Nasdaq's refusal).
    data_dir : Path
        Folder holding the pickled checkpoints.
    refresh : bool, default False
        Re-download even if the checkpoint exists.

    Returns
    -------
    dict of str to dict
        Ticker -> JSON payload, untouched (``data`` is null for non-Nasdaq tickers).

    Raises
    ------
    DataFetchError
        If the HTTP request fails or the answer is not JSON.

    Notes
    -----
    Nasdaq serves a rolling 12 months only, so tomorrow's answer differs from today's:
    the checkpoint, not the endpoint, is what makes the notebook reproducible. Nasdaq
    rejects requests without a browser-like User-Agent.
    """
    path = data_dir / NASDAQ_RAW_FILE
    if path.exists() and not refresh:
        cached: dict[str, dict[str, object]] = pd.read_pickle(path)
        return cached
    raw: dict[str, dict[str, object]] = {}
    for ticker in tickers:
        try:
            answer = requests.get(
                NASDAQ_URL.format(ticker=ticker),
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                timeout=TIMEOUT_S,
            )
            answer.raise_for_status()
            raw[ticker] = answer.json()
        except (requests.RequestException, ValueError) as err:
            raise DataFetchError(f"Nasdaq short interest for {ticker}: {err}") from err
    with path.open("wb") as handle:
        pickle.dump(raw, handle)
    logger.info("Saved Nasdaq short-interest checkpoint %s", path)
    return raw


def fetch_finra_raw(
    tickers: list[str], data_dir: Path, refresh: bool = False
) -> dict[str, list[dict[str, object]]]:
    """Download FINRA's consolidated short interest for each ticker, and checkpoint it as received.

    Parameters
    ----------
    tickers : list of str
        Tickers to query (NYSE and Nasdaq alike).
    data_dir : Path
        Folder holding the pickled checkpoints.
    refresh : bool, default False
        Re-download even if the checkpoint exists.

    Returns
    -------
    dict of str to list of dict
        Ticker -> list of FINRA records (one per settlement date), untouched.

    Raises
    ------
    DataFetchError
        If the HTTP request fails, the answer is not JSON, or a ticker has no record.

    Notes
    -----
    FINRA collects short interest from every broker for all exchange-listed stocks and
    publishes it twice a month; Nasdaq's page shows the same numbers for its own
    listings (checked in the Phase 4 notebook). The public API needs no key; one
    request returns the whole history available for a ticker (from December 2017).
    """
    path = data_dir / FINRA_RAW_FILE
    if path.exists() and not refresh:
        cached: dict[str, list[dict[str, object]]] = pd.read_pickle(path)
        return cached
    raw: dict[str, list[dict[str, object]]] = {}
    for ticker in tickers:
        query = {
            "limit": 5000,
            "compareFilters": [
                {"compareType": "equal", "fieldName": "symbolCode", "fieldValue": ticker}
            ],
        }
        try:
            answer = requests.post(
                FINRA_URL, json=query, headers={"Accept": "application/json"}, timeout=TIMEOUT_S
            )
            answer.raise_for_status()
            records = answer.json()
        except (requests.RequestException, ValueError) as err:
            raise DataFetchError(f"FINRA short interest for {ticker}: {err}") from err
        if not isinstance(records, list) or not records:
            raise DataFetchError(f"FINRA returned no short-interest record for {ticker}")
        raw[ticker] = records
    with path.open("wb") as handle:
        pickle.dump(raw, handle)
    logger.info("Saved FINRA short-interest checkpoint %s", path)
    return raw


def _number(value: object) -> float:
    """Read a number that may arrive as int, float or text with thousands separators."""
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value.replace(",", ""))
    raise DataFetchError(f"Not a number in the short-interest data: {value!r}")


def parse_nasdaq(payload: dict[str, object], ticker: str) -> pd.DataFrame:
    """Turn one Nasdaq payload into a table indexed by settlement date.

    Parameters
    ----------
    payload : dict
        One value of :func:`fetch_nasdaq_raw`.
    ticker : str
        Ticker, for the error message.

    Returns
    -------
    pd.DataFrame
        Columns ``short_interest`` (shares), ``avg_daily_volume`` (shares),
        ``days_to_cover``; sorted DatetimeIndex ``settlement_date``.

    Raises
    ------
    ShortInterestUnavailableError
        If Nasdaq has no data for the ticker (it covers Nasdaq-listed stocks only).
    DataFetchError
        If the payload does not have the expected layout.

    Notes
    -----
    Nasdaq's message for an NYSE ticker is kept in the error, so the notebook can show
    the source's own words.
    """
    data = payload.get("data")
    if data is None:
        raise ShortInterestUnavailableError(f"{ticker}: {payload.get('message')}")
    try:
        rows = data["shortInterestTable"]["rows"]  # type: ignore[index]
    except (KeyError, TypeError) as err:
        raise DataFetchError(f"Unexpected Nasdaq layout for {ticker}") from err
    table = pd.DataFrame(
        {
            "short_interest": [_number(r["interest"]) for r in rows],
            "avg_daily_volume": [_number(r["avgDailyShareVolume"]) for r in rows],
            "days_to_cover": [_number(r["daysToCover"]) for r in rows],
        },
        index=pd.DatetimeIndex(
            pd.to_datetime([r["settlementDate"] for r in rows], format="%m/%d/%Y"),
            name="settlement_date",
        ),
    )
    return _check(table.sort_index(), ticker)


def parse_finra(records: list[dict[str, object]], ticker: str) -> pd.DataFrame:
    """Turn FINRA records into the same table as :func:`parse_nasdaq`.

    Parameters
    ----------
    records : list of dict
        One value of :func:`fetch_finra_raw`.
    ticker : str
        Ticker, for checks and messages.

    Returns
    -------
    pd.DataFrame
        Columns ``short_interest``, ``avg_daily_volume``, ``days_to_cover``; sorted
        DatetimeIndex ``settlement_date``.

    Raises
    ------
    DataFetchError
        If a record belongs to another ticker or a field is not a number.

    Notes
    -----
    FINRA's ``currentShortPositionQuantity`` is the short interest; its days to cover
    is short interest divided by average daily volume, rounded to two decimals.
    """
    if any(r.get("symbolCode") != ticker for r in records):
        raise DataFetchError(f"FINRA records for {ticker} contain another ticker")
    table = pd.DataFrame(
        {
            "short_interest": [_number(r["currentShortPositionQuantity"]) for r in records],
            "avg_daily_volume": [_number(r["averageDailyVolumeQuantity"]) for r in records],
            "days_to_cover": [_number(r["daysToCoverQuantity"]) for r in records],
        },
        index=pd.DatetimeIndex(
            pd.to_datetime([str(r["settlementDate"]) for r in records]), name="settlement_date"
        ),
    )
    return _check(table.sort_index(), ticker)


def _check(table: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Assert the invariants of a short-interest table and return it."""
    assert table.index.is_unique, f"{ticker}: duplicate settlement dates"
    assert (table["short_interest"] > 0).all(), f"{ticker}: short interest must be positive"
    assert (table["avg_daily_volume"] > 0).all(), f"{ticker}: volume must be positive"
    return table


def interval_table(
    si_a: pd.Series,
    si_b: pd.Series,
    spread: pd.Series,
    publication_lag: int,
    horizon: int,
) -> pd.DataFrame:
    """One row per interval between two settlement dates: short-interest and spread changes.

    Parameters
    ----------
    si_a, si_b : pd.Series
        Short interest (shares) of leg A and leg B on the same settlement dates.
    spread : pd.Series
        Daily spread ``A - beta * B`` (dollars), indexed by trading day.
    publication_lag : int
        Trading days between a settlement date and the day its number is public.
    horizon : int
        Trading days over which the *next* spread change is measured, from publication.

    Returns
    -------
    pd.DataFrame
        Indexed by the settlement date ending each interval, with columns ``d_log_si_a``,
        ``d_log_si_b``, ``rel_shorting`` (= ``d_log_si_a - d_log_si_b``),
        ``spread_change`` (over the same interval) and ``next_spread_change`` (from
        publication to ``horizon`` days later; NaN where prices end too early).

    Raises
    ------
    AssertionError
        If the two legs do not share the same settlement dates.

    Notes
    -----
    Log changes make the two legs comparable although KO has about twice PEP's short
    interest. ``rel_shorting > 0`` means short sellers added more (or cut less) on A than
    on B. The spread on a settlement date is the last close on or before it. The
    "next" change starts at publication, not at settlement: before publication nobody
    could have traded on the number.
    """
    assert si_a.index.equals(si_b.index), "legs must share settlement dates"
    days = spread.index
    at = days.searchsorted(si_a.index, side="right") - 1  # last trading day <= settlement
    assert (at >= 0).all(), "spread must start before the first settlement date"
    level = spread.to_numpy()[at]
    out = pd.DataFrame(index=si_a.index[1:])
    out["d_log_si_a"] = np.diff(np.log(si_a.to_numpy()))
    out["d_log_si_b"] = np.diff(np.log(si_b.to_numpy()))
    out["rel_shorting"] = out["d_log_si_a"] - out["d_log_si_b"]
    out["spread_change"] = np.diff(level)
    start, end = at[1:] + publication_lag, at[1:] + publication_lag + horizon
    valid = end < len(days)
    nxt = np.full(len(out), np.nan)
    nxt[valid] = spread.to_numpy()[end[valid]] - spread.to_numpy()[start[valid]]
    out["next_spread_change"] = nxt
    return out


def spearman(x: pd.Series, y: pd.Series) -> dict[str, float]:
    """Rank correlation between two series on their common, non-missing dates.

    Parameters
    ----------
    x, y : pd.Series
        Values to compare.

    Returns
    -------
    dict of str to float
        ``r`` (Spearman), ``t_stat`` and ``n``.

    Raises
    ------
    ValueError
        If fewer than 10 common observations remain.

    Notes
    -----
    Ranks instead of values, so that one extreme half-month (short interest can jump by
    40% around an index rebalance) cannot drive the result on its own. ``t_stat`` =
    r * sqrt((n - 2) / (1 - r^2)), the usual test of a zero correlation: as in the other
    phases, |t| below about 2 can be luck.
    """
    both = pd.concat([x, y], axis=1).dropna()
    if len(both) < 10:
        raise ValueError(f"only {len(both)} common observations")
    r = float(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman"))
    n = len(both)
    t_stat = r * math.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else math.copysign(math.inf, r)
    return {"r": r, "t_stat": t_stat, "n": float(n)}

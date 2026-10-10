"""Phase 1 data pipeline: Yahoo download and raw / parsed / validated checkpoints."""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from pairs.config import PairConfig
from pairs.errors import DataFetchError

logger = logging.getLogger(__name__)

RAW_FILE = "prices_raw.pkl"


def fetch_raw_prices(pair: PairConfig, data_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Download daily prices for both legs once and checkpoint them exactly as received.

    Parameters
    ----------
    pair : PairConfig
        Tickers and the full sample window (``start`` to ``end``, both inclusive).
    data_dir : Path
        Folder holding the pickled checkpoints.
    refresh : bool, default False
        Re-download even if the raw checkpoint exists.

    Returns
    -------
    pd.DataFrame
        Yahoo output untouched: columns are a (field, ticker) MultiIndex, index is the date.

    Raises
    ------
    DataFetchError
        If Yahoo returns no rows, or no Adj Close for either ticker (yfinance reports
        most failures by logging and returning an empty frame, not by raising).

    Notes
    -----
    The raw checkpoint is read back instead of re-downloading because Yahoo rewrites
    adjusted prices after every dividend: a new download would not reproduce our numbers.
    ``auto_adjust=False`` keeps both Close and Adj Close, so the choice is made (and
    documented) in the parsing step, not hidden inside the download. Yahoo's ``end`` is
    exclusive, hence the extra day.
    """
    path = data_dir / RAW_FILE
    if path.exists() and not refresh:
        logger.info("Loading raw checkpoint %s", path)
        cached: pd.DataFrame = pd.read_pickle(path)
        return cached
    tickers = [pair.ticker_a, pair.ticker_b]
    logger.info("Downloading %s from Yahoo, %s to %s", tickers, pair.start, pair.end)
    raw: pd.DataFrame | None = yf.download(  # yfinance is untyped: pin the type here
        tickers,
        start=pair.start.isoformat(),
        end=(pair.end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise DataFetchError(f"Yahoo returned no rows for {tickers}")
    for ticker in tickers:
        if raw["Adj Close"][ticker].dropna().empty:
            raise DataFetchError(f"No Adj Close data for {ticker}")
    raw.to_pickle(path)
    logger.info("Saved raw checkpoint %s (%d rows)", path, len(raw))
    return raw

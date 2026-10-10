"""Scratch: the 25 pre-registered in-sample trials (logged in results/trials.sqlite)."""

from __future__ import annotations

import pandas as pd

from pairs.backtest import run_zscore_strategy, select_stable_cell, summarize
from pairs.config import PAIR, SIGNAL_GRID, Settings
from pairs.data import KNOWN_DIVIDEND_ERRORS, correct_dividend_errors, fetch_raw_prices
from pairs.explore import select_until
from pairs.spread import fit_hedge_ratio
from pairs.trials import count_trials, log_trial

settings = Settings.from_env()
db = settings.results_dir / "trials.sqlite"
raw = select_until(fetch_raw_prices(PAIR, settings.data_dir), PAIR.in_sample_end)
raw = raw.loc[raw.index >= pd.Timestamp(PAIR.start)]
prices = correct_dividend_errors(raw["Adj Close"][["KO", "PEP"]], raw, KNOWN_DIVIDEND_ERRORS).dropna()
fit = fit_hedge_ratio(prices, "KO", "PEP", PAIR.in_sample_end)
start = prices.index[max(SIGNAL_GRID.windows) - 1]
halves = {"h1": (start, pd.Timestamp("2009-12-31")), "h2": (pd.Timestamp("2010-01-01"), prices.index[-1])}
print("eval period", start.date(), prices.index[-1].date(), "beta", round(fit.beta, 4))

rows = []
for entry in SIGNAL_GRID.entry_thresholds:
    for window in SIGNAL_GRID.windows:
        out = run_zscore_strategy(prices, "KO", "PEP", fit.beta, entry, window, start).loc[start:]
        stats = summarize(out["ret"], out["position"])
        for h, (s, e) in halves.items():
            part = out.loc[s:e]
            stats[f"sharpe_{h}"] = summarize(part["ret"], part["position"])["sharpe"]
        params = {"entry": entry, "window": window, "exit": 0.0, "beta": round(fit.beta, 6)}
        log_trial(db, "phase2-grid", "KO/PEP", "in-sample", (start.date(), PAIR.in_sample_end), params, stats)
        rows.append({"entry": entry, "window": window, **stats})

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for col in ["sharpe", "sharpe_h1", "sharpe_h2", "n_trades", "ann_return", "max_drawdown", "pct_in_market", "avg_days_per_trade"]:
    print(f"\n{col}\n", res.pivot(index="entry", columns="window", values=col).round(2).to_string())
sharpe = res.pivot(index="entry", columns="window", values="sharpe")
trades = res.pivot(index="entry", columns="window", values="n_trades")
print("\nCHOSEN", select_stable_cell(sharpe, trades, SIGNAL_GRID.min_trades))
inner = sharpe.loc[[1.5, 2.0, 2.5], [60, 120, 190]]
print("zone scores")
import numpy as np
v = sharpe.to_numpy()
print(pd.DataFrame([[v[i-1:i+2, j-1:j+2].mean() for j in range(1, 4)] for i in range(1, 4)], index=[1.5, 2.0, 2.5], columns=[60, 120, 190]).round(3))
print("trials logged:", count_trials(db))

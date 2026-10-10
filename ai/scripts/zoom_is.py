"""Scratch: the 24 pre-registered zoom variants around the chosen cell (diagnostic only)."""
from __future__ import annotations
import math
import pandas as pd
from pairs.backtest import run_zscore_strategy, summarize
from pairs.config import PAIR, SIGNAL_GRID, Settings
from pairs.data import KNOWN_DIVIDEND_ERRORS, correct_dividend_errors, fetch_raw_prices
from pairs.explore import select_until
from pairs.spread import fit_hedge_ratio
from pairs.trials import count_trials, log_trial

CHOSEN = (1.5, 60)
settings = Settings.from_env()
db = settings.results_dir / "trials.sqlite"
raw = select_until(fetch_raw_prices(PAIR, settings.data_dir), PAIR.in_sample_end)
raw = raw.loc[raw.index >= pd.Timestamp(PAIR.start)]
prices = correct_dividend_errors(raw["Adj Close"][["KO", "PEP"]], raw, KNOWN_DIVIDEND_ERRORS).dropna()
fit = fit_hedge_ratio(prices, "KO", "PEP", PAIR.in_sample_end)
start = prices.index[max(SIGNAL_GRID.windows) - 1]
rows = []
for off in SIGNAL_GRID.threshold_offsets:
    for fac in SIGNAL_GRID.window_factors:
        entry, window = round(CHOSEN[0] + off, 2), round(CHOSEN[1] * fac)
        out = run_zscore_strategy(prices, "KO", "PEP", fit.beta, entry, window, start).loc[start:]
        stats = summarize(out["ret"], out["position"])
        if (entry, window) != CHOSEN:
            params = {"entry": entry, "window": window, "exit": 0.0, "beta": round(fit.beta, 6)}
            log_trial(db, "phase2-zoom", "KO/PEP", "in-sample", (start.date(), PAIR.in_sample_end), params, stats, note=f"zoom around {CHOSEN}")
        rows.append({"entry": entry, "window": window, **stats})
res = pd.DataFrame(rows)
for col in ["sharpe", "n_trades", "t_stat"]:
    print(f"\n{col}\n", res.pivot(index="entry", columns="window", values=col).round(2).to_string())
print("\nzoom sharpe min %.2f max %.2f median %.2f" % (res.sharpe.min(), res.sharpe.max(), res.sharpe.median()))
print("trials logged:", count_trials(db), "grid", count_trials(db, "phase2-grid"), "zoom", count_trials(db, "phase2-zoom"))

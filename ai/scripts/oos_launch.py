"""One-off out-of-sample run of the frozen Phase 2 strategy (logged as a trial)."""
from __future__ import annotations
import pandas as pd
from pairs.backtest import run_zscore_strategy, summarize
from pairs.config import PAIR, PHASE2_STRATEGY as S, Settings
from pairs.data import KNOWN_DIVIDEND_ERRORS, correct_dividend_errors, fetch_raw_prices
from pairs.spread import fit_hedge_ratio
from pairs.trials import count_trials, log_trial

settings = Settings.from_env()
db = settings.results_dir / "trials.sqlite"
raw = fetch_raw_prices(PAIR, settings.data_dir)
raw = raw.loc[(raw.index >= pd.Timestamp(PAIR.start)) & (raw.index <= pd.Timestamp(PAIR.end))]
prices = correct_dividend_errors(raw["Adj Close"][["KO", "PEP"]], raw, KNOWN_DIVIDEND_ERRORS).dropna()
fit = fit_hedge_ratio(prices, "KO", "PEP", PAIR.in_sample_end)
assert abs(fit.beta - S.beta) < 1e-6, "beta differs from the frozen value"
oos_start = prices.index[prices.index > pd.Timestamp(PAIR.in_sample_end)][0]
out = run_zscore_strategy(prices, "KO", "PEP", S.beta, S.entry, S.window, oos_start, S.exit).loc[oos_start:]
assert out["position"].iloc[0] == 0 or out["ret"].iloc[0] == 0
stats = summarize(out["ret"], out["position"])
params = {"entry": S.entry, "window": S.window, "exit": S.exit, "beta": S.beta}
log_trial(db, "phase2-oos", "KO/PEP", "out-of-sample", (oos_start.date(), out.index[-1].date()), params, stats, note="single out-of-sample run of the frozen strategy")
print("OOS", oos_start.date(), "->", out.index[-1].date(), len(out), "days")
for k, v in stats.items():
    print(f"  {k}: {v:.4f}")
yearly = out["ret"].groupby(out.index.year).agg(lambda r: (1 + r).prod() - 1)
print("yearly return\n", yearly.round(4).to_string())
print("trials:", count_trials(db))

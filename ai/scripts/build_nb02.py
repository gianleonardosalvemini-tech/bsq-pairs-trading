"""Build notebooks/02_signal.ipynb: Phase 2, the spread, the z-score signal and its validation.

Usage: python build_nb02.py <notebook path>
(The longer first version is kept as build_nb02_long.py.)
"""

import sys

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(
        """# Phase 2 — A mean-reversion signal on the KO / PEP spread, and its validation

The brief asks for an OLS spread `KO − β × PEP`, a check that it comes back to its average, a z-score
signal (long the spread when z is very negative, short when very positive, close near zero) and, most
of all, a careful validation. So I tune in-sample, test once out-of-sample, count what I tried and
check small changes.

**Result**
- The spread check fails. With one β for 2000–2018 the spread drifts in waves lasting years
  (Engle-Granger p = 0.12, half-life ~9 months).
- In-sample (2001–2018) all 25 variants earn a little, but none is significant. The chosen one has
  Sharpe 0.35 and t = 1.49 (1.69–1.78 with autocorrelation-robust errors in A.5, still below 2).
- Out-of-sample (2019–2026), run once with everything frozen, the Sharpe is −0.53. **The signal does
  not survive.** From 2024 KO pulls away from PEP and the strategy keeps betting on a return that
  never comes.

Details are in the appendix. Terms (spread, hedge ratio, z-score, Sharpe ratio, t-statistic, drawdown,
in-sample / out-of-sample, look-ahead) are in the README glossary."""
    ),
    md(
        """**The brief's Phase 2, sentence by sentence**

| The brief asks | What this notebook does | Section |
|---|---|---|
| "Run an OLS linear regression of one price on the other" and "Spread = PriceA − β · PriceB" | OLS of KO on PEP on in-sample adjusted prices, and this same spread (α left out) | 1 |
| "You should actually check it" (plot; does it keep crossing its own average?) | Spread chart, crossings per year, half-life, Engle-Granger test | 1, A.1 |
| "Standardize the spread into a z-score" against "its own rolling average"; long when very negative, short when very positive, close back toward zero | z on a rolling window (past and today only), entry at ±threshold, exit at 0 | 2, A.2 |
| "Split your data into an in-sample and an out-of-sample period ... report both, separately" | 2000–2018 / 2019–2026, fixed before downloading any price. One table with both | top, 7 |
| "Be explicit about how many things you tried" | Every backtest is in `results/trials.sqlite`, 50 in this phase (later phases add their own) | 6 |
| "Check how sensitive your result is to small changes (threshold, lookback window)" | 25-variant grid, the grid on two halves of the in-sample, 24 small changes | 3–5 |
| "Do not optimize on the out-of-sample period" | Rules committed before the first backtest, strategy committed before the single out-of-sample run | 2, 7 |"""
    ),
    code(
        """import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import coint

from pairs.backtest import run_zscore_strategy, select_stable_cell, summarize
from pairs.config import PAIR, PHASE2_STRATEGY, SIGNAL_GRID, Settings
from pairs.data import load_pair_prices
from pairs.logging_setup import configure_logging
from pairs.plots import equity_figure, finish_figure, save_figure
from pairs.spread import build_spread, fit_hedge_ratio, half_life, mean_crossings_per_year
from pairs.trials import count_trials, log_trial

configure_logging(logging.WARNING)  # notebook output: tables and charts only, no progress logs
pio.renderers.default = "plotly_mimetype+png"
settings = Settings.from_env()
figures_dir = settings.results_dir / "figures"
trials_db = settings.results_dir / "trials.sqlite"
SOURCE_IS = "Source: Yahoo Finance adjusted close, in-sample period 3 Jan 2000 - 31 Dec 2018."
SOURCE_ALL = (
    "Source: Yahoo Finance adjusted close, 3 Jan 2000 - 9 Oct 2026."
    " Shaded: out-of-sample (run once)."
)
A, B, PAIR_NAME, S = PAIR.ticker_a, PAIR.ticker_b, f"{PAIR.ticker_a}/{PAIR.ticker_b}", PHASE2_STRATEGY

prices = load_pair_prices(PAIR, settings.data_dir, PAIR.in_sample_end)  # in-sample only: 2000-2018
all_prices = load_pair_prices(PAIR, settings.data_dir, PAIR.end)  # used only for the out-of-sample run
fit = fit_hedge_ratio(prices, A, B, PAIR.in_sample_end)
start = prices.index[max(SIGNAL_GRID.windows) - 1]  # first day with a full 250-day window: same for every variant
oos_start = all_prices.index[all_prices.index > pd.Timestamp(PAIR.in_sample_end)][0]
pd.Series({"in-sample days": len(prices), "in-sample last day": prices.index[-1].date(),
           "backtests start": start.date(), "out-of-sample": f"{oos_start.date()} - {all_prices.index[-1].date()}"})"""
    ),
    md(
        """## Result at a glance

The frozen strategy (threshold 1.5, window 60 days, exit at 0, β = 0.388, chosen in sections 2–5),
first in-sample and then on the out-of-sample period it had never seen.

This notebook shows the result first, so it loads the 2019–2026 prices at the start. The real order of
the work is in the Git history. I committed the rules before the first backtest, then froze the
strategy in a commit, and only after that made a single out-of-sample run (section 7). Every estimate
below (β, grid, choice) uses only in-sample prices: `load_pair_prices(..., PAIR.in_sample_end)` drops
every later date and asserts it."""
    ),
    code(
        """chosen_is = run_zscore_strategy(prices, A, B, fit.beta, S.entry, S.window, start).loc[start:]
is_stats = summarize(chosen_is["ret"], chosen_is["position"])
assert abs(fit_hedge_ratio(all_prices, A, B, PAIR.in_sample_end).beta - S.beta) < 1e-6  # the frozen β
oos = run_zscore_strategy(all_prices, A, B, S.beta, S.entry, S.window, oos_start, S.exit).loc[oos_start:]
oos_stats = summarize(oos["ret"], oos["position"])
log_trial(trials_db, "phase2-oos", PAIR_NAME, "out-of-sample", (oos_start.date(), oos.index[-1].date()),
          {"entry": S.entry, "window": S.window, "exit": S.exit, "beta": S.beta}, oos_stats,
          "single out-of-sample run of the frozen strategy")
fig = equity_figure(pd.concat([chosen_is["ret"], oos["ret"]]),
                    "Chosen variant (1.5, 60 days): in-sample, then the single out-of-sample run",
                    SOURCE_ALL + " Before costs.", shade_from=oos_start)
save_figure(fig, "02_equity_is_oos", figures_dir)
fig.show()
pd.DataFrame({"in-sample 2001-2018": is_stats, "out-of-sample 2019-2026": oos_stats}).round(3)"""
    ),
    md(
        """## 1. The spread, and the brief's check

The two prices wander like random walks. If the stocks are related, some combination of them should
stay around a stable average. KO − PEP does not work because they do not move one-for-one, so I fit
`KO = α + β × PEP` by OLS on in-sample prices only (the function drops every date after 31 Dec 2018,
and a test checks that a later price does not change β). β is then frozen for out-of-sample.

A spread that keeps crossing the dashed line (its 2000–2018 average) is mean-reverting, which the
strategy needs. One that stays on one side for years is drifting. Besides the brief's check (crossings
per year), the table has the half-life, the days a deviation takes to shrink by half (from a regression
of the daily change on the previous level), and the Engle-Granger test. This is the standard test for
a spread built from an estimated β, stricter than a plain Dickey-Fuller test because β was fitted to
make the spread look as stable as possible. A p-value below 0.05 would mean mean-reverting."""
    ),
    code(
        """spread = build_spread(prices, A, B, fit.beta)
fig = go.Figure()
fig.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread KO - 0.388 x PEP", line={"color": "#1f4e9c"}))
fig.add_hline(y=spread.mean(), line_dash="dash", line_color="black", annotation_text="2000-2018 average")
finish_figure(fig, "The spread KO - 0.388 x PEP over the in-sample period", "Date", "Spread (dollars)", SOURCE_IS)
save_figure(fig, "02_spread_is", figures_dir)
fig.show()
crossings = mean_crossings_per_year(spread)
spread_half_life = half_life(spread)
eg_stat, eg_pvalue, eg_critical = coint(prices[A], prices[B])
pd.Series({
    "β (hedge ratio)": round(fit.beta, 4), "alpha (dollars)": round(fit.alpha, 3), "R squared": round(fit.r_squared, 3),
    "crossings of the average, 2000-2018": int(crossings.sum()), "years with no crossing": int((crossings == 0).sum()),
    "half-life (trading days)": round(spread_half_life, 1), "Engle-Granger p-value": round(eg_pvalue, 3),
}).to_frame("in-sample")"""
    ),
    md(
        """**The check fails.** β = 0.388, so KO moves 39 cents per dollar of PEP. The high R² only says both
prices grew together. The spread crosses its average 116 times, but mostly in a few years (23 in 2001,
38 in 2015), and in 8 years out of 19 it never crosses. It moves in waves of several years, below
average from 2004 to 2009 and above from 2011 to 2014. A deviation takes about 192 trading days (9
months) to halve, and Engle-Granger does not reject a drift (p = 0.12). With a single β for 19 years
the long-run relationship is weak.

I still go on, because the brief's z-score uses a rolling average, which follows the slow waves. So
the signal bets on short swings around the recent level, and the validation has to show if that works."""
    ),
    md(
        """## 2. The signal and the rules, fixed before the first backtest

The signal uses only prices known at each close: z = (spread − mean of the last N days) / std of the
last N days. When flat, I go long the spread if z < −threshold and short if z > +threshold, and I
close when z comes back to 0. The position decided at the close of day t earns the move of day t+1, so
there is no look-ahead (tested). The daily return is the P&L of both legs over the dollars invested
the day before (KO value plus PEP value, both positive), 0 without a position. Adjusted prices include
dividends, so the short leg pays them. The brief puts costs in the extensions, so all numbers are
before costs.

I wrote the rules in `src/pairs/config.py` (`SIGNAL_GRID`) and committed them (`feat: pre-register
phase 2 signal grid and selection rule`) **before any backtest**, having seen only the in-sample spread.

| Rule | Value | Why |
|---|---|---|
| Entry thresholds | 1.0, 1.5, 2.0, 2.5, 3.0 | The middle three are the brief's example ("1.5, 2, 2.5"). 1.0 and 3.0 are edges, to see what happens just outside. 1.0 is not really "very negative" and trades almost every swing |
| Windows (days) | 20, 60, 120, 190, 250 | Left free by the brief. With a half-life of ~192 days, short windows bet on swings around the recent average, windows near it on the spread's own pace. 20 = a month, 60 and 120 = a quarter and a half-year (the range the brief uses in Phase 3), 190 ≈ the half-life, 250 = a trading year |
| Exit | z = 0 | The brief: "close the position when it reverts back toward zero". Not tuned, so one thing less tried |
| β | 0.388, in-sample OLS, frozen | The in-sample is "where you look and tune", so β is estimated there and never re-estimated |
| Same period for all variants | from 27 Dec 2000 | First day with a full 250-day window, so shorter windows get no extra year |
| Selection | best zone, not best cell | Below |

Selection rule. The best of 25 cells is just the luckiest one. So for each of the 9 interior cells
(thresholds 1.5/2/2.5 × windows 60/120/190) I average the Sharpe ratio of the cell and its 8
neighbours and take the highest, the centre of a zone that works even if the parameters are a bit off.
Edge cells cannot be chosen: with only 5 or 3 neighbours their average is noisier and the comparison
would not be fair. They still count as neighbours. A cell needs at least 20 in-sample trades (a Sharpe
ratio over a handful of trades is chance), and ties go to fewer trades. Even if every cell lost, the
chosen cell goes out-of-sample and I report the result as it is. I also wrote down in advance that
with 19 years of data an in-sample Sharpe below about 0.4 is compatible with pure luck after 25
attempts.

Sensitivity: after the choice, 24 small changes around it (threshold ±0.1/±0.2, window ±5%/±10%), only
as a diagnostic."""
    ),
    md(
        """## 3. The 25 in-sample variants

Same days for all (27 Dec 2000 – 31 Dec 2018), each variant logged once in `results/trials.sqlite`.
Each square is one variant (row = threshold, column = window), blue for a positive Sharpe ratio, red
for a negative one. A wide blue area means no single lucky combination drives the result. A lone blue
square among red ones would be a warning."""
    ),
    code(
        """HALVES = {"2001-2009": (start, pd.Timestamp("2009-12-31")), "2010-2018": (pd.Timestamp("2010-01-01"), prices.index[-1])}


def run_variant(entry: float, window: int, phase: str, note: str = "") -> dict[str, float]:
    out = run_zscore_strategy(prices, A, B, fit.beta, entry, window, start).loc[start:]
    stats = summarize(out["ret"], out["position"])
    for name, (first, last) in HALVES.items():
        part = out.loc[first:last]
        stats[f"sharpe {name}"] = summarize(part["ret"], part["position"])["sharpe"]
    params = {"entry": entry, "window": window, "exit": 0.0, "beta": round(fit.beta, 6)}
    log_trial(trials_db, phase, PAIR_NAME, "in-sample", (start.date(), PAIR.in_sample_end), params, stats, note)
    return {"entry": entry, "window": window, **stats}


def heatmap(table: pd.DataFrame, title: str, name: str, fmt: str = ".2f", colorscale: str = "RdBu",
            zmid: float | None = 0.0) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=table.to_numpy(), x=[str(c) for c in table.columns], y=[str(i) for i in table.index],
                               colorscale=colorscale, zmid=zmid, texttemplate="%{z:" + fmt + "}", colorbar={"title": ""}))
    finish_figure(fig, title, "Rolling window (trading days)", "Entry threshold (|z|)", SOURCE_IS + " Before costs.")
    fig.update_layout(width=750, height=480)
    save_figure(fig, name, figures_dir)
    return fig


grid = pd.DataFrame([run_variant(e, w, "phase2-grid") for e in SIGNAL_GRID.entry_thresholds for w in SIGNAL_GRID.windows])
sharpe = grid.pivot(index="entry", columns="window", values="sharpe")
trades = grid.pivot(index="entry", columns="window", values="n_trades")
heatmap(sharpe, "In-sample Sharpe ratio of the 25 variants", "02_grid_sharpe").show()
values = sharpe.to_numpy()
zone = pd.DataFrame([[values[i - 1 : i + 2, j - 1 : j + 2].mean() for j in range(1, 4)] for i in range(1, 4)],
                    index=sharpe.index[1:4], columns=sharpe.columns[1:4])
chosen = select_stable_cell(sharpe, trades, SIGNAL_GRID.min_trades)
assert chosen == (S.entry, S.window)  # the frozen strategy is the rule's output
zone.round(3).rename_axis(index="zone score: entry", columns="window")"""
    ),
    md(
        """All 25 variants earn before costs (Sharpe 0.09 to 0.47), so there is no lonely lucky cell. But the
amounts are small (0.3–3% a year) and **no variant reaches t = 2**. The rule picks 1.5 / 60 days (zone
score 0.361). Its own Sharpe ratio, 0.35, is not the best of the grid: the 0.47 sits on an edge with
only 17 trades, which is the kind of pick the rule avoids. It makes 94 trades in 18 years, about 32
days each, and is in a position 66% of the days. One dollar becomes about 1.48 (+2.4%
a year), with a worst drawdown of 14%. Its t of 1.49 is below the luck threshold I wrote in advance.
Even this is optimistic: β comes from the same years and there are no costs."""
    ),
    md(
        """## 4. Is the good zone stable over time?

I score the same 25 backtests on the two halves of the in-sample period. If the blue zone stays in
place, the parameters are stable. If it moves, the "best" parameters are an average of two regimes."""
    ),
    code(
        """fig = make_subplots(rows=1, cols=2, subplot_titles=list(HALVES), horizontal_spacing=0.12)
for col, name in enumerate(HALVES, start=1):
    table = grid.pivot(index="entry", columns="window", values=f"sharpe {name}")
    fig.add_trace(go.Heatmap(z=table.to_numpy(), x=[str(c) for c in table.columns], y=[str(i) for i in table.index],
                             colorscale="RdBu", zmid=0, zmin=-1, zmax=1, texttemplate="%{z:.2f}", showscale=col == 2),
                  row=1, col=col)
finish_figure(fig, "In-sample Sharpe ratio of the 25 variants, first and second half", "Rolling window (trading days)",
              "Entry threshold (|z|)", SOURCE_IS + " Before costs.")
fig.update_layout(width=1000, height=480)
save_figure(fig, "02_grid_halves", figures_dir)
fig.show()"""
    ),
    md(
        """**The good zone moves.** In 2001–2009 short windows work best and long ones lose, and in 2010–2018 it
is the opposite. The chosen variant holds in both halves (0.40 and 0.30), but the "right" parameter
changes across decades. This was a warning for the out-of-sample test."""
    ),
    md(
        """## 5. Small changes of the parameters

The brief says: "If a tiny change collapses the result, that is a sign you found noise." So I ran 24
small changes around the chosen cell (centre), logged as trials and never used to choose again."""
    ),
    code(
        """zoom = pd.DataFrame([
    run_variant(round(chosen[0] + off, 2), round(chosen[1] * fac), "phase2-zoom", f"zoom around {chosen}")
    if (round(chosen[0] + off, 2), round(chosen[1] * fac)) != chosen
    else grid.set_index(["entry", "window"]).loc[chosen].to_dict() | {"entry": chosen[0], "window": chosen[1]}
    for off in SIGNAL_GRID.threshold_offsets for fac in SIGNAL_GRID.window_factors
])
heatmap(zoom.pivot(index="entry", columns="window", values="sharpe"),
        "Sharpe ratio for small changes around the chosen variant (centre: 1.5, 60 days)", "02_zoom_sharpe").show()"""
    ),
    md(
        """Nothing collapses: the Sharpe ratio stays between 0.27 and 0.47 for every small change. The result is
stable but weak, with t-statistics between 1.1 and 2.0."""
    ),
    md(
        """## 6. How many things I tried"""
    ),
    code(
        """pd.Series({"grid (to choose)": count_trials(trials_db, "phase2-grid"),
           "zoom (small changes)": count_trials(trials_db, "phase2-zoom"),
           "out-of-sample runs": count_trials(trials_db, "phase2-oos")}).to_frame("Phase 2 backtests logged")"""
    ),
    md(
        """**50 backtests in total**: 49 in-sample (25 to choose with one rule written in advance, 24 to check
small changes) and 1 out-of-sample. Nothing else: no other exit, stop-loss, β window or regression
direction. I stopped tuning because section 4 shows the best parameters change from one decade to the
next, so more tuning would only describe the past better."""
    ),
    md(
        """## 7. Out-of-sample: the single run, and why it fails

The order of events is in the Git history. (1) I froze the chosen variant in `config.py` as
`PHASE2_STRATEGY` and committed it (`feat: freeze the phase 2 strategy before the out-of-sample run`).
(2) I committed the 49 in-sample trials (`chore: log the 49 in-sample phase 2 trials`). (3) Only then
did I use the 2019–2026 prices, once (`chore: log the single phase 2 out-of-sample run`). The strategy
starts flat on 2 Jan 2019, so no in-sample trade spills over, and the z-score of early January uses
the last 60 days of 2018, which is allowed. The reading was pre-registered too: the strategy survives
if its out-of-sample Sharpe ratio is positive. The run is in the chart at the top. This notebook
re-executes it with the same frozen values, so the trial log does not count it twice.

**It does not survive.** Sharpe −0.53, −3.8% a year, worst drawdown 32%, mostly in 2025 (−15%) and
2026 (−9%). With t = −1.49 it is not a proven loser either, since 7.8 years cannot separate a small
effect from zero. What the run shows is no evidence that it makes money. Before blaming the market I
looked for a mistake: no prices are missing, the largest moves are the real ones of March 2020, trades
per year match the in-sample, and β re-estimated equals the frozen one."""
    ),
    code(
        """yearly = oos["ret"].groupby(oos.index.year).apply(lambda r: (1 + r).prod() - 1)
full_spread = build_spread(all_prices, A, B, S.beta)
fig = go.Figure()
fig.add_vrect(x0=oos_start, x1=full_spread.index[-1], fillcolor="#f39c12", opacity=0.12, line_width=0)
fig.add_trace(go.Scatter(x=full_spread.index, y=full_spread, name="Spread KO - 0.388 x PEP", line={"color": "#1f4e9c"}))
fig.add_hline(y=spread.mean(), line_dash="dash", line_color="black", annotation_text="2000-2018 average")
finish_figure(fig, "The spread with the frozen β, 2000-2026: it drifts away after 2024", "Date", "Spread (dollars)",
              SOURCE_ALL)
save_figure(fig, "02_spread_full", figures_dir)
fig.show()
held = oos["position"].shift(1).fillna(0)
ratio = all_prices[A] / all_prices[B]
pd.Series({
    "return 2025": f"{yearly[2025]:.1%}", "return 2026": f"{yearly[2026]:.1%}",
    "P&L of long trades ($ per KO share)": round(oos.loc[held > 0, "pnl"].sum(), 2),
    "P&L of short trades ($ per KO share)": round(oos.loc[held < 0, "pnl"].sum(), 2),
    "KO/PEP price ratio, 2023 average": round(ratio.loc["2023"].mean(), 3),
    "KO/PEP price ratio, 2026 average": round(ratio.loc["2026"].mean(), 3),
    "KO/PEP price ratio, lowest yearly average 2019-2026": ratio.loc["2019":].groupby(ratio.loc["2019":].index.year).mean().idxmin(),
    "spread, 2023 average ($)": round(full_spread.loc["2023"].mean(), 1),
    "spread, 2026 average ($)": round(full_spread.loc["2026"].mean(), 1),
    "spread, last day ($)": round(full_spread.iloc[-1], 1),
}).to_frame("out-of-sample")"""
    ),
    md(
        """Why it fails: from 2024 Coca-Cola grows much faster than PepsiCo. The price ratio goes from 0.35 (2023
average, its lowest out-of-sample) to 0.56 (2026 average), and the spread's yearly average climbs from
about −7 to +24 dollars (+39 on the last day), far above anything seen in 2000–2018. The strategy keeps
shorting the spread, betting the climb will reverse, and the rolling average catches up at a loss.
Most of the loss comes from the short trades. This is the drift the brief describes in Phase 3, and it
fits Phase 1, where the KO/PEP ratio moved in waves lasting decades."""
    ),
    md(
        """## 8. What I learn, and what I am unsure about

**What I learn**
- The brief's spread check fails for KO/PEP with a static β: waves lasting years, half-life ~9
  months, Engle-Granger p = 0.12.
- In-sample the z-score signal earns a little in every variant and survives small changes, but it is
  weak (Sharpe 0.35, t = 1.49) and its best parameters move between decades.
- **50 backtests, rules fixed in advance, one out-of-sample run: the signal did not survive.** I think
  the in-sample gains mostly described the swings of that period, and out-of-sample the relationship
  drifted for years while the strategy kept betting against the drift.

**What I am unsure about**
- In-sample numbers are optimistic: β uses the same years, and there are no costs.
- 7.8 out-of-sample years are few. Even a true Sharpe ratio of 0.5 could look like zero there.
- One pair and one split. Phase 5 (simulated paths) and the walk-forward extension test whether the
  conclusion depends on this slice of history.
- β estimated on price levels gives more weight to the later, higher prices.
- A hypothesis, not a result: a β that moves with the relationship (Kalman, extension) might keep the
  spread closer to a stable average. Phase 3 weakens it, because over 2024–2026 the 2-year β turns
  negative (−0.23 in August 2025). The stocks moved in opposite directions, and no positive hedge
  ratio makes such a spread flat. A moving β could at most shrink the losing bet, and it also absorbs
  part of the divergences the strategy wants to trade. It is untested here and would need its own rules
  and out-of-sample run, without touching the strategy frozen here."""
    ),
    md(
        """## Appendix: supporting details

**A.1 Crossings of the in-sample average, year by year** (the brief's basic check)."""
    ),
    code(
        """fig = go.Figure(go.Bar(x=crossings.index, y=crossings, marker_color="#1f4e9c", name="crossings"))
finish_figure(fig, "How many times a year the spread crosses its 2000-2018 average", "Year", "Crossings (number)",
              SOURCE_IS)
fig.update_xaxes(dtick=1)
save_figure(fig, "02_mean_crossings", figures_dir)
fig.show()"""
    ),
    md(
        """**A.2 The signal at work, 2002–2003.** The line is z. A trade opens when z crosses ±1.5 (triangles)
and closes when it comes back to 0 (crosses)."""
    ),
    code(
        """example = run_zscore_strategy(prices, A, B, fit.beta, S.entry, S.window, start).loc["2002":"2003"]
pos, prev = example["position"], example["position"].shift(1).fillna(0)
fig = go.Figure()
fig.add_trace(go.Scatter(x=example.index, y=example["z"], name="z-score (60 days)", line={"color": "#555555"}))
for level in (-S.entry, S.entry):
    fig.add_hline(y=level, line_dash="dash", line_color="#888888")
for label, mask, symbol, color in [
    ("open long", (pos == 1) & (prev != 1), "triangle-up", "#2e8b57"),
    ("open short", (pos == -1) & (prev != -1), "triangle-down", "#c0392b"),
    ("close", (pos == 0) & (prev != 0), "x", "black"),
]:
    fig.add_trace(go.Scatter(x=example.index[mask], y=example["z"][mask], mode="markers", name=label,
                             marker={"symbol": symbol, "size": 11, "color": color}))
finish_figure(fig, "The signal at work, 2002-2003: entries at z = ±1.5, exits at z = 0", "Date",
              "z-score (standard deviations)", SOURCE_IS)
save_figure(fig, "02_signal_example", figures_dir)
fig.show()"""
    ),
    md(
        """**A.3 Number of trades and full metrics of the 25 variants.** Short windows trade a lot (303 trades
with 1.0 / 20 days). Long windows with high thresholds trade very little (12 with 3.0 / 250, too few to
trust)."""
    ),
    code(
        """heatmap(trades, "Number of trades of the 25 variants (2001-2018)", "02_grid_trades", fmt=".0f",
        colorscale="Greys", zmid=None).show()
grid.set_index(["entry", "window"])[["sharpe", "t_stat", "ann_return", "max_drawdown", "n_trades",
                                     "pct_in_market", "avg_days_per_trade"]].round(2)"""
    ),
    md("""**A.4 The chosen variant on the in-sample period alone.**"""),
    code(
        """fig = equity_figure(chosen_is["ret"], "Chosen variant (1.5, 60 days), in-sample, before costs", SOURCE_IS)
save_figure(fig, "02_equity_is", figures_dir)
fig.show()"""
    ),
    md(
        """**A.5 How much to trust the t-statistic.** The t-statistic in the tables is the Sharpe ratio times the
square root of the years, so it treats every day as independent. A trade lasts about 32 days, so daily
returns could be correlated and that t could overstate the evidence. I recompute it with Newey-West
standard errors, which allow for correlation between days up to a chosen lag. I chose the lags (10,
20, 60, 120 days) after the out-of-sample run, so I show all four and pick none. It is the same
backtest with a different error bar, not a new trial."""
    ),
    code(
        """def newey_west_t(returns: pd.Series, lags: int) -> float:
    model = sm.OLS(returns.to_numpy(), np.ones(len(returns))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(model.tvalues[0])


NW_LAGS = (10, 20, 60, 120)
robust_t = pd.DataFrame({
    "in-sample 2001-2018": {"independent days (tables above)": is_stats["t_stat"],
                            **{f"Newey-West, {lag} lags": newey_west_t(chosen_is["ret"], lag) for lag in NW_LAGS}},
    "out-of-sample 2019-2026": {"independent days (tables above)": oos_stats["t_stat"],
                                **{f"Newey-West, {lag} lags": newey_west_t(oos["ret"], lag) for lag in NW_LAGS}},
})
daily_autocorr = chosen_is["ret"].autocorr(1)
robust_t.round(2)"""
    ),
    md(
        """The correction goes the opposite way from what I feared. Daily returns are barely correlated
(first-lag autocorrelation −0.03 in-sample), and the robust in-sample t is 1.69 to 1.78, slightly
higher than 1.49 and still below 2. Out-of-sample it is −1.47 to −1.64. So the verdict stays the same:
neither the in-sample gain nor the out-of-sample loss can be told apart from zero."""
    ),
    code(
        """# Every number quoted in the text, checked (expected values recorded on 10 Oct 2026)
assert len(prices) == 4779 and prices.index[-1] == pd.Timestamp("2018-12-31")
assert abs(fit.beta - 0.388372) < 1e-6 and abs(fit.alpha - 2.408) < 1e-3 and abs(fit.r_squared - 0.927) < 1e-3
assert int(crossings.sum()) == 116 and crossings[2001] == 23 and crossings[2015] == 38 and (crossings == 0).sum() == 8
assert abs(spread_half_life - 192.0) < 0.5 and abs(eg_pvalue - 0.117) < 0.005
assert start == pd.Timestamp("2000-12-27")
assert len(grid) == 25 and (grid["sharpe"] > 0).all()
assert abs(grid["sharpe"].min() - 0.09) < 0.005 and abs(grid["sharpe"].max() - 0.47) < 0.006
assert (grid["t_stat"] < 2).all()
assert chosen == (1.5, 60) and abs(zone.loc[1.5, 60] - 0.361) < 0.001
g = grid.set_index(["entry", "window"])
assert abs(is_stats["sharpe"] - 0.35) < 0.005 and abs(is_stats["t_stat"] - 1.49) < 0.005 and is_stats["n_trades"] == 94
assert abs(is_stats["max_drawdown"] + 0.14) < 0.005 and abs(is_stats["ann_return"] - 0.024) < 0.001
assert abs((1 + chosen_is["ret"]).prod() - 1.48) < 0.005
assert abs(grid["ann_return"].min() - 0.003) < 0.0005 and abs(grid["ann_return"].max() - 0.031) < 0.0005
assert g.loc[(1.0, 20), "n_trades"] == 303 and g.loc[(3.0, 250), "n_trades"] == 12 and g.loc[(3.0, 120), "n_trades"] == 17
assert abs(g.loc[chosen, "sharpe 2001-2009"] - 0.40) < 0.005 and abs(g.loc[chosen, "sharpe 2010-2018"] - 0.30) < 0.005
assert len(zoom) == 25 and abs(zoom["sharpe"].min() - 0.27) < 0.005 and abs(zoom["sharpe"].max() - 0.47) < 0.005
assert zoom["t_stat"].min() > 1.05 and zoom["t_stat"].max() < 2.0
assert count_trials(trials_db, "phase2-grid") == 25 and count_trials(trials_db, "phase2-zoom") == 24
assert count_trials(trials_db, "phase2-oos") == 1
assert oos_start == pd.Timestamp("2019-01-02") and oos.index[-1] == pd.Timestamp("2026-10-09")
assert abs(oos_stats["sharpe"] + 0.53) < 0.005 and abs(oos_stats["ann_return"] + 0.038) < 0.001
assert abs(oos_stats["t_stat"] + 1.49) < 0.005 and abs(oos_stats["max_drawdown"] + 0.32) < 0.005
assert oos_stats["n_trades"] == 36
assert abs(yearly[2025] + 0.154) < 0.001 and abs(yearly[2026] + 0.086) < 0.001
assert oos.loc[held < 0, "pnl"].sum() < oos.loc[held > 0, "pnl"].sum() < 0
assert abs(ratio.loc["2023"].mean() - 0.35) < 0.01 and abs(ratio.loc["2026"].mean() - 0.56) < 0.01
assert ratio.loc["2019":].groupby(ratio.loc["2019":].index.year).mean().idxmin() == 2023
assert round(full_spread.loc["2023"].mean()) == -7 and round(full_spread.loc["2026"].mean()) == 24
assert round(full_spread.iloc[-1]) == 39
rt = robust_t.round(2)
assert rt.iloc[1:, 0].min() == 1.69 and rt.iloc[1:, 0].max() == 1.78 and (rt.iloc[1:, 0] < 2).all()
assert rt.iloc[1:, 1].min() == -1.64 and rt.iloc[1:, 1].max() == -1.47
assert round(daily_autocorr, 2) == -0.03
for name in ("02_spread_is", "02_mean_crossings", "02_signal_example", "02_grid_sharpe", "02_grid_trades",
             "02_grid_halves", "02_zoom_sharpe", "02_equity_is", "02_equity_is_oos", "02_spread_full"):
    assert (figures_dir / f"{name}.png").exists(), name"""
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (.venv)", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, sys.argv[1])

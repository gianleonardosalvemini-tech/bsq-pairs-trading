"""Build notebooks/05_simulation.ipynb: Phase 5, simulating the pair.

Usage: python build_nb05.py <notebook path>
"""

import sys

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(
        """# Phase 5 — Simulating the pair

**What the brief asks.** Real history is one path. I generate other plausible joint paths of the two stocks
(keeping how they move together), re-run the Phase 2 signal on each and report how its performance varies,
as evidence for or against "a real, robust effect rather than a fit to one specific historical accident".

**Result.** Joint block bootstrap (the brief's "simple" option): 1,000 synthetic 19-year paths of KO and PEP
built from 10-day blocks of real in-sample days, the same days for both stocks.
1. The spread does pull back in 2000–2018, over weeks to months, but not to a fixed level. Only 0.5% of
   reshuffled paths pull back as much over 60 days (section 4). I added this test, and the one-day blocks it
   uses, after seeing the first results, so it is exploratory.
2. The signal is usually positive but small: median Sharpe 0.17 on the paths, 74% positive. The 0.35 of
   Phase 2 is a good draw, and the real history was unusually gentle (drawdown −14% against a median of −32%).
3. The bootstrap cannot contain a change of regime like the 2024–2026 divergence that sank the strategy
   out-of-sample.

**Answer to the brief:** the pull-back is not an accident of one path, but the strategy's profit from it is
too small to tell apart from luck. I find no evidence of a robust money-maker, which fits its out-of-sample
loss.

Only in-sample data (2000–2018) is used; the out-of-sample years are never resampled. Terms: README glossary."""
    ),
    md(
        """## 1. Method, and why I added a second one

My main method is a joint block bootstrap. I cut the in-sample history into blocks of consecutive trading
days, draw blocks at random with replacement and glue them into a new 19-year history. Each block comes from
both stocks at once, so on every synthetic day KO and PEP move as they really moved together on some real
day. No model is needed.

The "medium" option, an Ornstein-Uhlenbeck spread, assumes the spread is pulled back to a long-run average and
estimates the pull from the real spread. Phase 2 did not establish that pull (Engle-Granger p = 0.12,
half-life 192 days), so simulating from it would put into the paths the thing I want to test. So it cannot
be my main evidence.

I still ran it as a declared extra check (the brief says "pick one"), because the two methods describe
opposite worlds. In the bootstrap there is no long-run relationship: blocks of 10 days keep the daily
co-movement and anything within two weeks, but they are drawn independently, so a pull-back over months cannot
survive. In the OU the relationship is real, with the strength I measured. If the real Sharpe looks ordinary
in both worlds, the in-sample result says nothing about the relationship. The OU is one extra configuration
in the trial log, reported separately.

**These choices were fixed before computing anything**, together with a prediction: a centre between 0 and
0.1, the real 0.35 around the 85th–90th percentile, lower once the selection is repeated. The centre turned
out higher (section 4).

| Choice | Value | Why |
|---|---|---|
| Days resampled | In-sample only, 2000-01-03 to 2018-12-31 (`as_of`) | The out-of-sample was used once in Phase 2 and stays untouched; the 0.35 to compare is in-sample |
| Path | 4,779 days, the in-sample calendar, real prices on day one | Same length and evaluation period as Phase 2 |
| Block length | 10 days, with 5 and 20 as sensitivity | The middle and the two ends of the brief's 5–20 |
| Paths, seed | 1,000 per configuration, seed 20261010 | Stable 5th and 95th percentiles, reproducible |
| Hedge ratio | Re-estimated by OLS on each path | Phase 2 estimated it on the data it traded, so I do the same; 0.388 means nothing on other prices |
| Rules | Frozen: entry 1.5, window 60, exit 0, evaluated from the 250th day | The Phase 2 strategy, unchanged |
| Selection | Second analysis: the 25-cell grid and the stable-zone rule re-run on each path | The real 0.35 won a selection among 25 cells, so I compare winner against winner |"""
    ),
    code(
        """import logging
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display

from pairs.config import PAIR, PHASE2_STRATEGY, SIGNAL_GRID, Settings
from pairs.data import load_pair_prices
from pairs.logging_setup import configure_logging
from pairs.plots import finish_figure, save_figure
from pairs.simulate import (
    fit_ou,
    frozen_rules_on_path,
    selected_rules_on_path,
    simulate_joint_prices,
    simulate_ou,
    spread_pnl_sharpe,
)
from pairs.spread import build_spread, fit_hedge_ratio
from pairs.trials import count_trials, log_trial

configure_logging(logging.WARNING)
pio.renderers.default = "plotly_mimetype+png"
settings = Settings.from_env()
figures_dir = settings.results_dir / "figures"
trials_db = settings.results_dir / "trials.sqlite"
A, B, S, AS_OF = PAIR.ticker_a, PAIR.ticker_b, PHASE2_STRATEGY, PAIR.in_sample_end
SOURCE = "Source: Yahoo Finance adjusted close, in-sample 2000-2018; 1,000 simulated paths per case."

BLOCKS = (5, 10, 20)  # trading days; 10 is the main case
EXPLORATORY_BLOCK = 1  # added after seeing the results (section 4)
MAIN_BLOCK = 10
N_PATHS = 1000
SEED = 20261010
START = max(SIGNAL_GRID.windows) - 1  # first evaluated day, as in Phase 2 (the 250th)

prices = load_pair_prices(PAIR, settings.data_dir, AS_OF)  # in-sample only
real = frozen_rules_on_path(prices, A, B, S.entry, S.window, START, AS_OF)
print(f"Real in-sample: Sharpe {real['sharpe']:.3f}, beta {real['beta']:.4f}, {len(prices)} days")"""
    ),
    md(
        """The line above re-runs Phase 2 on the real data with the same function I use on every path. It gives
the Phase 2 Sharpe (0.35) and hedge ratio (0.388), so paths and history are measured the same way.

The simulation takes about four minutes, so I save it once as a checkpoint (`data/phase5_simulation.pkl`) and
re-read it. Deleting the file recomputes it with the same seed."""
    ),
    code(
        """def simulate_all() -> dict[str, pd.DataFrame]:
    out = {}
    for block in (EXPLORATORY_BLOCK, *BLOCKS):
        rng = np.random.default_rng([SEED, block])
        rows = []
        for _ in range(N_PATHS):
            path = simulate_joint_prices(prices, block, rng, AS_OF)
            row = frozen_rules_on_path(path, A, B, S.entry, S.window, START, AS_OF)
            ret = path.pct_change().iloc[1:]
            row["corr"] = float(ret[A].corr(ret[B]))
            if block == MAIN_BLOCK:
                spread = build_spread(path, A, B, row["beta"])
                row["dollar_sharpe"] = spread_pnl_sharpe(spread, S.entry, S.window, START)
                sel = selected_rules_on_path(path, A, B, SIGNAL_GRID, AS_OF)
                row.update({f"selected_{k}": v for k, v in sel.items()})
            rows.append(row)
        out[f"bootstrap_{block}"] = pd.DataFrame(rows)
    ou_fit = fit_ou(build_spread(prices, A, B, S.beta), AS_OF)
    rng = np.random.default_rng([SEED, 0])
    first_value = float(build_spread(prices, A, B, S.beta).iloc[0])
    out["ou"] = pd.DataFrame(
        {
            "dollar_sharpe": [
                spread_pnl_sharpe(
                    pd.Series(simulate_ou(ou_fit, first_value, len(prices), rng), index=prices.index),
                    S.entry,
                    S.window,
                    START,
                )
                for _ in range(N_PATHS)
            ]
        }
    )
    return out


checkpoint = settings.data_dir / "phase5_simulation.pkl"
if checkpoint.exists():
    with checkpoint.open("rb") as fh:
        sims = pickle.load(fh)
else:
    sims = simulate_all()
    with checkpoint.open("wb") as fh:
        pickle.dump(sims, fh)
main = sims[f"bootstrap_{MAIN_BLOCK}"]
assert all(len(v) == N_PATHS for v in sims.values())"""
    ),
    md(
        """## 2. Do the synthetic paths look like KO and PEP?

A bootstrap is only useful if each path is a believable joint history. I compare the 1,000 paths with
10-day blocks to the real in-sample on three checks, then I draw three paths as an example."""
    ),
    code(
        """ret = prices.pct_change().iloc[1:]
checks = pd.DataFrame(
    {
        "real": [ret[A].corr(ret[B]), real["beta"], real["n_trades"]],
        "paths: median": [main["corr"].median(), main["beta"].median(), main["n_trades"].median()],
        "paths: 5%": [main["corr"].quantile(0.05), main["beta"].quantile(0.05), main["n_trades"].quantile(0.05)],
        "paths: 95%": [main["corr"].quantile(0.95), main["beta"].quantile(0.95), main["n_trades"].quantile(0.95)],
    },
    index=["Daily return correlation KO-PEP", "Hedge ratio (OLS on the path)", "Trades of the frozen rules"],
)
checks.round(2)"""
    ),
    code(
        """example_rng = np.random.default_rng([SEED, 99])  # separate stream: only for this picture
fig = go.Figure()
fig.add_trace(go.Scatter(x=prices.index, y=prices[A] / prices[B], name="Real history", line={"color": "black", "width": 2}))
for k, color in enumerate(["#1f77b4", "#ff7f0e", "#2ca02c"], start=1):
    path = simulate_joint_prices(prices, MAIN_BLOCK, example_rng, AS_OF)
    fig.add_trace(go.Scatter(x=path.index, y=path[A] / path[B], name=f"Simulated path {k}", line={"color": color, "width": 1}))
finish_figure(
    fig,
    "KO/PEP price ratio: real in-sample history and three simulated paths (10-day blocks)",
    "Date (the simulated paths use the in-sample calendar)",
    "Price ratio KO / PEP",
    SOURCE,
)
save_figure(fig, "05_example_paths", figures_dir)
fig.show()"""
    ),
    md(
        """The co-movement survives. On 90% of the paths the daily correlation is between 0.50 and 0.60, around
the real 0.55, and the frozen rules trade about as often as in reality (median 98 trades against 94). The
long-run tie does **not** survive. The real ratio went from 0.73 in January 2000 to a low of 0.33 in 2005 and
0.43 at the end of 2018, while the simulated ratios go wherever their blocks take them. On 90% of the paths
the re-estimated hedge ratio lies between about 0 and 1.4 instead of near 0.39, and 6% even get a negative
one. So KO and PEP move together day by day but are not tied together over years, which is what the test
needs."""
    ),
    md(
        """## 3. Main result: the Phase 2 signal on 1,000 paths

In the chart each bar counts the paths whose Sharpe ratio fell in that range, and the black line is the real
in-sample Sharpe. If the line were far to the right of almost all bars, the real history would contain
something the paths do not."""
    ),
    code(
        """def describe(values: pd.Series, real_value: float) -> dict[str, float]:
    v = values.dropna()
    return {
        "paths": float(len(v)),
        "median": float(v.median()),
        "p05": float(v.quantile(0.05)),
        "p95": float(v.quantile(0.95)),
        "share_positive": float((v > 0).mean()),
        "real_value": real_value,
        "real_percentile": float((v < real_value).mean()),
    }


main_stats = describe(main["sharpe"], real["sharpe"])
fig = go.Figure(go.Histogram(x=main["sharpe"], nbinsx=50, marker_color="#7f8c8d", name="Simulated paths"))
fig.add_vline(x=real["sharpe"], line_color="black", line_width=3)
fig.add_annotation(x=real["sharpe"], y=1, yref="paper", text=f"Real in-sample: {real['sharpe']:.2f}", showarrow=False, xanchor="left", yanchor="top")
for q in ("p05", "p95"):
    fig.add_vline(x=main_stats[q], line_dash="dot", line_color="#7f8c8d")
finish_figure(
    fig,
    "Sharpe ratio of the frozen Phase 2 strategy on 1,000 bootstrap paths (10-day blocks)",
    "Annualised Sharpe ratio on the path (dotted: 5th and 95th percentiles)",
    "Number of paths",
    SOURCE,
)
save_figure(fig, "05_bootstrap_sharpe", figures_dir)
fig.show()
pd.Series(main_stats).round(3)"""
    ),
    md(
        """With history reshuffled in two-week pieces, the same strategy gives a Sharpe anywhere between −0.23
and +0.53 (90% of paths), with a median of 0.17, and 26% of paths lose money. The real 0.35 beats 80% of the
paths. It was a good draw, but one path in five did better just by reshuffling. With the usual 5% standard,
the real result **cannot be told apart** from reshuffled histories."""
    ),
    md(
        """The Sharpe ratio is only one number. The table below shows every Phase 2 metric across the same 1,000
paths, next to the real in-sample value. The last column is the share of paths worse than reality (lower
return, Sharpe or t-statistic, higher volatility, deeper drawdown). These are the same runs, no new
backtests."""
    ),
    code(
        """metrics = {
    "ann_return": ("Annual return", "higher"),
    "ann_vol": ("Annual volatility", "lower"),
    "sharpe": ("Sharpe ratio", "higher"),
    "t_stat": ("t-statistic of the mean daily return", "higher"),
    "max_drawdown": ("Maximum drawdown", "higher"),
    "n_trades": ("Number of trades", None),
}
rows = {}
for col, (label, better) in metrics.items():
    v = main[col]
    worse = {"higher": (v < real[col]).mean(), "lower": (v > real[col]).mean(), None: np.nan}[better]
    rows[label] = {
        "real": real[col],
        "paths: 5%": v.quantile(0.05),
        "paths: median": v.median(),
        "paths: 95%": v.quantile(0.95),
        "share of paths worse than real": worse,
    }
performance = pd.DataFrame(rows).T
near_beta = main.loc[(main["beta"] - real["beta"]).abs() < 0.1, "max_drawdown"]  # added after seeing the table
display(pd.Series({"paths with hedge ratio within 0.1 of the real one": len(near_beta),
                   "their median maximum drawdown": round(near_beta.median(), 3)}).to_frame("check"))
performance.round(3)"""
    ),
    md(
        """On the typical path the strategy earns 1.7% a year (real: 2.4%), and one path in four loses money over
the 19 years. Only 8% of paths reach a t-statistic of 2, the usual bar for "not luck". The drawdown surprised
me most. The real worst loss from a peak was −14%, shallower than on 99% of paths, where the median is −32%
and the worst 5% lose about 60% or more. The real volatility is lower than on all 1,000 paths. This is the
in-sample pull-back at work: the real spread came back before a losing position could run for years, while
without the long-run tie it can trend against the position for a long time.

After seeing the table I added one check: is the deeper drawdown just a worse hedge, since the paths' hedge
ratios vary widely? No. On the paths whose hedge ratio is within 0.1 of 0.388 the median drawdown is still
−31%.

So the in-sample backtest hid something: a strategy that looks gentle (−14%) loses a third of its peak value
in most comparable worlds. Out-of-sample, when the relationship broke, the drawdown was −32%, the typical
value on the paths (over 7.8 years instead of 19, so only an indication and not a test)."""
    ),
    md(
        """## 4. Block length, and where the positive median comes from

I predicted a centre near zero, but it came out at 0.17. There were two possible explanations, so after
seeing the results I added one exploratory configuration, logged separately in the trial log: blocks of one
day, which keep only the same-day co-movement and nothing of the following days.
1. If the positive centre came from estimating the hedge ratio on the same path it is traded on (a known
   flattering effect, also present in the real 0.35), it would remain with 1-day blocks.
2. If it came from short-term pull-backs of the spread kept inside each block, it would vanish."""
    ),
    code(
        """summary = {
    f"Bootstrap, {b}-day blocks{' (exploratory)' if b == EXPLORATORY_BLOCK else ''}": describe(sims[f"bootstrap_{b}"]["sharpe"], real["sharpe"])
    for b in (EXPLORATORY_BLOCK, *BLOCKS)
}
blocks_table = pd.DataFrame(summary).T
spread_real = build_spread(prices, A, B, S.beta)
daily_var = spread_real.diff().var()
variance_ratio = pd.Series(
    {h: spread_real.diff(h).var() / (h * daily_var) for h in (5, 10, 20, 60, 120)}, name="real spread"
)


def path_variance_ratio(block: int, rng: np.random.Generator, h: int = 60) -> float:
    path = simulate_joint_prices(prices, block, rng, AS_OF)
    spread = build_spread(path, A, B, fit_hedge_ratio(path, A, B, AS_OF).beta)
    return float(spread.diff(h).var() / (h * spread.diff().var()))


vr_paths = {}
for b in (EXPLORATORY_BLOCK, MAIN_BLOCK):
    vr_rng = np.random.default_rng([SEED, 100 + b])  # separate stream, a few seconds
    vr_paths[b] = pd.Series([path_variance_ratio(b, vr_rng) for _ in range(N_PATHS)])
vr_null = pd.DataFrame(
    {
        f"{b}-day blocks": {
            "median": vr_paths[b].median(),
            "share of paths <= real": (vr_paths[b] <= variance_ratio[60]).mean(),
        }
        for b in vr_paths
    }
)
display(blocks_table[["median", "p05", "p95", "share_positive", "real_percentile"]].round(3))
display(variance_ratio.round(2).to_frame().T)
vr_null.round(3)"""
    ),
    md(
        """The median Sharpe grows with the block length: 0.02 with 1-day blocks, 0.10 with 5, 0.17 with 10 and
0.27 with 20. So the hedge-ratio effect (explanation 1) is negligible, and the signal earns from something
that only longer blocks keep.

The second table shows what. **Like the 1-day blocks, this test was decided after seeing the results. It was
not planned, and both of its comparisons are logged as exploratory.** A variance ratio compares how much the
spread moves over h days with h times its daily move: 1 for a random walk, below 1 when moves are partly
undone later. For the real spread it falls from 0.92 at 5 days to 0.76 at 20 and 0.61 at 60 days.

Is that more than chance? I test only 60 days, the signal's window fixed in Phase 2. The third table computes
the 60-day ratio on 1,000 new paths of each kind, with the hedge ratio re-estimated on each path as for the
real spread (a spread fitted on its own data looks slightly mean-reverting by construction). With 1-day
blocks the median is 0.96 and only 0.5% of paths go as low as 0.61. With 10-day blocks the median is 0.82 and
6% go as low. So the pull-back over weeks to months is real in-sample, and 10-day blocks keep only part of
it, which is why longer blocks earn more.

But the signal extracts little profit from it. Even against the 1-day world, with no pull-back at all, the
real 0.35 reaches only the 92nd percentile, below the usual 95%, and its rank falls as more of the real
dynamics is kept (87th, 80th, 64th)."""
    ),
    md(
        """## 5. Accounting for the selection among 25 variants

In Phase 2 I backtested 25 combinations of threshold and window, and the pre-registered stable-zone rule
picked one. A chosen winner looks better than a single variant fixed in advance. So on each of the 1,000
paths (10-day blocks) I re-ran the whole Phase 2 procedure, all 25 cells and the same rule, and kept the
Sharpe of the cell the rule picks."""
    ),
    code(
        """selection_stats = describe(main["selected_sharpe"], real["sharpe"])
picked = pd.DataFrame(
    {
        "chosen threshold (share of paths)": main["selected_entry"].value_counts(normalize=True).sort_index(),
        "chosen window (share of paths)": main["selected_window"].value_counts(normalize=True).sort_index(),
    }
)
display(pd.Series(selection_stats).round(3))
picked.round(2)"""
    ),
    md(
        """With the selection repeated, the median rises from 0.17 to 0.25 and 86% of paths are positive: tuning
on the data finds something positive almost every time, even in reshuffled histories. The real 0.35 falls to
the **67th percentile**. The rule also picks entry 1.5 and window 60, the Phase 2 cells, more often than any
other (on 48% and 57% of paths). So the real choice is the typical one and does not point to a special
structure in the real data."""
    ),
    md(
        """## 6. Extra check: a world in which the relationship is real (Ornstein-Uhlenbeck)

An Ornstein-Uhlenbeck (OU) process is a random walk with a spring. Every day the spread is pulled back
toward a long-run average by a fixed fraction of its distance from it, plus a random shock. Sampled once a
day it is the regression of today's spread on yesterday's, so I estimate it on the real in-sample spread
(KO − 0.388 × PEP, with the Phase 2 hedge ratio) and simulate 1,000 spread paths of the same length.

An OU path gives only the spread, not the two prices, so I cannot compute returns on capital. Instead I
measure the Sharpe of the daily dollar profit of holding one unit of the spread, on the OU paths, on the
bootstrap paths (10-day blocks, each with its own hedge ratio) and on the real history."""
    ),
    code(
        """ou = fit_ou(spread_real, AS_OF)
real_dollar = spread_pnl_sharpe(spread_real, S.entry, S.window, START)
ou_variance_ratio_60 = (1 - ou.phi**60) / (60 * (1 - ou.phi))  # stationary OU, 60 days
worlds = pd.DataFrame(
    {
        "Bootstrap, 10-day blocks (no long-run tie)": describe(main["dollar_sharpe"], real_dollar),
        "OU (long-run tie, as estimated)": describe(sims["ou"]["dollar_sharpe"], real_dollar),
    }
).T
print(
    f"OU: mean {ou.mu:.2f} $, daily pull {1 - ou.phi:.4f}, half-life {ou.half_life:.0f} days, "
    f"daily shock {ou.sigma:.3f} $; 60-day variance ratio {ou_variance_ratio_60:.2f} "
    f"(real {variance_ratio[60]:.2f})"
)
worlds[["median", "p05", "p95", "share_positive", "real_value", "real_percentile"]].round(3)"""
    ),
    code(
        """fig = go.Figure()
for name, values, color in [
    ("Bootstrap, 1-day blocks (exploratory)", sims[f"bootstrap_{EXPLORATORY_BLOCK}"]["sharpe"], "#bdc3c7"),
    ("Bootstrap, 5-day blocks", sims["bootstrap_5"]["sharpe"], "#95a5a6"),
    ("Bootstrap, 10-day blocks (main)", main["sharpe"], "#7f8c8d"),
    ("Bootstrap, 20-day blocks", sims["bootstrap_20"]["sharpe"], "#5d6d7e"),
    ("Bootstrap 10, grid + selection rule", main["selected_sharpe"], "#2e86c1"),
    ("OU, dollar P&L (relationship real)", sims["ou"]["dollar_sharpe"], "#c0392b"),
]:
    fig.add_trace(go.Box(x=values.dropna(), name=name, marker_color=color, boxpoints=False, orientation="h"))
fig.add_vline(x=real["sharpe"], line_color="black", line_width=3)
fig.add_annotation(x=real["sharpe"], y=1, yref="paper", text=f"Real in-sample: {real['sharpe']:.2f}", showarrow=False, xanchor="left", yanchor="bottom")
fig.add_trace(
    go.Scatter(
        x=[real_dollar],
        y=["OU, dollar P&L (relationship real)"],
        mode="markers",
        marker={"symbol": "diamond", "size": 12, "color": "black"},
    )
)
finish_figure(
    fig,
    "Sharpe ratio of the Phase 2 strategy across simulated worlds (box: middle 50% of paths)",
    "Annualised Sharpe ratio on the path",
    "",
    "Source: Yahoo Finance, in-sample 2000-2018; 1,000 paths per row. "
    f"OU row in dollar P&L; diamond = real value ({real_dollar:.2f}).",
)
fig.update_layout(showlegend=False)
save_figure(fig, "05_sharpe_by_world", figures_dir)
fig.show()"""
    ),
    md(
        """Where the long-run relationship is real by construction, the strategy does **no better** than where
it does not exist: median 0.17 against 0.16. The real history (0.37 in this dollar measure) sits at the 80th
and 83rd percentile of the two. There are two reasons. First, with a half-life of 192 days the spring is
weak: in 60 days, the signal's window, the OU undoes little of a move (variance ratio 0.90), so a 60-day
z-score barely sees it. Second, the real spread pulls back faster over weeks to months than the estimated OU
says (variance ratio 0.61 at 60 days). One spring fitted on daily changes misses the short-term behaviour the
strategy trades.

So the OU misses the speed of the pull-back at the signal's horizon, which confirms it should not be the main
method. And the strategy's Sharpe barely separates a world with the relationship from one without it, so it
is a weak measuring instrument. My evidence on the relationship itself is the variance ratio of section 4."""
    ),
    md(
        """## 7. What I conclude, and what I am unsure about

**Answer to the brief**, in three statements:
1. The relationship. In 2000–2018 the spread does pull back over weeks to months (60-day variance ratio 0.61,
   reached by 0.5% of paths without dynamics). It is not a fit to one accidental path, but not a pull to a
   fixed level either (Phase 2). The test behind this, and the one-day blocks it uses, were added after seeing
   the first results: strong evidence on one history, but not a pre-registered finding.
2. The signal. On histories rebuilt from the same days it is usually positive (74% of paths), but its typical
   Sharpe is about 0.17, half the 0.35 of Phase 2. The 0.35 is a good draw (80th percentile, 67th once the
   grid search is repeated). In 19 years the profit is too small to tell apart from luck.
3. A new regime. No path contains a break like 2024–2026, because every path is made of 2000–2018 days. This
   is the risk that actually happened out-of-sample.

Overall, the pull-back is real, but the strategy built on it is a small edge that history showed at its best,
and nothing in-sample could have protected it from the break. This agrees with Phase 2 (in-sample t = 1.49,
out-of-sample loss) and Phase 3 (unstable hedge ratio).

**What I am unsure about**
- The bootstrap treats blocks as interchangeable, so a 2008 block can follow a 2017 block. Regimes (the
  2000–2005 slide of the ratio, the 2008 crisis) get mixed, which probably widens the distribution.
- The hedge ratio is estimated on the same path it trades, as in the Phase 2 in-sample backtest. The
  exploratory 1-day check shows this adds little here, but it is an optimistic convention.
- The OU uses one constant pull and normal shocks, and section 6 shows it misses the short-term behaviour.
- I do not place the out-of-sample Sharpe (−0.53) in these distributions as a test: it covers 7.8 years, not
  19, with a hedge ratio frozen from earlier data, so its natural spread is different.
- The blocks start at random rows and are not wrapped around, so the last days of 2018 are drawn slightly
  less often.
- The simulation is read from its checkpoint, so the automated tests and the CI re-run this notebook without
  the 4-minute simulation. The variance-ratio paths of section 4 are recomputed every time.

## Appendix: trial log and checks

I add eight rows to the trial log. Five are configurations fixed in advance (three block lengths, the
selection analysis, the OU check). Three are exploratory, decided after seeing the results: the 1-day
blocks, and the 60-day variance ratio of the real spread against the 1-day and the 10-day paths."""
    ),
    code(
        """period = (prices.index[START].date(), AS_OF)
base = {"entry": S.entry, "window": S.window, "exit": S.exit, "n_paths": N_PATHS, "seed": SEED}
note_main = "Joint block bootstrap of in-sample daily returns; beta re-estimated on each path."
for b in BLOCKS:
    log_trial(trials_db, "phase5-simulation", f"{A}/{B}", "in-sample", period,
              {**base, "method": "joint block bootstrap", "block": b, "rules": "frozen"},
              summary[f"Bootstrap, {b}-day blocks"], note_main)
log_trial(trials_db, "phase5-simulation", f"{A}/{B}", "in-sample", period,
          {**base, "method": "joint block bootstrap", "block": MAIN_BLOCK, "rules": "grid + stable-zone rule"},
          selection_stats, note_main + " Sharpe of the cell picked by the Phase 2 rule on each path.")
log_trial(trials_db, "phase5-simulation", f"{A}/{B}", "in-sample", period,
          {**base, "method": "Ornstein-Uhlenbeck spread", "beta": S.beta, "rules": "frozen"},
          worlds.loc["OU (long-run tie, as estimated)"].to_dict(),
          "Declared extra check; Sharpe of the daily dollar P&L of one unit of the spread.")
log_trial(trials_db, "phase5-exploratory", f"{A}/{B}", "in-sample", period,
          {**base, "method": "joint block bootstrap", "block": EXPLORATORY_BLOCK, "rules": "frozen"},
          summary[f"Bootstrap, {EXPLORATORY_BLOCK}-day blocks (exploratory)"],
          "Decided after seeing the results: separates short-term dynamics from the in-path beta fit.")
for b in vr_paths:
    log_trial(trials_db, "phase5-exploratory", f"{A}/{B}", "in-sample", period,
              {"method": "variance ratio of the spread", "horizon": 60, "null": f"joint bootstrap, {b}-day blocks",
               "n_paths": N_PATHS, "seed": SEED},
              {"real_variance_ratio": float(variance_ratio[60]), "paths_median": float(vr_paths[b].median()),
               "share_paths_at_or_below_real": float((vr_paths[b] <= variance_ratio[60]).mean())},
              "Decided after seeing the results: is the real spread's 60-day pull-back more than chance?")
print("Trials in the log:", count_trials(trials_db))"""
    ),
    code(
        """# Every number quoted in the text, checked
def r3(x: float) -> float:
    return round(float(x), 3)


assert round(real["sharpe"], 2) == 0.35 and round(real["beta"], 3) == 0.388 and real["n_trades"] == 94
assert len(prices) == 4779 and prices.index[0] == pd.Timestamp("2000-01-03")
assert r3(main_stats["median"]) == 0.165 and r3(main_stats["p05"]) == -0.230 and r3(main_stats["p95"]) == 0.526
assert round(main_stats["share_positive"], 2) == 0.74
perf = performance.round(3)
assert perf.loc["Annual return", "paths: median"] == 0.017 and round(real["ann_return"], 3) == 0.024
assert round((main["t_stat"] >= 2).mean(), 2) == 0.08
assert round(real["max_drawdown"], 2) == -0.14 and perf.loc["Maximum drawdown", "paths: median"] == -0.32
assert perf.loc["Maximum drawdown", "paths: 5%"] <= -0.595
assert perf.loc["Maximum drawdown", "share of paths worse than real"] >= 0.985
assert perf.loc["Annual volatility", "share of paths worse than real"] == 1.0
assert round(near_beta.median(), 2) == -0.31 and len(near_beta) > 150
assert round((main["beta"] < 0).mean(), 2) == 0.06
assert round(vr_null.loc["median", "1-day blocks"], 2) == 0.96
assert round(vr_null.loc["share of paths <= real", "1-day blocks"], 3) == 0.005
assert round(vr_null.loc["median", "10-day blocks"], 2) == 0.82
assert round(vr_null.loc["share of paths <= real", "10-day blocks"], 2) == 0.06
assert round(1 - main_stats["share_positive"], 2) == 0.26 and round(main_stats["real_percentile"], 2) == 0.80
c = checks.round(2)
assert c.iloc[0, 0] == 0.55 and c.iloc[0, 2] >= 0.50 and c.iloc[0, 3] <= 0.60
assert c.iloc[1, 2] == -0.02 and c.iloc[1, 3] == 1.38 and c.iloc[2, 1] == 98
ratio = prices[A] / prices[B]
assert round(ratio.iloc[0], 2) == 0.73 and round(ratio.iloc[-1], 2) == 0.43
assert round(ratio.min(), 2) == 0.33 and ratio.idxmin().year == 2005
t = blocks_table.round(3)
assert [t.iloc[i]["median"] for i in range(4)] == [0.017, 0.1, 0.165, 0.268]
assert [round(x, 2) for x in blocks_table["real_percentile"]] == [0.92, 0.87, 0.80, 0.64]
assert [round(variance_ratio[h], 2) for h in (5, 20, 60)] == [0.92, 0.76, 0.61]
assert r3(selection_stats["median"]) == 0.248 and round(selection_stats["share_positive"], 2) == 0.86
assert round(selection_stats["real_percentile"], 2) == 0.67 and selection_stats["paths"] == N_PATHS
assert main["selected_entry"].value_counts(normalize=True).idxmax() == 1.5
assert main["selected_window"].value_counts(normalize=True).idxmax() == 60
assert round(picked.loc[1.5].iloc[0], 2) == 0.48 and round(picked.loc[60].iloc[1], 2) == 0.57
w = worlds.round(3)
assert round(ou.half_life) == 192 and round(ou_variance_ratio_60, 2) == 0.90
assert round(w.iloc[0]["median"], 2) == 0.16 and round(w.iloc[1]["median"], 2) == 0.17
assert round(real_dollar, 2) == 0.37
assert [round(x, 2) for x in worlds["real_percentile"]] == [0.83, 0.80]
assert count_trials(trials_db, "phase5-simulation") == 5
assert count_trials(trials_db, "phase5-exploratory") == 3
for name in ("05_example_paths", "05_bootstrap_sharpe", "05_sharpe_by_world"):
    assert (figures_dir / f"{name}.png").exists()"""
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (.venv)", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, sys.argv[1])

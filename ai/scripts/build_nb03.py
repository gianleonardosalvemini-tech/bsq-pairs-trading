"""Build notebooks/03_stability.ipynb: Phase 3, is the KO / PEP relationship stable?

Usage: python build_nb03.py <notebook path>
"""

import sys

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(
        """# Phase 3 — Is the KO / PEP relationship stable?

The hedge ratio β of Phase 2 is only an estimate, and the brief warns that the relationship "can drift or
break down entirely over time". I check it one way (simple: rolling correlation, medium: rolling hedge
ratio and/or rolling ADF), flag the weak periods, and ask: does the Phase 2 signal do worse in them?

**Result**
- The hedge ratio is far from stable. On 2-year windows it goes from −0.07 to 0.75 inside 2000–2018 and
  falls to −0.23 in 2025, against the frozen 0.388.
- Flags from a rolling stationarity test do not separate good trades from bad. The label fixed in advance
  calls almost every day "weak". A relative label gives a small, not significant difference whose sign
  changes with the window.
- **Answer to the brief: no evidence that the signal does worse in the flagged periods.** The label does
  catch the 2025 break, but the strategy lost just as much in "normal" periods.

The ADF test (Augmented Dickey-Fuller) checks stationarity, i.e. whether a series keeps coming back to a
fixed average. Its p-value is low when the data say yes, high when they cannot tell the series from a
drift. Other terms are in the README glossary. Everything here about 2019–2026 is analysis after the
single out-of-sample run of Phase 2, and nothing in this notebook changes the strategy."""
    ),
    md(
        """## 1. Set-up and data

Same prices as Phase 2, now 2000–2026. The two backtests are the Phase 2 runs of the frozen strategy
(threshold 1.5, window 60, exit 0, β 0.388372), re-run unchanged. The cell checks they give the Phase 2
Sharpe ratios (0.35 and −0.53)."""
    ),
    code(
        """import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from pairs.backtest import run_zscore_strategy, summarize
from pairs.config import PAIR, PHASE2_STRATEGY, Settings
from pairs.data import load_pair_prices
from pairs.explore import select_until
from pairs.logging_setup import configure_logging
from pairs.plots import finish_figure, save_figure
from pairs.spread import build_spread, fit_hedge_ratio
from pairs.stability import (
    compare_groups,
    pvalue_cutoff,
    rolling_adf_pvalue,
    rolling_hedge_ratio,
    trade_returns,
    weak_labels,
)
from pairs.trials import count_trials, log_trial

configure_logging(logging.WARNING)  # notebook output: tables and charts only, no progress logs
pio.renderers.default = "plotly_mimetype+png"
settings = Settings.from_env()
figures_dir = settings.results_dir / "figures"
trials_db = settings.results_dir / "trials.sqlite"
SOURCE = (
    "Source: Yahoo Finance adjusted close, 3 Jan 2000 - 9 Oct 2026."
    " Shaded: out-of-sample (after the single run)."
)
A, B, PAIR_NAME, S = PAIR.ticker_a, PAIR.ticker_b, f"{PAIR.ticker_a}/{PAIR.ticker_b}", PHASE2_STRATEGY


prices = load_pair_prices(PAIR, settings.data_dir, PAIR.end)
is_prices = select_until(prices, PAIR.in_sample_end)
assert abs(fit_hedge_ratio(prices, A, B, PAIR.in_sample_end).beta - S.beta) < 1e-6  # the frozen β
is_start = is_prices.index[249]  # same first day as Phase 2 (first full 250-day window of the grid)
oos_start = prices.index[prices.index > pd.Timestamp(PAIR.in_sample_end)][0]
runs = {
    "in-sample": run_zscore_strategy(is_prices, A, B, S.beta, S.entry, S.window, is_start, S.exit).loc[is_start:],
    "out-of-sample": run_zscore_strategy(prices, A, B, S.beta, S.entry, S.window, oos_start, S.exit).loc[oos_start:],
}
stats = {name: summarize(run["ret"], run["position"]) for name, run in runs.items()}
pd.DataFrame(stats).loc[["sharpe", "t_stat", "ann_return", "n_trades"]].round(3)"""
    ),
    md(
        """## 2. Why check, and which method

In Phase 2 the spread check was negative. The spread moves in waves of several years, a deviation takes
about 192 trading days to halve, and Engle-Granger does not reject a drifting spread (p = 0.12). The
strategy still earned a little in-sample because its z-score uses the 60-day rolling average of the
spread, which follows the slow waves, so it bets on short swings around the recent level. Out-of-sample
it lost, while the KO/PEP price ratio went from 0.35 to 0.56. Phase 3 asks if those losses come from
periods where the relationship itself was weak.

I use the medium option (rolling hedge ratio + rolling ADF test). A rolling correlation of daily returns
cannot see my kind of failure: two stocks can move together day by day (Phase 1: correlation 0.51 on
average, highest after 2007) while the ratio of their price levels slides for years, and the strategy
bets on levels. I chose the method already knowing the cause of the out-of-sample loss (Phase 2, section
7), so it follows the mechanism and was not a blind pick.

The window is 500 trading days (about 2 years) for both. With a half-life of 192 days, in one year the
spread returns to its average at most once, too little for any test. 500 days are about 2.6 half-lives.
Section 7 shows 250 and 750 days as sensitivity."""
    ),
    md(
        """## 3. The rolling hedge ratio

Every day, an OLS of KO on PEP on the last 500 trading days (today included). It is only descriptive and
never used to trade.

With a stable relationship the line would stay near the dashed β = 0.388 frozen in Phase 2. Where it is
far from it, a different mix of the two stocks would have made the spread flat in that period."""
    ),
    code(
        """BETA_WINDOW = 500
rolling_beta = rolling_hedge_ratio(prices, A, B, BETA_WINDOW)
beta_is, beta_oos = rolling_beta.loc[: pd.Timestamp(PAIR.in_sample_end)].dropna(), rolling_beta.loc[oos_start:]

fig = go.Figure()
fig.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta, name="β on the last 500 days", line={"color": "#1f4e9c"}))
fig.add_hline(y=S.beta, line_dash="dash", line_color="black", annotation_text="frozen β = 0.388")
fig.add_hline(y=0, line_color="grey", line_width=1)
fig.add_vrect(x0=oos_start, x1=prices.index[-1], fillcolor="#f39c12", opacity=0.12, line_width=0)
finish_figure(fig, "Hedge ratio of KO on PEP re-estimated on rolling 2-year windows", "Date",
              "β (dollars of KO per dollar of PEP)", SOURCE)
save_figure(fig, "03_rolling_beta", figures_dir)
fig.show()
pd.Series({
    "in-sample minimum": f"{beta_is.min():.3f} ({beta_is.idxmin().date()})",
    "in-sample maximum": f"{beta_is.max():.3f} ({beta_is.idxmax().date()})",
    "in-sample days within ±0.1 of 0.388": f"{(beta_is - S.beta).abs().lt(0.1).mean():.1%}",
    "out-of-sample minimum": f"{beta_oos.min():.3f} ({beta_oos.idxmin().date()})",
    "last value (9 Oct 2026)": f"{rolling_beta.iloc[-1]:.3f}",
}).to_frame("rolling β, 500 days")"""
    ),
    md(
        """The relationship is **not stable at a 2-year horizon, even in-sample**. β goes from −0.07 (May 2002) to
0.75 (May 2012), and only 30% of in-sample days are within ±0.1 of the frozen 0.388. So the frozen value
is an average of very different periods rather than a constant of the pair. Out-of-sample β falls to
−0.23 (August 2025): over those two years KO and PEP moved in opposite directions, so no positive β could
have made the spread flat. This is the break that sank the out-of-sample run.

So β drifts, as the brief feared. But are the trades worse when the relationship is weak? For that I need
a daily label."""
    ),
    md(
        """## 4. The rolling stationarity test, and the label fixed in advance

Every week (every 5 trading days, the days in between keep the last value) I run an ADF test on the last
500 days of the spread `KO − 0.388 × PEP`, the one the strategy actually trades.

**I fixed the label before computing it**: a day is "weak" if the ADF p-value is above 0.10, i.e. on the
last two years the test cannot say that the spread comes back to its average. The label of day t uses
data up to t−1 only, so it was known before that day's trading decision.

A low p-value means the last two years look mean-reverting, a high one that they look like a drift. Below
the red line (0.10) the absolute rule calls the period normal."""
    ),
    code(
        """ADF_WINDOW = 500
pvalues = {w: rolling_adf_pvalue(build_spread(prices, A, B, S.beta), w) for w in (250, 500, 750)}
pv = pvalues[ADF_WINDOW]
cutoffs = {w: pvalue_cutoff(p, PAIR.in_sample_end, share=0.25) for w, p in pvalues.items()}

fig = go.Figure()
fig.add_trace(go.Scatter(x=pv.index, y=pv, name="ADF p-value, last 500 days", line={"color": "#1f4e9c"}))
fig.add_hline(y=0.10, line_dash="dash", line_color="#c0392b", annotation_text="0.10: absolute rule (fixed in advance)")
fig.add_hline(y=cutoffs[ADF_WINDOW], line_dash="dot", line_color="black",
              annotation_text=f"{cutoffs[ADF_WINDOW]:.2f}: relative rule (top 25% of in-sample, section 5)")
fig.add_vrect(x0=oos_start, x1=prices.index[-1], fillcolor="#f39c12", opacity=0.12, line_width=0)
finish_figure(fig, "Is the spread KO - 0.388 x PEP stationary on the last two years? (ADF test)", "Date",
              "p-value (low = mean-reverting)", SOURCE)
save_figure(fig, "03_adf_pvalue", figures_dir)
fig.show()
pv_is, pv_oos = pv.loc[: pd.Timestamp(PAIR.in_sample_end)].dropna(), pv.loc[oos_start:]
pd.DataFrame({
    "in-sample": {"median p-value": round(pv_is.median(), 3), "share of days with p < 0.10": round(pv_is.lt(0.10).mean(), 3)},
    "out-of-sample": {"median p-value": round(pv_oos.median(), 3), "share of days with p < 0.10": round(pv_oos.lt(0.10).mean(), 3)},
})"""
    ),
    md(
        """**The label fixed in advance does not work.** On 2-year windows the test calls the spread stationary on
only 10% of in-sample days and 5% of out-of-sample days. So the absolute rule flags about 90% of the
in-sample as "weak", including the years in which the strategy made money (numbers in section 6). A label
that is on almost every day cannot separate anything.

This is the Engle-Granger result of Phase 2 seen window by window, not a new error: the spread does not
come back to a fixed level. Either it really drifts, or it reverts too slowly (half-life 192 days) for two
years of data to show it, and the test cannot tell which. For Phase 2 this means the strategy never
traded a "stationary" spread. It bets on swings around a moving 60-day average, a weaker property that the ADF
test does not measure."""
    ),
    md(
        """## 5. One relative label, declared after the first one failed

The absolute rule failed for a mechanical reason (the test almost never says "stationary" on this pair),
not because of the returns. So I replace it with one relative rule, and only this one.

A day is "weak" if the p-value is in the highest 25% of its in-sample values (above the 75th percentile
of the 2002–2018 p-values, frozen and reused unchanged on 2019–2026). So I ask "is the spread less
mean-reverting than usual?" instead of "is it mean-reverting?", like the simple method of the brief
("drops well below its historical norm"). By construction it flags about a quarter of the in-sample days.

- **This rule is post-hoc.** I decided it after seeing the numbers of the absolute rule (including how its
  trades split) and before computing its own. I chose 25% to get two groups of usable size, not from any
  return.
- I wrote the prediction before computing it: if a weak relationship explains the losses, trades opened
  in the weak quartile earn less than the others, both in-sample and out-of-sample.
- Nothing is tuned: one share (25%), the same three windows, and every comparison goes into the trial log
  (section 9).
- The cutoff uses all in-sample p-values, so 2005 is labelled with a threshold that "knows" 2018, an
  in-sample calibration like the β itself."""
    ),
    md(
        """## 6. The brief's question: does the signal perform worse in weak periods?

The main comparison is per trade. Each trade of the frozen strategy gets the label of its entry day (known
the day before), and I compare the average trade return of the two groups. The question becomes: could
you have known, when opening the trade, that it was a bad time? The 4 in-sample trades of 2001 have no
label (the first 500-day window ends on 31 Dec 2001), so I leave them out."""
    ),
    code(
        """def compare_trades(run: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    trades = trade_returns(run)
    entry_label = labels.reindex(trades["entry"]).set_axis(trades.index)
    out = compare_groups(trades["ret"], entry_label)
    out["share_weak_days"] = float(labels.reindex(run.index).mean())
    return out


def compare_days(run: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    held = run["position"].shift(1).fillna(0.0).ne(0)  # a day's return comes from yesterday's position
    return compare_groups(run["ret"][held], labels.shift(1).reindex(run.index)[held])


rules = {
    "absolute (p > 0.10)": {w: weak_labels(p, 0.10) for w, p in pvalues.items()},
    "relative (top 25% in-sample)": {w: weak_labels(p, cutoffs[w]) for w, p in pvalues.items()},
}
trade_cmp = {(rule, w, name): compare_trades(run, labels[w])
             for rule, labels in rules.items() for w in pvalues for name, run in runs.items()}


def trade_table(window: int) -> pd.DataFrame:
    rows = {}
    for (rule, w, name), c in trade_cmp.items():
        if w == window:
            rows[(rule, name)] = {
                "weak days": f"{c['share_weak_days']:.0%}",
                "trades weak / normal": f"{c['n_weak']:.0f} / {c['n_normal']:.0f}",
                "avg trade weak": f"{c['mean_weak']:+.2%}", "avg trade normal": f"{c['mean_normal']:+.2%}",
                "winners weak / normal": f"{c['win_weak']:.0%} / {c['win_normal']:.0%}",
                "Welch t (weak - normal)": round(c["welch_t"], 2),
            }
    return pd.DataFrame(rows).T


trade_table(ADF_WINDOW)"""
    ),
    md(
        """"avg trade" is the average return of one trade from opening to closing. The hypothesis of the brief
predicts weak < normal, i.e. a negative Welch t.

- Absolute rule: 81 in-sample trades are "weak" and only 9 "normal" (out-of-sample 33 and 3). With 3 or 9
  trades in a group no comparison means anything. The sign is opposite to the hypothesis (weak trades
  earn more) and not significant.
- Relative rule, in-sample: the prediction goes the right way. Trades opened in the weak quartile earn on
  average +0.05% against +0.57% for the others, with 70% winners against 80%. But the difference is about
  one standard error (t = −1.07), so it can be luck.
- Relative rule, out-of-sample (analysis after the single run): no difference, −0.68% against −0.85%
  (t = 0.12). **On average the out-of-sample losses are spread over normal and weak periods alike.** The
  chart below shows where they fall year by year."""
    ),
    code(
        """# Frozen strategy (in-sample, then out-of-sample); weak days of the relative rule shaded
weak_rel = rules["relative (top 25% in-sample)"][ADF_WINDOW]
equity = (1 + pd.concat([runs["in-sample"]["ret"], runs["out-of-sample"]["ret"]])).cumprod()
flag = weak_rel.reindex(equity.index).eq(1)
blocks = (flag != flag.shift()).cumsum()[flag]
fig = go.Figure()
for _, days in blocks.groupby(blocks):
    fig.add_vrect(x0=days.index[0], x1=days.index[-1], fillcolor="#c0392b", opacity=0.15, line_width=0)
fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Value of 1 dollar (frozen strategy)", line={"color": "#1f4e9c"}))
fig.add_vline(x=oos_start, line_dash="dash", line_color="black", annotation_text="out-of-sample starts")
finish_figure(fig, "Frozen Phase 2 strategy; red bands = weak relationship (relative rule, 500-day ADF)", "Date",
              "Value of 1 dollar", "Source: Yahoo Finance adjusted close, 27 Dec 2000 - 9 Oct 2026. Before costs. "
              "Out-of-sample = analysis after the single run.")
save_figure(fig, "03_equity_weak", figures_dir)
fig.show()

# Where the out-of-sample loss falls (log returns add up: weak part + normal part = total)
oos_ret, oos_lab = runs["out-of-sample"]["ret"], weak_rel.reindex(runs["out-of-sample"].index)
log_ret = np.log1p(oos_ret)
split = pd.DataFrame({
    period: {"weak days (share)": f"{oos_lab.loc[first:last].mean():.0%}",
             "log return on weak days": round(log_ret.loc[first:last][oos_lab.loc[first:last].eq(1)].sum(), 3),
             "log return on normal days": round(log_ret.loc[first:last][oos_lab.loc[first:last].eq(0)].sum(), 3)}
    for period, (first, last) in {"2019-2024": ("2019", "2024"), "2025-2026": ("2025", "2026"),
                                  "whole out-of-sample": ("2019", "2026")}.items()
})
split"""
    ),
    md(
        """If weak periods explained the losses, the line would fall inside the red bands and rise outside them.
In-sample it rises and falls both inside and outside.

**Out-of-sample (analysis after the single run) the picture is mixed.** The table splits the
out-of-sample log return by the label of each day. In 2025–2026 the label works: it switches on in the
third quarter of 2024, covers 86% of the days, and most of the big fall is inside the red bands (−19%,
against −6% on the few normal days). In 2019–2024 it is the opposite: the strategy loses on "normal" days
(−19%) and gains on weak ones (+13%). Over the whole out-of-sample, more is lost on normal days (−25%)
than on weak days (−7%).

So label and losses coincide in one episode, and not in the other six years or in-sample. That is why
the trade averages show no difference. I saw this only after the out-of-sample run, so it is an
observation, not a validated rule."""
    ),
    md(
        """## 7. Sensitivity: does the answer depend on the 2-year window?

Same two rules, ADF windows of 250 and 750 days. The relative cutoff is recomputed on the in-sample
p-values of each window, always the top 25%."""
    ),
    code(
        """sens = {}
for (rule, w, name), c in trade_cmp.items():
    sens[(rule, name, w)] = round(c["welch_t"], 2)
sens_table = pd.Series(sens).unstack().rename(columns=lambda w: f"Welch t, {w}-day ADF")
sens_table"""
    ),
    md(
        """**The answer changes with the window.** With the relative rule the in-sample t is +2.50 at 250 days (weak
trades earn more, against the hypothesis), −1.07 at 500 and −0.50 at 750. Out-of-sample it is +0.66,
+0.12 and −0.99. A sign that flips with the window looks like noise, by the standard of Phase 2: "if a
tiny change collapses the result, that is a sign you found noise". The two values beyond ±2 (2.11 and
2.50, both at 250 days, in-sample) go against the hypothesis, and among 12 comparisons one or two that
large are expected by chance alone."""
    ),
    md(
        """## 8. What I learn, and what I am unsure about

**Answer to the brief: no, there is no evidence that the Phase 2 signal performs worse in the periods
flagged as weak.** The prediction I wrote before computing the relative label is not confirmed. The
in-sample difference goes the predicted way but is within luck, the out-of-sample one is zero, and the
sign flips with the window.

On stability, the relationship drifts a lot: the 2-year β moves between −0.07 and 0.75 in-sample and
reaches −0.23 in 2025, so the frozen 0.388 is an average of very different periods. With a frozen β the
spread is almost never stationary on 2-year windows (10% of in-sample days pass the 0.10 test),
consistent with the Engle-Granger p = 0.12 of Phase 2.

Why the label does not explain the losses on average (my reading, not proven):
1. It measures a return to a fixed level, while the strategy trades swings around a moving 60-day
   average. The strategy also loses when the spread is "normal" by this measure (2019–2024).
2. It has little power on a spread with a 9-month half-life, so "weak" and "normal" differ only a little.
3. It caught one big break (2025, label on from late 2024), but one episode seen after the out-of-sample
   run is not evidence. In-sample the same label did not separate good from bad trades.

What I am unsure about:
- Few trades (90 in-sample, 36 out-of-sample), so a real but modest effect could be invisible.
- The ADF p-values ignore that β was fitted in-sample, so in-sample windows look somewhat more stationary
  than they are. The relative rule partly absorbs this because it compares the test with itself.
- The β drift in section 3 lines up with the 2025 losses, but a label built on it now would be designed
  after seeing them, so I do not test it here. Whether following the drift helps is the question of the
  Kalman extension, with its own rules fixed in advance."""
    ),
    md(
        """## 9. How many things I tried

Phase 3 runs no new backtest, it only splits the returns of the frozen strategy. I still log every
comparison, because each one is a chance to find something by luck: 2 rules × 3 windows × 2 periods = **12**."""
    ),
    code(
        """for (rule, w, name), c in trade_cmp.items():
    run = runs[name]
    log_trial(trials_db, "phase3-stability", PAIR_NAME, name, (run.index[0].date(), run.index[-1].date()),
              {"rule": rule, "adf_window": w, "cutoff": round(cutoffs[w] if rule.startswith("relative") else 0.10, 4),
               "entry": S.entry, "window": S.window, "exit": S.exit, "beta": S.beta},
              c, "trade returns of the frozen strategy split by weak-relationship label at entry")
pd.Series({"phase 2 (backtests)": count_trials(trials_db, "phase2-grid") + count_trials(trials_db, "phase2-zoom")
           + count_trials(trials_db, "phase2-oos"),
           "phase 3 (comparisons)": count_trials(trials_db, "phase3-stability"),
           "total": count_trials(trials_db)}).to_frame("logged")"""
    ),
    md(
        """## Appendix: a second view, per day

The average return of the days spent in a position, grouped by the label of the day before. This is
partly circular: when the spread runs away the test drifts toward "weak" and the strategy loses, so the
two move together by construction. Also, daily returns inside one trade are correlated, so the t
overstates the evidence."""
    ),
    code(
        """day_rows = {}
for rule, labels in rules.items():
    for name, run in runs.items():
        c = compare_days(run, labels[ADF_WINDOW])
        day_rows[(rule, name)] = {
            "days weak / normal": f"{c['n_weak']:.0f} / {c['n_normal']:.0f}",
            "yearly return weak": f"{252 * c['mean_weak']:+.1%}", "yearly return normal": f"{252 * c['mean_normal']:+.1%}",
            "Welch t": round(c["welch_t"], 2),
        }
day_cmp = {key: compare_days(runs[key[1]], rules[key[0]][ADF_WINDOW]) for key in day_rows}
pd.DataFrame(day_rows).T"""
    ),
    md(
        """The days give the same picture as the trades. With the relative rule the days in a position earn +3.3% a
year (weak) against +3.9% (normal) in-sample, and −6.2% against −4.3% out-of-sample. Both t-statistics
are far from 2."""
    ),
    code(
        """# Every number quoted in the text, checked (expected values recorded on 10 Oct 2026)
assert abs(stats["in-sample"]["sharpe"] - 0.35) < 0.005 and abs(stats["out-of-sample"]["sharpe"] + 0.53) < 0.005
assert stats["in-sample"]["n_trades"] == 94 and stats["out-of-sample"]["n_trades"] == 36
assert is_start == pd.Timestamp("2000-12-27") and oos_start == pd.Timestamp("2019-01-02")
assert abs(beta_is.min() + 0.067) < 0.001 and beta_is.idxmin() == pd.Timestamp("2002-05-08")
assert abs(beta_is.max() - 0.745) < 0.001 and beta_is.idxmax() == pd.Timestamp("2012-05-14")
assert abs((beta_is - S.beta).abs().lt(0.1).mean() - 0.300) < 0.001
assert abs(beta_oos.min() + 0.234) < 0.001 and beta_oos.idxmin() == pd.Timestamp("2025-08-14")
assert abs(rolling_beta.iloc[-1] - 0.006) < 0.001
assert pv.first_valid_index() == pd.Timestamp("2001-12-31")
assert abs(pv_is.lt(0.10).mean() - 0.104) < 0.001 and abs(pv_oos.lt(0.10).mean() - 0.046) < 0.001
assert abs(cutoffs[500] - 0.620) < 0.001
c = trade_cmp[("absolute (p > 0.10)", 500, "in-sample")]
assert (c["n_weak"], c["n_normal"]) == (81, 9) and abs(c["share_weak_days"] - 0.896) < 0.001
assert c["mean_weak"] > c["mean_normal"]
c = trade_cmp[("absolute (p > 0.10)", 500, "out-of-sample")]
assert (c["n_weak"], c["n_normal"]) == (33, 3) and abs(c["share_weak_days"] - 0.954) < 0.001
c = trade_cmp[("relative (top 25% in-sample)", 500, "in-sample")]
assert (c["n_weak"], c["n_normal"]) == (20, 70) and abs(c["share_weak_days"] - 0.25) < 0.001
assert abs(c["mean_weak"] - 0.0005) < 0.00005 and abs(c["mean_normal"] - 0.0057) < 0.00005
assert abs(c["win_weak"] - 0.70) < 0.001 and abs(c["win_normal"] - 0.80) < 0.001 and abs(c["welch_t"] + 1.07) < 0.005
c = trade_cmp[("relative (top 25% in-sample)", 500, "out-of-sample")]
assert (c["n_weak"], c["n_normal"]) == (11, 25)
assert abs(c["mean_weak"] + 0.0068) < 0.00005 and abs(c["mean_normal"] + 0.0085) < 0.00005 and abs(c["welch_t"] - 0.12) < 0.005
d = day_cmp[("relative (top 25% in-sample)", "in-sample")]
assert abs(252 * d["mean_weak"] - 0.033) < 0.0005 and abs(252 * d["mean_normal"] - 0.039) < 0.0005
d = day_cmp[("relative (top 25% in-sample)", "out-of-sample")]
assert abs(252 * d["mean_weak"] + 0.062) < 0.0005 and abs(252 * d["mean_normal"] + 0.043) < 0.0005
t = {(r[:3], n, w): v for (r, n, w), v in sens.items()}
assert (t[("rel", "in-sample", 250)], t[("rel", "in-sample", 500)], t[("rel", "in-sample", 750)]) == (2.50, -1.07, -0.50)
assert (t[("rel", "out-of-sample", 250)], t[("rel", "out-of-sample", 500)], t[("rel", "out-of-sample", 750)]) == (0.66, 0.12, -0.99)
assert t[("abs", "in-sample", 250)] == 2.11 and sum(abs(v) >= 2 for v in sens.values()) == 2
assert count_trials(trials_db, "phase3-stability") == 12
assert abs(split.loc["log return on weak days", "2025-2026"] + 0.193) < 0.001
assert abs(split.loc["log return on normal days", "2025-2026"] + 0.064) < 0.001
assert abs(split.loc["log return on normal days", "2019-2024"] + 0.188) < 0.002
assert abs(split.loc["log return on weak days", "whole out-of-sample"] + 0.066) < 0.001
assert abs(split.loc["log return on normal days", "whole out-of-sample"] + 0.252) < 0.001
assert abs(oos_lab.loc["2025":"2026"].mean() - 0.865) < 0.001
q = oos_lab.loc["2024"].resample("QE").mean()
assert abs(split.loc["log return on weak days", "2019-2024"] - 0.127) < 0.001
assert q.iloc[1] == 0 and q.iloc[2] > 0.5  # off in Q2 2024, on from Q3 2024
for name in ("03_rolling_beta", "03_adf_pvalue", "03_equity_weak"):
    assert (figures_dir / f"{name}.png").exists(), name"""
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (.venv)", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, sys.argv[1])

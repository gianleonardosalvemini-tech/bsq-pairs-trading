"""Build notebooks/02_signal.ipynb: Phase 2, the spread, the z-score signal and its validation.

Usage: python build_nb02.py <notebook path>
"""

import sys

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(
        """# Phase 2 — A mean-reversion signal on the KO / PEP spread, and its validation

**What the brief asks.** Build the *spread* between the two prices with an OLS regression, check that
it really comes back to its average, turn it into a *z-score* signal (long the spread when the z-score
is very negative, short when very positive, close when it reverts toward zero) and, above all,
**validate it honestly**: tune on an in-sample period, test once on an out-of-sample period, say how
many things were tried, and check how sensitive the result is to small changes of the parameters.
The brief calls this "the part we weight most heavily".

**The result in two lines.** On the in-sample period (2001–2018) the signal earns a little in all 25
variants tried, but the effect is weak and not statistically significant. On the out-of-sample period
(2019–2026), run once with everything frozen, it **loses money**: the relationship drifted.

**Words used below**

| Term | Meaning |
|---|---|
| Spread | `KO − β × PEP`: a combination of the two prices that, if the stocks are truly related, should hover around a stable average |
| Hedge ratio β | How many dollars KO tends to move for each dollar PepsiCo moves; estimated with an OLS regression |
| OLS regression | The straight line `KO = α + β × PEP` that fits the two prices best (least squares) |
| z-score | How many standard deviations the spread is away from its own rolling average today |
| Rolling average / window | The average of the last N trading days only (N = the *window*), recomputed every day |
| Long the spread | Buy 1 KO share and sell β PEP shares: it gains if the spread goes back up |
| Short the spread | The opposite: sell 1 KO share and buy β PEP shares |
| Sharpe ratio | Average yearly return divided by its yearly volatility: return per unit of risk (0.5 is decent, 1 is good) |
| t-statistic | How many standard errors the average daily return is away from zero; below about 2 it can be luck |
| Drawdown | The loss from the highest value reached so far; the maximum drawdown is the worst one |
| In-sample / out-of-sample | 2000–2018, where we look and tune / 2019–2026, untouched until the strategy is frozen, then run once |
| Look-ahead | Using, on a given day, information that was not yet known on that day: it makes any backtest look better than reality |"""
    ),
    md(
        """## 1. The brief's Phase 2, sentence by sentence

| The brief asks | What this notebook does | Section |
|---|---|---|
| "Run an OLS linear regression of one price on the other: PriceA = α + β · PriceB" | OLS of KO on PEP, on adjusted prices, in-sample only | 3 |
| "Spread = PriceA − β · PriceB" | Exactly this formula (α left out) | 3 |
| "You should actually check it" (plot; does it keep crossing its own average?) | Spread chart, crossings of the mean year by year, half-life, a formal test | 3.1–3.2 |
| "Standardize the spread into a z-score (... away from its own rolling average)" | z = (spread − rolling mean) / rolling standard deviation, past and today only | 4 |
| "Long the spread when very negative, short when very positive, close when it reverts back toward zero" | Enter at ±threshold, exit at z = 0 | 4 |
| "Split your data into an in-sample period and an out-of-sample period ... report both, separately" | 2000–2018 / 2019–2026, fixed before downloading any price; one table with both | 2, 11 |
| "Be explicit about how many things you tried" | Every backtest is logged in `results/trials.sqlite`; the count is printed | 10 |
| "Check how sensitive your result is to small changes (threshold, lookback window)" | 25-variant grid, the same grid on two halves of the in-sample, a 24-variant zoom of small changes | 6–8 |
| "Do not optimize on the out-of-sample period" | Rules fixed in a commit before the first backtest; strategy frozen in a commit before the single out-of-sample run | 5, 11 |"""
    ),
    md(
        """## 2. Set-up and data

Same prices as Phase 1: Yahoo adjusted close, with the duplicated KO dividend of 10 September 2001
removed (see `01_data.ipynb`, section 3.4). **Sections 3–10 use only the in-sample period, 3 January
2000 – 31 December 2018**: the `select_until` call below drops every later date and an `assert`
checks it. The out-of-sample prices are loaded only in section 11."""
    ),
    code(
        """from datetime import date

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import coint

from pairs.backtest import run_zscore_strategy, select_stable_cell, summarize
from pairs.config import PAIR, PHASE2_STRATEGY, SIGNAL_GRID, Settings
from pairs.data import KNOWN_DIVIDEND_ERRORS, correct_dividend_errors, fetch_raw_prices
from pairs.explore import select_until
from pairs.logging_setup import configure_logging
from pairs.plots import finish_figure, save_figure
from pairs.spread import build_spread, fit_hedge_ratio, half_life, mean_crossings_per_year
from pairs.trials import count_trials, log_trial

configure_logging()
pio.renderers.default = "plotly_mimetype+png"
settings = Settings.from_env()
figures_dir = settings.results_dir / "figures"
trials_db = settings.results_dir / "trials.sqlite"
SOURCE_IS = "Source: Yahoo Finance adjusted close, in-sample period 3 Jan 2000 - 31 Dec 2018."
A, B, PAIR_NAME = PAIR.ticker_a, PAIR.ticker_b, f"{PAIR.ticker_a}/{PAIR.ticker_b}"


def load_prices(last_day: date) -> pd.DataFrame:
    raw = fetch_raw_prices(PAIR, settings.data_dir)
    raw = select_until(raw.loc[raw.index >= pd.Timestamp(PAIR.start)], last_day)
    return correct_dividend_errors(raw["Adj Close"][[A, B]], raw, KNOWN_DIVIDEND_ERRORS).dropna()


prices = load_prices(PAIR.in_sample_end)
assert prices.index.max() <= pd.Timestamp(PAIR.in_sample_end)
pd.Series({"Trading days": len(prices), "First day": prices.index[0].date(), "Last day": prices.index[-1].date()})"""
    ),
    md(
        """## 3. Building the spread

The two prices wander like random walks, with no level to return to. The brief's idea: if the two
stocks are truly related, some combination of them should hover around a stable average. The plain
difference KO − PEP does not work because the stocks do not move one-for-one, so we fit the line
`KO = α + β × PEP` by OLS on the **in-sample prices only** (`as_of` = 31 Dec 2018: the function drops
every later date before estimating, and a test checks that changing a later price leaves β unchanged).
β is then frozen: the same number is used later on the out-of-sample period."""
    ),
    code(
        """fit = fit_hedge_ratio(prices, A, B, PAIR.in_sample_end)
spread = build_spread(prices, A, B, fit.beta)
pd.Series(
    {"alpha (dollars)": round(fit.alpha, 3), "beta (hedge ratio)": round(fit.beta, 4),
     "R squared": round(fit.r_squared, 3), "days used": fit.n_obs, "estimated as of": fit.as_of}
).to_frame("OLS of KO on PEP")"""
    ),
    md(
        """**Reading.** β = 0.388: for each dollar PepsiCo moves, Coca-Cola tends to move 39 cents. Long the
spread = buy 1 KO share and sell 0.388 PEP shares. R² = 0.93 only says that both prices grew over the
same 19 years; it does not prove that the spread comes back to its mean. The brief asks to check that
separately.

### 3.1 Does the spread come back to its average?

**How to read it:** a spread that keeps crossing the dashed line (its 2000–2018 average) is
mean-reverting, which is what the strategy needs. A spread that stays on one side for years is
drifting."""
    ),
    code(
        """fig = go.Figure()
fig.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread KO - 0.388 x PEP", line={"color": "#1f4e9c"}))
fig.add_hline(y=spread.mean(), line_dash="dash", line_color="black", annotation_text="2000-2018 average")
finish_figure(fig, "The spread KO - 0.388 x PEP over the in-sample period", "Date", "Spread (dollars)", SOURCE_IS)
save_figure(fig, "02_spread_is", figures_dir)
fig.show()"""
    ),
    md(
        """### 3.2 Three checks: crossings of the average, half-life, a formal test

- **Crossings per year**: the brief's own basic check.
- **Half-life**: how many trading days a deviation of the spread takes, on average, to shrink by half
  (from a regression of the daily change of the spread on its level of the day before).
- **Engle-Granger test**: the standard test of whether a spread built from an estimated β is
  mean-reverting. It needs stricter thresholds than the plain Dickey-Fuller test, because β was fitted
  to make the spread look as stable as possible. A p-value below 0.05 would mean "mean-reverting"."""
    ),
    code(
        """crossings = mean_crossings_per_year(spread)
spread_half_life = half_life(spread)
eg_stat, eg_pvalue, eg_critical = coint(prices[A], prices[B])

fig = go.Figure(go.Bar(x=crossings.index, y=crossings, marker_color="#1f4e9c", name="crossings"))
finish_figure(fig, "How many times a year the spread crosses its 2000-2018 average", "Year",
              "Crossings (number)", SOURCE_IS)
fig.update_xaxes(dtick=1)
save_figure(fig, "02_mean_crossings", figures_dir)
fig.show()
pd.Series(
    {"crossings 2000-2018": int(crossings.sum()), "years with zero crossings": int((crossings == 0).sum()),
     "half-life (trading days)": round(spread_half_life, 1),
     "Engle-Granger statistic": round(eg_stat, 2), "5% threshold": round(eg_critical[1], 2),
     "10% threshold": round(eg_critical[2], 2), "p-value": round(eg_pvalue, 3)}
).to_frame("check")"""
    ),
    md(
        """**Reading.** The check gives an honest *no*. The spread crosses its average 116 times, but almost all
crossings happen in a few years (23 in 2001, 38 in 2015); in 8 years out of 19 it never crosses. It
moves in **waves of several years**: below average from 2004 to 2009, above from 2011 to 2014. A
deviation takes about **192 trading days (9 months)** to halve, and the Engle-Granger test does not
reject a drifting spread (p = 0.12). With a single β for 19 years the long-run relationship is weak.

This does not end the phase. The brief's z-score is measured against the spread's own **rolling**
average, which follows the slow waves: the signal bets on shorter swings around the recent average.
Whether that works is exactly what the validation below has to find out."""
    ),
    md(
        """## 4. The signal

Every day, with the prices known at that day's close:

- **z-score** = (spread today − average of the last N days) / standard deviation of the last N days.
  Only today and the past enter (a test checks that changing a future price leaves today's z unchanged).
- **Flat** → go **long** the spread if z < −threshold, **short** if z > +threshold.
- **Long** → close when z comes back to 0 or above; **short** → close when z comes back to 0 or below
  (the brief: "close the position when it reverts back toward zero").
- The position decided at the close of day t earns the price change of day **t+1** (no look-ahead).

**Daily return** = profit of the two legs / dollars invested the day before (value of KO plus value of
PEP, both counted positive); 0 on days without a position. Adjusted prices include dividends, so the
short leg pays them. **No transaction costs**: the brief does not include them in Phase 2 (it lists them
among the optional extensions), so all results here are *before costs*.

**How to read the chart:** the line is z; when it crosses a dashed threshold a trade opens (triangle),
when it comes back to zero it closes (cross). Example with the chosen variant (threshold 1.5, 60 days)."""
    ),
    code(
        """start = prices.index[max(SIGNAL_GRID.windows) - 1]  # first day with a full 250-day window
example = run_zscore_strategy(prices, A, B, fit.beta, PHASE2_STRATEGY.entry, PHASE2_STRATEGY.window, start)
example = example.loc["2002":"2003"]
pos, prev = example["position"], example["position"].shift(1).fillna(0)
fig = go.Figure()
fig.add_trace(go.Scatter(x=example.index, y=example["z"], name="z-score (60 days)", line={"color": "#555555"}))
for level in (-PHASE2_STRATEGY.entry, PHASE2_STRATEGY.entry):
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
        """## 5. The rules, fixed before the first backtest

The brief: "With enough attempts, something will look good on any fixed slice of history purely by
chance." So every choice below was written in `src/pairs/config.py` (`SIGNAL_GRID`) and committed
(`feat: pre-register phase 2 signal grid and selection rule`) **before running any backtest**. Only the
in-sample spread of section 3 had been looked at.

| Rule | Value | Why |
|---|---|---|
| Entry thresholds | 1.0, 1.5, 2.0, 2.5, 3.0 | The three middle ones are the brief's own example ("1.5, 2, 2.5"). 1.0 and 3.0 are edges: they exist only to see what happens just outside (1.0 is not really "very negative": it trades almost every swing) |
| Rolling windows | 20, 60, 120, 190, 250 trading days | The brief leaves the window free. Section 3.2 measured a half-life of ~192 days: windows much shorter than that bet on short swings around the recent average, windows close to it on the spread's own pace. 20 = a month, 60 and 120 = a quarter and a half-year (the range the brief uses in Phase 3), 190 ≈ the half-life, 250 = a trading year. The grid covers both ideas |
| Exit | z = 0 | The brief: "close the position when it reverts back toward zero". Not tuned: one thing less tried |
| Hedge ratio | β = 0.388, OLS on 2000-2018, then frozen | The in-sample is "where you look and tune": β is estimated there and never re-estimated on later data |
| Costs | none | Not part of the brief's Phase 2 |
| Same period for all variants | from 27 Dec 2000 | The first day the 250-day window is full; shorter windows would otherwise get an extra year |
| **How the variant is chosen** | see below | Chosen before seeing any result |

**The selection rule.** Picking the single best of 25 results means picking the luckiest one. So we
look at **zones**, not cells: for each of the 9 *interior* cells (thresholds 1.5/2/2.5 × windows
60/120/190) we average the Sharpe ratio of the cell and its 8 neighbours, and choose the cell with the
highest average, the centre of the zone where the strategy works even if the parameters are a bit off.
**Edge cells cannot be chosen**: they have only 5 or 3 neighbours, so their average is noisier and the
comparison would not be fair; they still count as neighbours. A cell needs **at least 20 in-sample
trades** (a Sharpe ratio over a handful of trades is chance). Ties go to fewer trades. Whatever the
outcome, even if every cell lost, the chosen cell goes to the out-of-sample test and the result is
reported as it is. Written down in advance: with 19 years of data, an in-sample Sharpe below about
0.4 is compatible with pure luck after 25 attempts.

**Small changes (the brief's sensitivity check).** Grid steps are large (60 → 120 days), so after the
choice we also try a **zoom** of 24 small changes around the chosen cell: threshold ±0.1 and ±0.2,
window ±5% and ±10%. The zoom is a *diagnostic only*: it is never used to choose again."""
    ),
    code(
        """pd.DataFrame(
    {"entry thresholds": [SIGNAL_GRID.entry_thresholds], "windows (days)": [SIGNAL_GRID.windows],
     "exit": [SIGNAL_GRID.exit_threshold], "min trades": [SIGNAL_GRID.min_trades],
     "zoom threshold offsets": [SIGNAL_GRID.threshold_offsets], "zoom window factors": [SIGNAL_GRID.window_factors]}
).T.rename(columns={0: "SIGNAL_GRID (src/pairs/config.py)"})"""
    ),
    md(
        """## 6. The 25 in-sample trials

Every variant runs on the same days (27 Dec 2000 – 31 Dec 2018) and is logged in the trial log
`results/trials.sqlite`. Each variant is logged once: re-running this notebook does not add rows.

**How to read the heatmaps:** each square is one variant (row = entry threshold, column = window).
Blue = positive Sharpe ratio, red = negative. A large blue area means the result does not hinge on one
lucky combination; an isolated blue square in a red sea would be a warning."""
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


def heatmap(
    table: pd.DataFrame, title: str, name: str, fmt: str = ".2f", colorscale: str = "RdBu", zmid: float | None = 0.0
) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=table.to_numpy(), x=[str(c) for c in table.columns], y=[str(i) for i in table.index],
        colorscale=colorscale, zmid=zmid, texttemplate="%{z:" + fmt + "}", colorbar={"title": ""},
    ))
    finish_figure(fig, title, "Rolling window (trading days)", "Entry threshold (|z|)", SOURCE_IS + " Before costs.")
    fig.update_layout(width=750, height=480)
    save_figure(fig, name, figures_dir)
    return fig


grid = pd.DataFrame([run_variant(e, w, "phase2-grid") for e in SIGNAL_GRID.entry_thresholds for w in SIGNAL_GRID.windows])
sharpe = grid.pivot(index="entry", columns="window", values="sharpe")
trades = grid.pivot(index="entry", columns="window", values="n_trades")
heatmap(sharpe, "In-sample Sharpe ratio of the 25 variants", "02_grid_sharpe").show()"""
    ),
    code(
        """heatmap(trades, "Number of trades of the 25 variants (2001-2018)", "02_grid_trades", fmt=".0f",
        colorscale="Greys", zmid=None).show()
grid.set_index(["entry", "window"])[["sharpe", "t_stat", "ann_return", "max_drawdown", "n_trades",
                                     "pct_in_market", "avg_days_per_trade"]].round(2)"""
    ),
    md(
        """**Reading.** All 25 variants make money before costs (Sharpe between 0.09 and 0.47): there is no
lonely lucky cell. But the amounts are small (0.3–3% a year on the dollars invested) and no variant
reaches a t-statistic of 2. Short windows trade a lot (303 trades with threshold 1.0 and 20 days), long
windows and high thresholds very little (12 trades with 3.0 and 250 days: too few to trust).

### 6.1 The selection rule, applied

The table shows, for each interior cell, the average Sharpe ratio of the cell and its 8 neighbours."""
    ),
    code(
        """values = sharpe.to_numpy()
zone = pd.DataFrame(
    [[values[i - 1 : i + 2, j - 1 : j + 2].mean() for j in range(1, 4)] for i in range(1, 4)],
    index=sharpe.index[1:4], columns=sharpe.columns[1:4],
)
chosen = select_stable_cell(sharpe, trades, SIGNAL_GRID.min_trades)
assert chosen == (PHASE2_STRATEGY.entry, PHASE2_STRATEGY.window)  # the frozen strategy is the rule's output
print(f"Chosen: entry threshold {chosen[0]}, window {chosen[1]} days")
zone.round(3)"""
    ),
    md(
        """**Reading.** The rule picks **entry 1.5, window 60 days** (zone average 0.361). Its own Sharpe ratio,
0.35, is not the best of the grid (0.47 sits on the edge, with 3.0 and 120 days and only 17 trades):
that is the point of the rule. The chosen variant makes 94 trades in 18 years, about 32 days each,
and holds a position 66% of the days."""
    ),
    md(
        """## 7. Is the good zone stable over time?

One Sharpe ratio over 18 years can hide that a variant worked only in some years. The same 25 variants
(no new backtest) are scored separately on the two halves of the in-sample period.

**How to read it:** if the blue zone sits in the same place in both halves, the parameters are stable;
if it moves, the "best" parameters of the whole period are an average of two different regimes."""
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
fig.show()
grid.set_index(["entry", "window"])[["sharpe 2001-2009", "sharpe 2010-2018"]].loc[[chosen]].round(2)"""
    ),
    md(
        """**Reading.** The good zone **moves**. In 2001–2009 short windows (20–60 days) with threshold 2 work
best and the long windows lose; in 2010–2018 long windows (190–250 days) with high thresholds work
best. The "right" parameter is not stable from one decade to the next, a warning sign for the
out-of-sample test. The chosen variant holds up in both halves (0.40 and 0.30)."""
    ),
    md(
        """## 8. Small changes of the parameters

The brief: "If a tiny change collapses the result, that is a sign you found noise." The 24 zoom
variants move the threshold by ±0.1 and ±0.2 and the window by ±5% and ±10% around the chosen cell
(★ in the centre). They are logged as trials and never used to choose again."""
    ),
    code(
        """zoom = pd.DataFrame([
    run_variant(round(chosen[0] + off, 2), round(chosen[1] * fac), "phase2-zoom", f"zoom around {chosen}")
    if (round(chosen[0] + off, 2), round(chosen[1] * fac)) != chosen
    else grid.set_index(["entry", "window"]).loc[chosen].to_dict() | {"entry": chosen[0], "window": chosen[1]}
    for off in SIGNAL_GRID.threshold_offsets for fac in SIGNAL_GRID.window_factors
])
zoom_sharpe = zoom.pivot(index="entry", columns="window", values="sharpe")
heatmap(zoom_sharpe, "Sharpe ratio for small changes around the chosen variant (centre: 1.5, 60 days)",
        "02_zoom_sharpe").show()
zoom[["sharpe", "t_stat"]].describe().loc[["min", "50%", "max"]].round(2)"""
    ),
    md(
        """**Reading.** No collapse: the Sharpe ratio stays between 0.27 and 0.47 (median 0.33) for every small
change. The result does not hinge on a lucky decimal. But robust is not the same as strong: the
t-statistics stay between 1.1 and 2.0."""
    ),
    md(
        """## 9. The chosen variant on the in-sample period

**How to read it:** the top line is the value of 1 dollar invested in the strategy (gains reinvested);
the bottom area is the drawdown, the loss from the best value reached so far."""
    ),
    code(
        """chosen_is = run_zscore_strategy(prices, A, B, fit.beta, chosen[0], chosen[1], start).loc[start:]
is_stats = summarize(chosen_is["ret"], chosen_is["position"])


def equity_figure(
    ret: pd.Series, title: str, name: str, source: str, shade_from: pd.Timestamp | None = None
) -> go.Figure:
    equity = (1 + ret).cumprod()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Value of 1 dollar", line={"color": "#1f4e9c"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=equity.index, y=100 * (equity / equity.cummax() - 1), name="Drawdown (%)",
                             fill="tozeroy", line={"color": "#c0392b"}), row=2, col=1)
    if shade_from is not None:
        fig.add_vrect(x0=shade_from, x1=equity.index[-1], fillcolor="#f39c12", opacity=0.12, line_width=0)
    finish_figure(fig, title, "", "Value of 1 dollar", source)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    save_figure(fig, name, figures_dir)
    return fig


equity_figure(chosen_is["ret"], "Chosen variant (1.5, 60 days), in-sample, before costs", "02_equity_is",
              SOURCE_IS).show()
pd.Series(is_stats).round(3).to_frame("in-sample 2001-2018")"""
    ),
    md(
        """**Reading.** One dollar becomes about 1.48 dollars in 18 years (about +2.4% a year) with a worst
drawdown of 14%. A Sharpe ratio of 0.35 with a t-statistic of 1.49: below the luck threshold written
down in section 5. Two more reasons to read this number as optimistic: β was estimated on the same
years the backtest runs on (the 2003 trades use a β that "knows" 2018), and there are no costs."""
    ),
    md(
        """## 10. How many things we tried

Every backtest of the project goes through the trial log `results/trials.sqlite`. The README reports
the same count."""
    ),
    code(
        """pd.Series({phase: count_trials(trials_db, phase) for phase in ("phase2-grid", "phase2-zoom")}
          | {"in-sample total": count_trials(trials_db, "phase2-grid") + count_trials(trials_db, "phase2-zoom")}
          ).to_frame("backtests logged")"""
    ),
    md(
        """**49 in-sample backtests**: 25 to choose (one rule, written in advance) and 24 to check small changes.
Nothing else was tried: no other exit level, stop-loss, β window or regression direction. We decided
not to add more tuning because section 7 shows the best parameters change from one decade to the next:
more tuning would only describe the past better."""
    ),
    md(
        """## 11. Out-of-sample: the single run

**The order of events, visible in the Git history.** (1) The variant chosen by the rule (threshold 1.5,
window 60, exit 0, β = 0.388372) was written in `config.py` as `PHASE2_STRATEGY` and committed
(`feat: freeze the phase 2 strategy before the out-of-sample run`). (2) The trial log with the 49
in-sample backtests was committed (`chore: log the 49 in-sample phase 2 trials`). (3) Only then were
2019–2026 prices used, **once** (`chore: log the single phase 2 out-of-sample run`).

This cell re-executes that same run with the same frozen values, so it gives the same numbers and the
log does not count it again. From here on the strategy is not changed: everything later is analysis.

**Out-of-sample details.** 2 January 2019 – 9 October 2026. The strategy starts **flat** on the first
day, so no in-sample trade spills over. The z-score of early January 2019 uses the last 60 days of
2018: that is the past, so it is allowed. Pre-registered reading: the strategy *survives* if its
out-of-sample Sharpe ratio is positive, like in-sample."""
    ),
    code(
        """SOURCE_ALL = (
    "Source: Yahoo Finance adjusted close, 3 Jan 2000 - 9 Oct 2026."
    " Shaded: out-of-sample (run once)."
)
all_prices = load_prices(PAIR.end)
assert abs(fit_hedge_ratio(all_prices, A, B, PAIR.in_sample_end).beta - PHASE2_STRATEGY.beta) < 1e-6  # frozen β
S = PHASE2_STRATEGY
oos_start = all_prices.index[all_prices.index > pd.Timestamp(PAIR.in_sample_end)][0]
oos = run_zscore_strategy(all_prices, A, B, S.beta, S.entry, S.window, oos_start, S.exit).loc[oos_start:]
oos_stats = summarize(oos["ret"], oos["position"])
log_trial(trials_db, "phase2-oos", PAIR_NAME, "out-of-sample", (oos_start.date(), oos.index[-1].date()),
          {"entry": S.entry, "window": S.window, "exit": S.exit, "beta": S.beta}, oos_stats,
          "single out-of-sample run of the frozen strategy")
comparison = pd.DataFrame({"in-sample 2001-2018": is_stats, "out-of-sample 2019-2026": oos_stats})
comparison.round(3)"""
    ),
    code(
        """both = pd.concat([chosen_is["ret"], oos["ret"]])
equity_figure(both, "Chosen variant (1.5, 60 days): in-sample, then the single out-of-sample run", "02_equity_is_oos",
              SOURCE_ALL + " Before costs.", shade_from=oos_start).show()
yearly = oos["ret"].groupby(oos.index.year).apply(lambda r: (1 + r).prod() - 1)
(100 * yearly).round(1).to_frame("out-of-sample return (%)").T"""
    ),
    md(
        """**Verdict: the strategy does not survive the out-of-sample test.** The Sharpe ratio goes from +0.35
to **−0.53**; the strategy loses about 3.8% a year with a worst drawdown of 32%, most of it in 2025
(−15%) and 2026 (−9%). The t-statistic (−1.49) does not prove the strategy is a sure loser either:
7.8 years are too few to separate a small effect from zero. What it shows is that there is **no
evidence it makes money**.

**First, look for a mistake.** No missing out-of-sample prices; the largest daily moves are the real
ones of March 2020; the number of trades per year (36 in 7.8 years) matches the in-sample (94 in 18);
β re-estimated from the data equals the frozen one.

### 11.1 Why it fails

**How to read the chart:** the same spread `KO − 0.388 × PEP`, now over 2000–2026. The dashed line
is the in-sample average. Mean reversion needs the line to keep coming back to it."""
    ),
    code(
        """full_spread = build_spread(all_prices, A, B, S.beta)
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
    "out-of-sample profit of long trades ($ per KO share)": round(oos.loc[held > 0, "pnl"].sum(), 2),
    "out-of-sample profit of short trades ($ per KO share)": round(oos.loc[held < 0, "pnl"].sum(), 2),
    "KO/PEP price ratio, 2023 average": round(ratio.loc["2023"].mean(), 3),
    "KO/PEP price ratio, 2026 average": round(ratio.loc["2026"].mean(), 3),
    "spread, 2026 average (in-sample average 2.41)": round(full_spread.loc["2026"].mean(), 2),
}).to_frame("out-of-sample")"""
    ),
    md(
        """**Reading.** From 2024 Coca-Cola grows much faster than PepsiCo: the price ratio goes from 0.35 to
0.56 and the spread climbs from about −7 to +24 dollars, far above anything seen in 2000–2018. A
mean-reversion strategy keeps betting that the climb will reverse: it goes short the spread again and
again, and the rolling average keeps catching up with the spread at a loss. Most of the loss comes from
the **short** trades. This is the case the brief describes in Phase 3: the relationship can "drift or
break down entirely over time".

It is also consistent with the 50-year view of Phase 1: the KO/PEP ratio moved in waves lasting
decades (1.02 in 1974, 0.31 in 1982, 1.17 in 1998). The 2024–2026 drift is the latest of those waves,
not a one-off accident."""
    ),
    md(
        """## 12. What we learn, and what I am unsure about

**What we learn**
- The brief's spread check fails for KO/PEP with a static β: the spread moves in multi-year waves,
  half-life ~9 months, Engle-Granger p = 0.12.
- In-sample the z-score signal earns a little in every variant tried and is robust to small changes,
  but weak (Sharpe 0.35, t = 1.49) and its best parameters move between decades.
- Out-of-sample it loses (Sharpe −0.53). **50 backtests, rules fixed in advance, one out-of-sample
  run: the signal did not survive.** Our honest read: the in-sample gains were mostly a description of
  that period's swings, and a static β cannot follow a relationship that drifts for years.

**What I am unsure about**
- The in-sample results are optimistic: β uses the same years, and there are no costs (costs would
  lower every number; they are treated in the extension).
- 7.8 out-of-sample years are few: even a real Sharpe ratio of 0.5 could look like zero there.
- One pair and one split: Phase 5 (simulated paths) and the walk-forward extension on 1972–2026 test
  whether this conclusion depends on this particular slice of history.
- β estimated on price levels gives more weight to the later, higher prices.

**An idea to test, not a result: a moving β.** The failure comes from a β frozen on 2000–2018 while the
KO/PEP relationship kept drifting. A β re-estimated on a rolling window of past data (Phase 3) or
updated every day by a Kalman filter (extension) would follow that drift, and over the long run it
could have kept the spread closer to a stable average, with fewer losing short trades in 2024–2026.
It could also fail: a β that adapts quickly also absorbs part of the divergences the strategy wants to
trade, and it always lags a sudden break. So it is a hypothesis, to be tested later with its own rules
fixed in advance and its own out-of-sample run, without changing the Phase 2 strategy frozen here."""
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
assert count_trials(trials_db, "phase2-grid") == 25 and count_trials(trials_db, "phase2-zoom") == 24
assert count_trials(trials_db, "phase2-oos") == 1
assert oos_start == pd.Timestamp("2019-01-02") and oos.index[-1] == pd.Timestamp("2026-10-09")
assert abs(oos_stats["sharpe"] + 0.53) < 0.005 and abs(oos_stats["ann_return"] + 0.038) < 0.001
assert abs(oos_stats["t_stat"] + 1.49) < 0.005 and abs(oos_stats["max_drawdown"] + 0.32) < 0.005
assert oos_stats["n_trades"] == 36
assert abs(yearly[2025] + 0.154) < 0.001 and abs(yearly[2026] + 0.086) < 0.001
assert oos.loc[held < 0, "pnl"].sum() < oos.loc[held > 0, "pnl"].sum() < 0
for name in ("02_spread_is", "02_mean_crossings", "02_signal_example", "02_grid_sharpe", "02_grid_trades",
             "02_grid_halves", "02_zoom_sharpe", "02_equity_is", "02_equity_is_oos", "02_spread_full"):
    assert (figures_dir / f"{name}.png").exists(), name"""
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (.venv)", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, sys.argv[1])

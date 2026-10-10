"""Patch build_nb01.py: why 2000, pre-2000 context section, updated checks."""

from pathlib import Path

p = Path(__file__).with_name("build_nb01.py")
s = p.read_text(encoding="utf-8")
Q = '"""'


def rep(old: str, new: str) -> None:
    global s
    assert old in s, old[:80]
    s = s.replace(old, new)


# --- Set-up: why 2000 and why Phase 1 on in-sample only
rep(
    """**We only look at the in-sample period, 3 Jan 2000 – 31 Dec 2018.** The years after that are the final
exam of the strategy: looking at them now, even just at a chart, could influence our choices and would
make that final test dishonest.\"\"\"""",
    """**Why the sample starts in 2000, although Yahoo has prices back to 1972 for both stocks.**

1. **PepsiCo only became comparable to Coca-Cola at the end of the 1990s.** Until 1997 it also owned
   Pizza Hut, KFC and Taco Bell, spun off as Tricon Global Restaurants (today Yum! Brands) effective
   6 Oct 1997 ([SEC filing](https://www.sec.gov/Archives/edgar/data/0001041061/000104106198000004/0001041061-98-000004.txt));
   its bottling arm was listed separately as Pepsi Bottling Group on 31 Mar 1999
   ([SEC filing](https://www.sec.gov/Archives/edgar/data/0001076405/000107640599000017/0001076405-99-000017.txt)).
   From 2000 both are beverage and snack brand companies. One large change stays inside our sample and
   is worth remembering: PepsiCo merged with Quaker Oats (owner of Gatorade) on 2 Aug 2001
   ([CNN](https://money.cnn.com/2001/08/02/deals/pepsico/)).
2. **Trading worked differently before 2001.** US stocks were quoted in fractions of a dollar (1/8,
   1/16) until the New York Stock Exchange switched to cents on 29 Jan 2001 (Nasdaq on 9 Apr 2001)
   ([The Daily Record](https://thedailyrecord.com/2001/01/29/nyse-goes-totally-decimal-8212-finally/)),
   so trading costs were much higher and not comparable with today's.
3. **The spread of Phase 2 comes from a regression on price levels.** From 1972 to 2018 the adjusted
   prices grow 131 times (KO) and 236 times (PEP), against 2.8 and 4.7 times from 2000: one regression
   over 1972–2018 would be dominated by the most recent years.

The start date was fixed in the code before any price was downloaded. Having now seen the 2000–2018
data, it **cannot be moved** (for example to 2005, after the early drift seen below): that would be
choosing the sample by looking at the results. The history before 2000 is still shown, as context with
less weight, in section 8.

**We only look at the in-sample period, 3 Jan 2000 – 31 Dec 2018.** The years after that are the final
exam of the strategy: looking at them now, even just at a chart, could influence our choices and would
make that final test dishonest. The brief requires this split for testing (Phase 2 onwards); applying it
to this descriptive phase too is our own, more cautious choice. After the single out-of-sample run at the
end of the project, the same charts are shown on the full 2000–2026 period.\"\"\"""",
)

rep(
    "from pairs.data import (\n    KNOWN_DIVIDEND_ERRORS,\n    VALIDATED_FILE,\n"
    "    fetch_corporate_actions,\n    fetch_raw_prices,\n)",
    "from pairs.data import (\n    KNOWN_DIVIDEND_ERRORS,\n    VALIDATED_FILE,\n"
    "    correct_dividend_errors,\n    fetch_context_prices,\n"
    "    fetch_corporate_actions,\n    fetch_raw_prices,\n)",
)

# --- Section 8: context before 2000, inserted before "What we learn"
CONTEXT = f"""    md(
        {Q}## 8. Before 2000: context, with less weight

The brief asks whether the relationship drifts "over the decades". Our sample covers two; Yahoo has
prices for both stocks from **1 June 1972** (PepsiCo's first day on Yahoo; Coca-Cola starts in 1962).
This history lies *before* the in-sample period, so looking at it cannot contaminate the final test,
and nothing in the project is estimated on it. It is given **less weight** for the three reasons of
section 1: PepsiCo was a different company (restaurants until 1997, bottlers until 1999), prices were
quoted in fractions of a dollar, and price levels were 100-200 times lower.

The same checks as for the main data (saved in `data/prices_context_raw.pkl`):{Q}
    ),
    code(
        {Q}context_raw = fetch_context_prices(PAIR, settings.data_dir)
context_adj = context_raw["Adj Close"][["KO", "PEP"]]
both_quoted = context_adj.notna().all(axis=1)
context_checks = pd.DataFrame(
    [
        ("Days with only KO quoted (before PEP's first day, removed)", int((~both_quoted).sum())),
        ("Days with both stocks", int(both_quoted.sum())),
        ("Missing values on those days", int(context_raw.loc[both_quoted].isna().sum().sum())),
        ("Duplicate dates", int(context_raw.index.duplicated().sum())),
        ("First common day", context_adj.loc[both_quoted].index[0].date()),
    ],
    columns=["Check", "Result"],
).set_index("Check")
context_checks{Q}
    ),
    md(
        {Q}**Splits and dividends before 2000.** Both companies split their shares several times; Yahoo applies
all of them backwards, as in 2012. Among the dividends there is one special record: on 7 Oct 1997
Yahoo lists a "dividend" of 3.20 dollars per PepsiCo share, which is the value of the Tricon shares
distributed in the spin-off (one Tricon share per ten PepsiCo shares). Treating a spin-off as a cash
distribution is the standard way to adjust prices for it. The amount is plausible: Tricon traded at
about 31 dollars in its first days, i.e. about 3.1 dollars per PepsiCo share (not verified to the
cent).{Q}
    ),
    code(
        {Q}all_events = fetch_corporate_actions(PAIR, settings.data_dir)
before = all_events.loc[(all_events.index >= pd.Timestamp("1972-06-01")) & (all_events.index < pd.Timestamp(PAIR.start))]
context_splits = before.loc[before["split"] > 0, ["ticker", "split"]].rename(columns={{"split": "new shares per old share"}})
spin_off = before.loc[(before["ticker"] == "PEP") & (before.index == pd.Timestamp("1997-10-07")), ["ticker", "dividend"]]
display(
    context_splits,
    before.loc[before["dividend"] > 0].groupby("ticker").size().to_frame("dividends 1972-1999"),
    spin_off.rename(columns={{"dividend": "Tricon spin-off recorded as a dividend ($)"}}),
){Q}
    ),
    md(
        {Q}**Extreme days before 2000**, with the same evidence as in section 4:{Q}
    ),
    code(
        {Q}common = context_raw.loc[both_quoted]
largest_moves(common["Close"][["KO", "PEP"]], common["Volume"][["KO", "PEP"]], 6).round(3){Q}
    ),
    md(
        {Q}The three KO days of 19-21 October 1987 are the stock market crash of "Black Monday" and the rebound
after it, on 7-11 times normal volume. PepsiCo's -14.3% of 14 December 1982 traded on 17 times normal
volume and kept its level, so it is a real company event (its cause is not investigated here). The pair
of PepsiCo days of 31 August - 1 September 1998 (-11.2% then +16.1%) falls in a market-wide crash and
rebound, but the second move is larger than the first and the volume only 2.5-2.7 times normal: we
**cannot rule out a data problem** there. It has no effect on the project, since this period is only
shown as context.

**The ratio and the correlation over five decades.** To compare the two periods on one chart, the
history before 2000 is joined to the in-sample data, with the same dividend correction of section 3.4
applied to it (the wrong dividend of 2001 also lowered every earlier KO adjusted price).{Q}
    ),
    code(
        {Q}long_raw = pd.concat([common, raw])
long_prices = correct_dividend_errors(long_raw["Adj Close"][["KO", "PEP"]], long_raw, KNOWN_DIVIDEND_ERRORS)
long_ratio = long_prices["KO"] / long_prices["PEP"]
long_by_year = yearly_correlation(daily_returns(long_prices), "KO", "PEP")
in_sample_days = long_prices.index >= pd.Timestamp(PAIR.start)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=long_ratio.index[~in_sample_days],
        y=long_ratio[~in_sample_days],
        name="1972-1999: context only",
        line={{"color": "#aaaaaa"}},
    )
)
fig.add_trace(
    go.Scatter(
        x=long_ratio.index[in_sample_days],
        y=long_ratio[in_sample_days],
        name="2000-2018: in-sample",
        line={{"color": "#444444"}},
    )
)
finish_figure(fig, "Price ratio Coca-Cola / PepsiCo since 1972 (adjusted prices)", "Date", "KO price / PEP price", "Source: Yahoo Finance, 1972-2018. Before 2000: context only, not used for any estimate.")
save_figure(fig, "01_context_ratio", figures_dir)
fig.show()

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=long_by_year.index,
        y=long_by_year,
        marker_color=["#aaaaaa" if year < PAIR.start.year else "#1f4e9c" for year in long_by_year.index],
        name="Correlation within the year (grey: context before 2000)",
    )
)
finish_figure(fig, "Correlation of KO and PEP daily returns, year by year, since 1972", "Year", "Correlation (-1 to +1)", "Source: Yahoo Finance, 1972-2018 (1972: from June). Before 2000: context only.")
fig.update_yaxes(range=[0, 1])
save_figure(fig, "01_context_correlation", figures_dir)
fig.show()

context_returns = daily_returns(long_prices.loc[~in_sample_days])
context_overall = context_returns["KO"].corr(context_returns["PEP"])
pd.Series(
    {{
        "Correlation of daily returns 1972-1999": context_overall,
        "Correlation of daily returns 2000-2018": overall,
        "Ratio KO/PEP: lowest before 2000": long_ratio[~in_sample_days].min(),
        "Ratio KO/PEP: highest before 2000": long_ratio[~in_sample_days].max(),
    }}
).round(2).to_frame("value"){Q}
    ),
    md(
        {Q}**How to read it.** Before 2000 the ratio does not oscillate quickly around one level: it moves in
**long waves lasting about a decade**. Coca-Cola loses ground from 1972 to 1982 (ratio from 0.78 to its
low of 0.31), gains it back and more until 1998 (high of 1.17), then loses it again until about 2005,
which is the drift already seen at the start of our sample. The correlation of daily returns before
2000 is a little lower (0.50 against 0.55) and varies more from year to year (from 0.25 to 0.78).
With the caveats above, this suggests that the two prices are linked day to day, but that their
relative level can drift for many years before it comes back: a warning for any strategy that expects
the spread to revert within weeks or months.{Q}
    ),
"""

rep('    md("## 8. What we learn\\n\\n" + LEARN),', CONTEXT + '    md("## 9. What we learn\\n\\n" + LEARN),')
rep('"""## 9. Checks', '"""## 10. Checks')

# --- Checks
rep(
    """for name in ("01_prices_rebased", "01_price_ratio", "01_yearly_correlation", "01_dividend_yield"):""",
    """# Context before 2000: only before the sample, same correction, matches the in-sample data exactly
assert common.index.max() < pd.Timestamp(PAIR.start) and len(common) == 6970
assert common.index[0] == pd.Timestamp("1972-06-01")
assert len(context_splits) == 9
assert abs(spin_off["dividend"].iloc[0] - 3.2) < 1e-9
assert (long_prices.loc[in_sample_days] - prices).abs().max().max() < 1e-9
assert abs(context_overall - 0.50) < 0.005, context_overall
assert abs(long_ratio[~in_sample_days].min() - 0.31) < 0.005 and abs(long_ratio[~in_sample_days].max() - 1.17) < 0.005

for name in (
    "01_prices_rebased",
    "01_price_ratio",
    "01_yearly_correlation",
    "01_dividend_yield",
    "01_context_ratio",
    "01_context_correlation",
):""",
)

p.write_text(s, encoding="utf-8")
print("patched")

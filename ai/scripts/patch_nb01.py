"""Patch build_nb01.py: split check, dividend error section, updated asserts."""

from pathlib import Path

p = Path(__file__).with_name("build_nb01.py")
s = p.read_text(encoding="utf-8")


def rep(old: str, new: str) -> None:
    global s
    assert old in s, old[:80]
    s = s.replace(old, new)


Q = '"""'

rep(
    "from pairs.data import VALIDATED_FILE, fetch_corporate_actions, fetch_raw_prices",
    "from pairs.data import (\n    KNOWN_DIVIDEND_ERRORS,\n    VALIDATED_FILE,\n"
    "    fetch_corporate_actions,\n    fetch_raw_prices,\n)",
)

rep(
    """**One split in 19 years: Coca-Cola, 2-for-1, on 13 Aug 2012.** PepsiCo had none. Yahoo divides all
of KO's earlier prices (and dividends) by 2, so that the series has no artificial halving in 2012.
Consequence: **the KO close of 28.19 on 3 Jan 2000 means about 56.4 dollars on the market that day,
well above PepsiCo's 36.9.** At the start of our sample Coca-Cola was the more expensive share.
""",
    f"""**One split in 19 years: Coca-Cola, 2-for-1, on 13 Aug 2012.** PepsiCo had none.

**How the split is handled.** We do not adjust anything ourselves: Yahoo already rewrites KO's history
backwards, dividing every price before 13 Aug 2012 by 2 (open, high, low, close and adjusted close),
multiplying the volumes by 2 and dividing the dividend amounts by 2. Without this, KO's price would
show a fake -50% drop on the split day. We check that it is really the case:{Q}
    ),
    code(
        {Q}split_day = pd.Timestamp("2012-08-13")
around_split = raw.xs("KO", axis=1, level=1).loc["2012-08-08":"2012-08-16", ["Close", "Adj Close", "Volume"]]
around_split["Close return %"] = 100 * raw[("Close", "KO")].pct_change().loc[around_split.index]
ko_volume = raw[("Volume", "KO")]
volume_check = pd.Series(
    {{
        "median daily volume, 3 months before (M shares)": ko_volume.loc["2012-05-13":"2012-08-10"].median() / 1e6,
        "median daily volume, 3 months after (M shares)": ko_volume.loc["2012-08-13":"2012-11-13"].median() / 1e6,
    }}
)
display(around_split.round(2), volume_check.round(1).to_frame("value")){Q}
    ),
    md(
        {Q}On the split day KO moves by -0.24%, an ordinary day, and the traded volume is of the same size
before and after: prices and volumes are consistently split-adjusted. (Over the whole period KO's
largest daily moves are -10.1% and +13.9%, so no other split is hiding in the data.)

Consequence for reading the data: **the KO close of 28.19 on 3 Jan 2000 means about 56.4 dollars on
the market that day, well above PepsiCo's 36.9.** At the start of our sample Coca-Cola was the more
expensive share.
""",
)

rep(
    """the average share price of the year, i.e. the percentage of the price paid out in cash.\"\"\"""",
    """the average share price of the year, i.e. the percentage of the price paid out in cash. One record
reported by Yahoo is wrong and is left out here; it is explained in section 3.4.\"\"\"""",
)
rep(
    """dividends = events.loc[events["dividend"] > 0]
close = raw["Close"]""",
    """reported = events.loc[events["dividend"] > 0]  # everything Yahoo reports
wrong = [(pd.Timestamp(e.ex_date), e.ticker) for e in KNOWN_DIVIDEND_ERRORS]
is_wrong = [(day, ticker) in wrong for day, ticker in zip(reported.index, reported["ticker"], strict=True)]
dividends = reported.loc[[not flag for flag in is_wrong]]  # the dividends actually paid
close = raw["Close"]""",
)

rep(
    """**Check: are Yahoo's adjusted prices explained by the dividends above, and by nothing else?** We
rebuild the adjustment ourselves from the dividend list (each price before an ex-date is multiplied by
`1 - dividend / previous close`) and compare it with Yahoo's ratio adjusted / close, over the whole
in-sample period. The largest difference on any day:""",
    """**Check: are Yahoo's adjusted prices explained by the dividends Yahoo reports, and by nothing else?**
We rebuild the adjustment ourselves from Yahoo's dividend list (each price before an ex-date is
multiplied by `1 - dividend / previous close`) and compare it with Yahoo's ratio adjusted / close, over
the whole in-sample period. The largest difference on any day:""",
)
rep(
    """    paid = dividends.loc[dividends["ticker"] == ticker, "dividend"]
    ours = adjustment_from_dividends(close[ticker], paid)
    yahoo""",
    """    paid = reported.loc[reported["ticker"] == ticker, "dividend"]
    ours = adjustment_from_dividends(close[ticker], paid)
    yahoo""",
)

rep(
    """The differences are a few thousandths of a percent (rounding of prices to the cent), so **the
adjusted prices contain the reported dividends and nothing else**: no hidden correction, no missing
event.

Because levels""",
    f"""The differences are a few thousandths of a percent (rounding of prices to the cent), so **the
adjusted prices contain the reported dividends and nothing else**. This check proves consistency, not
correctness: if a reported dividend is itself wrong, it is inside the adjusted prices too, and the
check cannot see it. Section 3.4 shows that this happens once.

### 3.4 An error in Yahoo's data: a dividend that was never paid

Listing KO's dividends date by date shows two records in the week of 11 September 2001:{Q}
    ),
    code(
        {Q}reported.loc["2001-06-01":"2001-12-31"].query("ticker == 'KO'"){Q}
    ),
    md(
        {Q}KO paid 0.09 dollars per quarter that year (split-adjusted), and the ex-date of September moved to the
17th because the exchange was closed from the 11th to the 14th. **The 0.18 record of 10 September is a
duplicate of the same dividend at its amount before the 2012 split (0.09 x 2)**, i.e. a split-handling
error in Yahoo's data. It is the only such case in 2000-2018 (no other two dividends of the same stock
are less than 40 days apart). Yahoo applied it to the adjusted prices, which creates a fake gain on
10 September 2001:{Q}
    ),
    code(
        {Q}error_day = pd.Timestamp("2001-09-10")
prices = select_until(pd.read_pickle(settings.data_dir / VALIDATED_FILE), PAIR.in_sample_end)
fake_gain = pd.Series(
    {{
        "Close (price only, no dividend that day)": raw[("Close", "KO")].pct_change().loc[error_day],
        "Yahoo adjusted close (with the wrong dividend)": raw[("Adj Close", "KO")].pct_change().loc[error_day],
        "Our adjusted close (corrected)": prices["KO"].pct_change().loc[error_day],
    }}
)
fake_gain.mul(100).round(2).to_frame("KO return on 2001-09-10 (%)"){Q}
    ),
    md(
        {Q}**Correction** (`correct_dividend_errors` in `src/pairs/data.py`, applied before the "validated"
checkpoint): the adjusted KO prices before 10 September 2001 are divided back by the factor Yahoo
applied, so they rise by 0.73% (425 trading days); PepsiCo is untouched and the raw download is kept
exactly as Yahoo sent it. After the correction the return of that day equals the price return, as it
must, and our adjusted prices match the list of dividends **actually paid**:{Q}
    ),
    code(
        {Q}corrected_error = {{}}
for ticker in ("KO", "PEP"):
    paid = dividends.loc[dividends["ticker"] == ticker, "dividend"]
    ours = adjustment_from_dividends(close[ticker], paid)
    used = prices[ticker] / close[ticker]
    corrected_error[ticker] = float((ours / (used / used.iloc[-1]) - 1).abs().max())
pd.Series(corrected_error, name="largest difference (%)").mul(100).round(4).to_frame(){Q}
    ),
    md(
        {Q}Because levels""",
)

rep(
    """        \"\"\"prices = select_until(pd.read_pickle(settings.data_dir / VALIDATED_FILE), PAIR.in_sample_end)
rebased = 100 * prices / prices.iloc[0]""",
    """        \"\"\"rebased = 100 * prices / prices.iloc[0]  # prices = validated and corrected (section 3.4)""",
)

rep(
    """assert (dividends["ticker"] == "KO").sum() == 77 and (dividends["ticker"] == "PEP").sum() == 76
assert max(rebuild_error.values()) < 1e-4, rebuild_error""",
    """assert len(reported) == 153 and len(dividends) == 152  # one wrong record removed
assert (dividends["ticker"] == "KO").sum() == 76 and (dividends["ticker"] == "PEP").sum() == 76
assert (dividend_table[("KO", "number")] == 4).all() and (dividend_table[("PEP", "number")] == 4).all()
assert abs(raw[("Close", "KO")].pct_change().loc[split_day]) < 0.01  # no fake -50% on the split day
assert max(rebuild_error.values()) < 1e-4, rebuild_error  # Yahoo consistent with its own list
assert max(corrected_error.values()) < 1e-4, corrected_error  # ours consistent with dividends paid
assert abs(fake_gain.iloc[2] - fake_gain.iloc[0]) < 1e-4  # fake gain removed
assert fake_gain.iloc[1] - fake_gain.iloc[0] > 0.007  # it was there (+0.73%)""",
)

p.write_text(s, encoding="utf-8")
print("patched")

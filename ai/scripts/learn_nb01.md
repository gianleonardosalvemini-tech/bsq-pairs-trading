**About the data**

- 4,779 trading days (3 Jan 2000 – 31 Dec 2018), nothing missing, nothing removed; the longest pauses
  are real closures of the exchange and the extreme days are real market events.
- One stock split (Coca-Cola, 2-for-1, 13 Aug 2012), correctly handled by Yahoo. 152 quarterly
  dividends (76 per company), all reported above.
- **One error in Yahoo's data, found and corrected**: a KO dividend of 10 Sep 2001 that was never paid
  (a duplicate at its pre-split amount). Left in, it would add a fake +0.73% gain to KO on one day.
  This was found by listing dividends one by one, not by the consistency check, which could not see it.

**About the relationship**

- **The two stocks are linked, and increasingly so.** The correlation of daily returns is 0.55 over
  2000–2018, never below 0.3 in any year, and it rises over time: about 0.35 in 2000–01, about 0.45 in
  2003–06, between 0.6 and 0.8 after 2007.
- **The relationship is not stable over the whole period.** In January 2000 Coca-Cola was the more
  expensive share (about 56 dollars against 37) and the price ratio starts high (0.73, with a peak of
  0.9 in early 2000). Between 2000 and 2004 Coca-Cola steadily loses this premium and the ratio falls to
  about 0.45: a drift, not an oscillation. **After 2004 the ratio moves between 0.34 and 0.58 and crosses
  its average again and again**, which is the behaviour pairs trading needs.

**Before 2000 (context, less weight)**

- From 1972 to 1999 the daily link is similar (correlation 0.50), but the price ratio moves in **long
  waves of about a decade** (high near 1.0 in 1974, low of 0.31 in 1982, high of 1.17 in 1998). The
  relative level of the two stocks can drift for years before coming back.

**What this means for the next phases**

- A single hedge ratio estimated once on the whole 2000–2018 period will be pulled by the early drift,
  and the long history warns that drifts of several years are normal for this pair.
  The spread built from it must therefore be **checked, not assumed** (chart, how often it crosses its
  mean) in Phase 2, and the stability of the relationship over time is exactly what Phase 3 tests.

**What I am unsure about**

- I did not investigate *why* Coca-Cola lost ground in 2000–2004 (a commonly cited reason is its very
  high valuation at the end of the 1990s, but I have not verified it): the data shows the drift, not
  its cause.

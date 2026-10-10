"""Scratch: in-sample spread check (2000-2018 only). Not a backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller

from pairs.config import PAIR, Settings
from pairs.data import KNOWN_DIVIDEND_ERRORS, correct_dividend_errors, fetch_raw_prices
from pairs.explore import select_until
from pairs.spread import build_spread, fit_hedge_ratio

settings = Settings.from_env()
raw = select_until(fetch_raw_prices(PAIR, settings.data_dir), PAIR.in_sample_end)
raw = raw.loc[raw.index >= pd.Timestamp(PAIR.start)]
prices = correct_dividend_errors(raw["Adj Close"][["KO", "PEP"]], raw, KNOWN_DIVIDEND_ERRORS).dropna()
print("IS days", len(prices), prices.index[0].date(), prices.index[-1].date())

fit = fit_hedge_ratio(prices, "KO", "PEP", PAIR.in_sample_end)
print(fit)
spread = build_spread(prices, "KO", "PEP", fit.beta)
mean = spread.mean()
print("spread mean %.3f std %.3f min %.3f max %.3f" % (mean, spread.std(), spread.min(), spread.max()))

side = np.sign(spread - mean)
cross = (side != side.shift()).iloc[1:]
per_year = cross.groupby(cross.index.year).sum()
print("crossings of full-IS mean per year:\n", per_year.to_string())
print("total crossings", int(cross.sum()))

lag = spread.shift(1).dropna()
d = spread.diff().dropna()
phi = np.polyfit(lag.to_numpy(), d.to_numpy(), 1)[0]
print("AR slope %.5f half-life days %.1f" % (phi, -np.log(2) / phi))
adf = adfuller(spread.to_numpy(), autolag="AIC")
print("ADF stat %.3f p %.4f" % (adf[0], adf[1]))

for name, (s, e) in {"2000-04": ("2000", "2004"), "2005-09": ("2005", "2009"), "2010-14": ("2010", "2014"), "2015-18": ("2015", "2018")}.items():
    seg = spread.loc[s:e]
    print(name, "mean %.2f std %.2f" % (seg.mean(), seg.std()))

# rolling-window half-life for context (60/120d windows of the spread, mean-reversion speed within window)
fig = go.Figure()
fig.add_scatter(x=spread.index, y=spread, name="spread")
fig.add_hline(y=mean, line_dash="dash")
fig.update_layout(title=f"KO - {fit.beta:.3f} PEP, in-sample", width=1100, height=450)
fig.write_image(r"C:\Users\gianl\AppData\Local\Temp\claude\C--Users-gianl-OneDrive-Documenti-A-Magistrale-poliimi-BSQ\53b42bea-10f1-42e6-a4d4-a2e9f7d46221\scratchpad\spread_is.png")

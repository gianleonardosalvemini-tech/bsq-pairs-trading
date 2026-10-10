# Pairs trading: a mean-reversion signal and how hard I tried not to fool myself

BlackSwanQuants take-home project (returning contributors track), 10 October 2026.

## Summary (half a page)
_To be written at the end of the day: which phases I completed, which method I picked where there was a
choice and why, what I would do with more time._

## How to run
```bash
uv sync                      # exact versions from uv.lock, Python 3.12
uv run pytest                # unit tests
```
Notebooks (one per phase) are in `notebooks/` and run on the `.venv` kernel.
No API key is needed. `.env` is optional (see `.env.example`).

## Repository layout
| Path | Content |
|---|---|
| `src/pairs/` | Library code (typed, documented, logged) |
| `notebooks/` | One notebook per phase, each ending with assertions on its numbers |
| `tests/` | pytest unit tests |
| `results/` | Figures and the SQLite log of every backtest I ran |

## Phases
_One section per phase: method chosen, why, result, honest read._

## Everything I tried
_Number of variants tested and how the shown one was chosen (from `results/trials.sqlite`)._

## Engineering standards
_Which of the programming guidelines I followed and where; which I skipped and why._

## AI usage disclosure
_What I asked the assistant to do, what I checked myself, where I overrode or corrected it._

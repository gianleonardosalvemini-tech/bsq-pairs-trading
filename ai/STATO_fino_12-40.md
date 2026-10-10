# Stato del lavoro

Aggiornato da Claude dopo ogni passo. Una chat nuova riparte da qui.

**Ultimo aggiornamento:** sab 10 ott, 12:40. **CAMBIO CHAT: si rifà la fase 1 aderendo al PDF.**
**Leggi per prima cosa `ai/ripartenza.md`** (passaggio di consegne completo: cosa teniamo, nuova fase 1, decisioni
D1–D3 da prendere subito, tutte le scoperte sui dati con i numeri, lezioni tecniche, tabella di marcia).

## Fatto
- Setup: uv + Python 3.12.15, `pyproject.toml` + `uv.lock`, ruff / mypy (strict) / pytest configurati,
  pre-commit (uv sync --locked, ruff check, ruff format, mypy, pytest), CI GitHub Actions su push a `main` e sulle pull
  request (`actions/checkout@v7`, `astral-sh/setup-uv@v10.3.0`: setup-uv vuole il tag esatto). `.gitattributes`
  forza LF.
- Pacchetto `src/pairs/`: `errors.py` (eccezioni di dominio), `config.py` (`Settings` da `.env`,
  `PairConfig` con lo split in-sample / out-of-sample validato), `logging_setup.py` (Rich). 3 test.
- Repository GitHub privato `gianleonardosalvemini-tech/bsq-pairs-trading`, inviti a `zanisimone` e `giutala`.
- Commit su `main` (tutti pushati, CI verde sull'ultimo):
  `chore: initial project structure`, `ci: update checkout and setup-uv to current releases`,
  `fix: pin setup-uv to an existing release tag`, `refactor: align modules with logging and docstring guidelines`,
  `chore: version data checkpoints for reproducibility`.
- Verificato: API Nasdaq dello short interest funziona per PEP e AMD (serve uno User-Agent da browser),
  non per KO (NYSE), come dice il PDF. Download Yahoo ok, export PNG Plotly (kaleido) ok.

- 10:40, revisione riga per riga contro le linee guida (richiesta dall'utente): logger in **ogni** modulo, docstring
  con Parameters / Returns / Raises / Notes ovunque, hook pre-commit `uv sync --locked`, nessuna eccezione alle
  regole docstring di ruff. Commit `refactor: align modules with logging and docstring guidelines`, CI verde.

- 10:55, audit contro i due PDF originali + GitHub: repo privato ok, inviti `write` a zanisimone e giutala ancora
  pending (dipende da loro), CI verde sull'ultimo commit, hook pre-commit installato, ruff/mypy/pytest puliti in locale.
  Difetti trovati (corretti alle 11:00, commit `fix: align CI sync flag and checkpoint docs`): (a) docstring di `Settings.data_dir` dice "not versioned" ma i
  pickle ora sono versionati; (b) CI usa `uv sync --frozen`, pre-commit `--locked`: allineare a `--locked` (verifica
  anche che il lock sia aggiornato); (c) wiki OneDrive con righe superate (commit fatti dall'utente, pkl fuori da Git,
  decisioni 1–3 "da prendere", copia vecchia di regole_oro): non va consegnato, solo da ripulire.

## Decisioni prese e perché
- Lavoro su `main` con CI su push e pull request (alle 21 conta solo `main`).
- Funzioni in `src/pairs/`, un notebook per fase in `notebooks/` con `assert` finali, test in `tests/`.
- **Commit e push li fa Claude**, dicendo ogni volta cosa fa (decisione dell'utente, 10:30).
- **Le chat si aprono in `BSQ/`** (scelta dell'utente, 10:50): `BSQ/CLAUDE.md` importa questo file e il `CLAUDE.md`
  del repository; si lavora qui con percorsi assoluti. Il materiale del colloquio è in `BSQ/archivio_colloquio/`.
- File dell'AI (`CLAUDE.md`, `ai/`, `.claude/`): **per ora nel `.gitignore`, restano solo sul PC** (decisione
  dell'utente, 10:25). Claude continua ad aggiornarli. **Alla fine** l'utente li rivede e solo allora si tolgono
  le tre righe dal `.gitignore` e si aggiungono al commit (il PDF chiede di lasciarli nella cartella), insieme
  alla riga `ai/`, `CLAUDE.md` nella tabella "Repository layout" del README.

- 10:45, decise con l'utente:
  1. **checkpoint pickle di `data/` versionati** (fatto: tolti dal `.gitignore`, commit
     `chore: version data checkpoints for reproducibility`, CI verde). Motivo: Yahoo riscrive gli Adj Close a ogni
     dividendo, senza i file i numeri non sono riproducibili. Da dire nel README.
  2. **test pytest che esegue tutti i notebook end-to-end** con nbclient (`tests/test_notebooks.py`): da scrivere
     appena esiste il primo notebook; ogni notebook finisce con `assert` sui suoi numeri.
  3. **PySide6 saltato, dichiarandolo nel README**. **Possibilità a fine giornata**, se avanza tempo dopo le fasi 1–5:
     visualizzatore opzionale dei checkpoint (gruppo di dipendenze opzionale, ~40 righe). Ricordarlo all'utente
     verso le 19:00.

## Prove tentate
Nessuna ancora.

- 11:00, **decisione dell'utente** (su proposta di Claude): coppia **KO/PEP**; `PAIR` in `config.py`: inizio
  2000-01-03, **fine in-sample 2018-12-31**, fine 2026-10-09 (OOS ~7,8 anni, ~29%). Fissata **prima di scaricare
  qualunque prezzo**: commit `feat: fix pair and in-sample split before looking at data` (091e49f, 10:45 UTC),
  da citare nel README come prova. Perché il 2000: livelli di prezzo confrontabili per l'OLS sui livelli, mercato
  moderno. **Fase 1 solo sull'in-sample** (guardare i prezzi post-2018 = sbirciare l'OOS). **Adj Close** per
  rendimenti e spread. CI ora `uv sync --locked`.
- 11:05, **l'utente ha messo in dubbio KO/PEP** (INTC/AMD ha lo short interest su entrambe). Dopo il confronto tra
  le tre coppie **ha confermato KO/PEP** (fasi 2, 3, 5 piene; fase 4 su solo PEP, ammessa dal PDF).
  **Fase 6, se c'è tempo: seconda coppia INTC/AMD** come storia della relazione che si rompe nell'OOS e fase 4
  completa. Lo short interest (12 mesi, ott 2025 – ott 2026) cade **tutto nell'OOS**: la fase 4 si fa **dopo** il
  lancio unico dell'OOS ed è dichiarata descrittiva. Da scrivere nel README (scelta coppia + AI disclosure).

- 11:10, `src/pairs/data.py`: `fetch_raw_prices` (Yahoo, `auto_adjust=False`, rilegge il pickle se c'è),
  `parse_prices` (Adj Close, colonne [KO, PEP]), `validate_prices` (scarta i giorni con una gamba mancante, mai
  forward-fill; `StaleDataError` se le date non coprono la finestra ±5 giorni; `assert` su ordine, indice, prezzi > 0).
  4 test in `tests/test_data.py` (8 in tutto). Dato reale: **6.733 giorni, 2000-01-03 → 2026-10-09, 0 scartati**
  (guardati solo forma e date). Checkpoint `data/prices_{raw,parsed,validated}.pkl` committati.

- 11:25, `explore.py` (`select_until` con `as_of`, `daily_returns`, `yearly_correlation`), `plots.py`
  (`finish_figure` = stile comune, `save_figure` = PNG). Commit `feat: add phase 1 return and correlation helpers`.
  `notebooks/01_data.ipynb` (non ancora committato), generato da `scratchpad/build_nb01.py`, eseguito, 3 PNG in
  `results/figures/01_*.png`. **Numeri in-sample**: 4.779 giorni; correlazione rendimenti 0,552 (per anno da 0,35 nel
  2000–01 a 0,74–0,79 nel 2015–18, mai sotto 0,3); rapporto KO/PEP medio 0,454, da 0,9 (2000) a ~0,34 (2006), poi
  0,40–0,58: **forte deriva nei primi anni**, oscillazione attorno alla media dopo il 2004. `assert` fissati su questi.

- 11:50, su richiesta dell'utente: notebook con sezioni Set-up, Raw data, **Splits and dividends** (split KO 2:1
  2012-08-13; dividendi per anno, yield KO 1,2→3,4%, PEP 1,3→3,2%; ricostruzione Adj Close dai dividendi, errore
  ≤ 0,003%), Data cleaning (0 NaN/duplicati/prezzi ≤ 0, 248–253 giorni/anno, buchi = chiusure vere: 11/9, Ford,
  Sandy; 6 movimenti estremi con volume 2,8–7,5× = eventi veri, tenuti). Nuove funzioni testate:
  `longest_gaps`, `largest_moves`, `adjustment_from_dividends`, `fetch_corporate_actions` (`actions_raw.pkl`).
  Log: solo il pacchetto `pairs` a INFO (kaleido intasava il notebook). **Trovato dividendo KO duplicato di Yahoo**
  (0,18 $ il 2001-09-10, oltre al vero 0,09 $ il 2001-09-17): rendimento KO finto +0,73% quel giorno.

- 12:05, **fase 1 chiusa**. Dividendo KO duplicato corretto (`KNOWN_DIVIDEND_ERRORS`, `correct_dividend_errors`,
  validated rigenerato: 425 giorni KO +0,73%, PEP invariato). Notebook 01 con sezioni: Set-up, Raw data, Splits
  and dividends (3.1 split + verifica, 3.2 tabella e grafico yield, 3.3 coerenza Adj Close, 3.4 errore Yahoo e
  correzione), Data cleaning, 3 grafici, What we learn, Checks (assert su tutto). `tests/test_notebooks.py`
  (nbclient, PNG in cartella temporanea; ~20 s). README "Phase 1" con 3 PNG. Numeri invariati dopo la correzione:
  correlazione 0,552, rapporto medio 0,454. **Prove tentate: 0** (nessun backtest ancora).
  Commit: `fix: correct a duplicate KO dividend...`, `test: run every notebook end to end` (contiene anche notebook,
  PNG e validated: due commit rifiutati dal pre-commit, vedi correzioni.md), `docs: add phase 1 to README`.

- 12:30, su domanda dell'utente: **perché il 2000** spiegato nel set-up del notebook e nel README con 3 motivi
  verificati e linkati (PepsiCo: ristoranti scorporati 6/10/1997, imbottigliatori IPO 31/3/1999, Quaker 2/8/2001 dentro
  il campione; prezzi in frazioni fino al 29/1/2001; prezzi ×131 KO e ×236 PEP dal 1972 contro ×2,8 e ×4,7 dal 2000)
  + l'inizio non si sposta più dopo aver visto i dati. Fase 1 solo IS = scelta nostra, dichiarata; a fine progetto,
  dopo l'OOS, stessi grafici su 2000–2026. **Sezione 8 "Before 2000: context"** (`fetch_context_prices`,
  `prices_context_raw.pkl`, 6.970 giorni dal 1972-06-01): correlazione 0,50, rapporto a onde decennali (1,0 nel 1974,
  0,31 nel 1982, 1,17 nel 1998). PNG nel notebook per GitHub. Script dell'AI che generano il notebook in `ai/scripts/`.
  **Prove tentate: 0.**

## Prossimo passo
1. Chiedere all'utente le decisioni D1–D3 di `ai/ripartenza.md` (D1: la fase 1 include l'OOS 2019–2026?).
2. Descrivere la nuova fase 1 in dettaglio e aspettare l'ok; poi rifare notebook, test e README (entro le 13:45).

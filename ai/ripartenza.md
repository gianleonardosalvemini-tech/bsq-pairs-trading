# Ripartenza della fase 1 (sab 10 ott, 12:40): passaggio di consegne completo

**Decisione dell'utente (12:40):** si rifà la fase 1 dall'inizio, **aderendo alla lettera al PDF** (l'aderenza alla
traccia è uno dei criteri). La fase 1 è un **warm-up**: descrizione "macro" su tutta la storia disponibile. Il lavoro
fatto non si butta: infrastruttura, funzioni, test e scoperte sui dati restano; si **riordina** il notebook della fase 1.
Siamo **in ritardo**: tabella di marcia in fondo.

## 1. Cosa teniamo (tutto su `main`, CI verde)
- Setup: uv + Python 3.12, ruff / mypy strict / pytest, pre-commit, CI su push e pull request. Repo privato, inviti
  `write` a `zanisimone` e `giutala` (pending, dipende da loro).
- `src/pairs/config.py`: `PAIR` = KO / PEP, `start` 2000-01-03, `in_sample_end` 2018-12-31, `end` 2026-10-09.
  Fissato nel commit `091e49f` (`feat: fix pair and in-sample split before looking at data`) **prima di scaricare
  qualunque prezzo**: è la prova di pre-registrazione, da citare nel README.
- `src/pairs/data.py`: `fetch_raw_prices` (2000–2026), `fetch_context_prices` (1962–1999, `CONTEXT_START`; PEP parte
  dal 1972-06-01), `fetch_corporate_actions` (dividendi e split dal 1962), `parse_prices` (Adj Close),
  `KNOWN_DIVIDEND_ERRORS` + `correct_dividend_errors`, `validate_prices`. Checkpoint versionati in `data/`:
  `prices_raw.pkl`, `prices_context_raw.pkl`, `actions_raw.pkl`, `prices_parsed.pkl`, `prices_validated.pkl` (corretto).
- `src/pairs/explore.py`: `select_until(prices, as_of)`, `daily_returns`, `yearly_correlation`, `longest_gaps`,
  `largest_moves`, `adjustment_from_dividends`. `src/pairs/plots.py`: `finish_figure` (stile comune), `save_figure`.
- `tests/`: 18 test, compreso `test_notebooks.py` (esegue ogni notebook con nbclient; PNG in cartella temporanea via
  `PAIRS_RESULTS_DIR`).
- Da **rifare**: `notebooks/01_data.ipynb` (generato da `ai/scripts/build_nb01.py` + `learn_nb01.md`: riusabili come
  base, i pezzi di codice funzionano) e il paragrafo "Phase 1" del README.

## 2. La nuova fase 1, nell'ordine del PDF
Testo del PDF: *"Download the daily price history … Plot both price series together, plot the ratio or difference
between them over time, and look at the correlation of their daily returns. Get a feel for whether the relationship
looks stable or whether it visibly drifts over the decades."* Breve: è un riscaldamento.
1. Introduzione per non addetti + glossario (close, adjusted close, split, dividendo, rendimento, correlazione, IS/OOS).
2. **Dati**: Yahoo, Adj Close, periodo **dal 1972-06-01** (primo giorno PEP su Yahoo; KO dal 1962) a fine periodo
   (vedi decisione D1). Unire `prices_context_raw` + `prices_raw` e applicare `correct_dividend_errors` all'insieme.
3. **Split e dividendi su TUTTO il periodo** (lista completa sotto), verifica che nessuno split crei un salto finto,
   dividendi per anno/decennio, ricostruzione dell'Adj Close, duplicato KO 2001 corretto, scorporo Tricon 1997.
4. **Pulizia su tutto il periodo**: valori mancanti, duplicati, prezzi ≤ 0, giorni per anno, buchi, movimenti estremi.
5. **I tre grafici del PDF su tutta la storia**: prezzi insieme (**asse logaritmico**: in 46 anni crescono 131× e 236×,
   in scala lineare non si legge nulla), rapporto KO/PEP, correlazione dei rendimenti giornalieri anno per anno.
6. **Lettura "macro" per decenni** con gli eventi societari sul grafico (1997 Tricon, 1999 PBG, 2001 Quaker, 2012 split).
7. **Perché dal 2000 in poi è più significativo per la strategia** (i 3 motivi verificati, sotto) → la fase 2 stima
   tutto solo sull'in-sample 2000–2018.
8. Cosa impariamo; cosa non so. 9. `assert` su ogni numero.

## 3. Decisioni da prendere SUBITO con l'utente (prima di qualunque grafico)
- **D1. I grafici della fase 1 includono anche il 2019–2026 (l'out-of-sample)?** L'utente propende per sì ("descrivere
  nel macro anche l'out-of-sample"). Rischio: il PDF dice di non guardare l'OOS *"until you are done"*; vedere ora il
  rapporto 2019–2026 può orientare le scelte della fase 2 (es. un β mobile) e il criterio n.1 è la validazione onesta.
  **Raccomandazione di Claude:** fase 1 su 1972–2018 (46 anni, "over the decades" è rispettato) e gli stessi grafici
  estesi al 2026 **dopo** il lancio unico dell'OOS. Se l'utente sceglie di includerlo: fissare **prima** tutte le regole
  della fase 2 (griglia, criterio di scelta) e dichiarare nel README che i grafici descrittivi includono l'OOS e che
  nessun parametro è stato scelto da essi.
- **D2. Split della strategia:** tenere 2000-01-03 / 2018-12-31 / 2026-10-09 (raccomandato: l'OOS non è mai stato
  guardato e la data è pre-registrata; cambiarla ora sarebbe sceglierla dopo aver visto i dati). In-sample dal 1972
  sconsigliato per i 3 motivi sotto.
- **D3.** Inizio della storia: 1972-06-01 (primo giorno comune). Proposta: costante `HISTORY_START` in `config.py`.

## 4. Tutto quello che abbiamo scoperto sui dati (numeri da riportare)
**Periodo 2000–2018 (in-sample):** 4.779 giorni; 0 mancanti, 0 duplicati, 0 prezzi ≤ 0, 0 giorni scartati; 248–253
giorni/anno. Buchi più lunghi = chiusure vere: 2001-09-10→09-17 (11 settembre, 7 giorni), 2006-12-29→2007-01-03 (lutto
per Ford), 2012-10-26→10-31 (uragano Sandy). Movimenti estremi, tutti veri e tenuti (volume 2,8–7,5× la mediana):
PEP +14,9% 2002-10-08; KO +13,9% 2008-10-13 (il giorno dopo −7,5%); PEP −11,9% 2008-10-14; KO +10,3% 2008-10-28;
PEP −10,2% 2002-07-19; KO −10,1% 2002-10-16.
**Periodo 1972–1999 (contesto):** 6.970 giorni comuni (2.599 giorni solo-KO prima del 1972-06-01, scartati); buchi max
4 giorni. Estremi: KO −24,7% 1987-10-19 (Black Monday, 7,4× volume), +19,7% il 20, +13,0% il 21; PEP −14,3%
1982-12-14 (16,9× volume, livello mantenuto, causa non indagata); **PEP −11,2% 1998-08-31 e +16,1% 1998-09-01 con
volume solo 2,5–2,7×: possibile errore nei dati**, dichiarato (non corretto).
**Split** (Yahoo li applica all'indietro a prezzi, volumi e importi dei dividendi): KO 1977-06-01 2:1, 1986-07-01 3:1,
1990-05-14 2:1, 1992-05-12 2:1, 1996-05-13 2:1, **2012-08-13 2:1** (più 1965 e 1968, prima dell'inizio di PEP);
PEP 1977-05-31 3:1, 1986-05-29 3:1, 1990-09-05 3:1, 1996-05-29 2:1. Verifica fatta solo sul 2012 (KO −0,24% quel
giorno, volume mediano 13,4 M prima e 13,7 M dopo): **da estendere a ogni split**. Il movimento più grande di KO
in tutta la storia è −24,7% (Black Monday), quindi nessuno split è rimasto non gestito (uno split 2:1 darebbe −50%).
**Dividendi:** trimestrali; 2000–2018: 76 KO + 76 PEP; dividend yield KO 1,2%→3,4%, PEP 1,3%→3,2%. Prima del 2000:
KO 151 dal 1962, PEP 112 dal 1972 (da ricontare dal 1972-06-01). Adj/Close il 2000-01-03: KO 0,474, PEP 0,500.
**Record sbagliati o speciali** (ricerca: due dividendi dello stesso titolo a meno di 40 giorni; in tutta la storia
solo questi due):
- **KO 2001-09-10, 0,18 $ = duplicato** del vero 0,09 $ del 2001-09-17 (data spostata dalla chiusura per l'11/9)
  all'importo pre-split 2012: errore di gestione dello split di Yahoo. Crea un rendimento finto (+1,17% invece di
  +0,44% quel giorno) e abbassa dello 0,73% ogni prezzo aggiustato KO precedente. **Corretto** con
  `correct_dividend_errors` (raw lasciato com'è). Il controllo di ricostruzione non poteva vederlo: **coerenza ≠
  correttezza**; trovato elencando i dividendi data per data, su richiesta dell'utente di segnalarli tutti.
- **PEP 1997-10-07, 3,20 $** = lo scorporo Tricon registrato come dividendo (1 azione Tricon ogni 10 PEP; Tricon
  ≈ 31 $ nei primi giorni → ≈ 3,1 $ per azione PEP): plausibile, tenuto, segnalato.
**Ricostruzione dell'Adj Close dai dividendi di Yahoo** (2000–2018): errore massimo 0,0033% KO, 0,0001% PEP.
**KO nel 2000 era il titolo più caro:** close 28,19 (diviso per lo split 2012) = ≈ 56,4 $ reali contro 36,9 $ di PEP.
**Crescita dei prezzi aggiustati:** 1972→2018 KO ×131, PEP ×236; 2000→2018 ×2,8 e ×4,7.
**Relazione:** correlazione dei rendimenti giornalieri 0,552 nel 2000–2018 (per anno da 0,35 nel 2000–01 a 0,6–0,8
dopo il 2007, mai sotto 0,3) e 0,50 nel 1972–1999 (per anno 0,25–0,78). Rapporto KO/PEP a **onde di circa dieci
anni**: ≈ 1,0 nel 1974, minimo 0,31 nel 1982, massimo 1,17 ad agosto 1998, 0,73–0,9 nel 2000, ≈ 0,34 nel 2006,
0,40–0,58 dopo il 2004; media 2000–2018 0,454.
**I 3 motivi verificati per stimare la strategia dal 2000** (fonti da linkare):
1. PepsiCo diventa confrontabile con KO solo a fine anni '90: ristoranti (Pizza Hut, KFC, Taco Bell) scorporati in
   Tricon il **6 ott 1997** ([SEC](https://www.sec.gov/Archives/edgar/data/0001041061/000104106198000004/0001041061-98-000004.txt));
   imbottigliatori quotati come Pepsi Bottling Group il **31 mar 1999**
   ([SEC](https://www.sec.gov/Archives/edgar/data/0001076405/000107640599000017/0001076405-99-000017.txt)).
   Dentro il campione resta la fusione con **Quaker Oats (Gatorade), 2 ago 2001**
   ([CNN](https://money.cnn.com/2001/08/02/deals/pepsico/)).
2. Prezzi in frazioni di dollaro fino al passaggio ai decimali: NYSE **29 gen 2001**, Nasdaq 9 apr 2001
   ([The Daily Record](https://thedailyrecord.com/2001/01/29/nyse-goes-totally-decimal-8212-finally/)): costi di
   transazione non confrontabili.
3. Lo spread della fase 2 viene da una regressione sui livelli: con prezzi cresciuti 131× e 236× dal 1972 una
   regressione unica sarebbe dominata dagli anni recenti.
**Non verificato (dirlo):** perché KO perde terreno nel 2000–2004 (si cita spesso la valutazione altissima di fine
anni '90); causa di PEP +14,9% 2002-10-08 e −14,3% 1982-12-14; picco di volume KO 2012-08-10 (40,6 M, 3×).

## 5. Accorgimenti tecnici (lezioni della mattina)
- Logging: `configure_logging` mette la radice a WARNING e solo il pacchetto `pairs` a INFO (kaleido intasava il notebook).
- Plotly: `pio.renderers.default = "plotly_mimetype+png"` → interattivo in VS Code **e** PNG visibile su GitHub
  (notebook da 7,4 a 1,8 MB). Ogni grafico: titolo, assi con unità, nota sulla fonte (`finish_figure`); legenda al posto
  delle annotazioni sulle linee (si sovrapponevano); `dtick=1` sugli anni.
- **Eseguire i notebook sempre con `PYTHONUTF8=1`** (`export PYTHONUTF8=1`): senza, `jupyter execute --inplace` su
  Windows li legge in cp1252 e rovina simboli e trattini (successo al 01 e al 02, corretto alle 13:45).
- Notebook generato da script (`ai/scripts/`), poi `uv run ruff format notebooks` e
  `uv run jupyter execute --inplace notebooks/01_data.ipynb`. Patch lunghe: scriverle in un file col tool Write
  (un heredoc bash con molte virgolette si è rotto).
- **Commit uno alla volta, controllando l'esito** (`if git commit …; then echo OK; else …`): alle 12:05 due commit sono
  stati rifiutati dal pre-commit (mypy: `pyproject.toml` modificato ma non in stage viene messo da parte dall'hook) e
  i loro file sono finiti nel commit successivo.
- Windows: "Windows fatal exception 0x8007000e" in pytest è rumore del faulthandler (`-p no:faulthandler`); warning zmq
  innocuo. Shell: `export PATH="$HOME/.local/bin:/c/Program Files/GitHub CLI:$PATH"`.
- Tipi: `raw: pd.DataFrame | None = yf.download(...)`; `pd.DataFrame(raw["Adj Close"][[a, b]])` per pandas-stubs;
  mypy `untyped_calls_exclude = ["nbformat"]`. Yahoo: `end` esclusivo (+1 giorno); `actions` con fuso orario →
  `tz_localize(None).normalize()`.
- File dell'AI (`CLAUDE.md`, `ai/`, `.claude/`) nel `.gitignore` fino alle 20:00: l'utente li rivede, poi si tolgono
  le tre righe e si committano (il PDF chiede di lasciarli "as-is").

## 6. Regole di lavoro (dell'utente)
- **Una fase alla volta**: prima descrivere in dettaglio (celle, grafici, numeri, perché) e aspettare l'ok; la fase si
  chiude con notebook + test + paragrafo del README. Chiarezza **per un non addetto ai lavori**.
- Piccoli passi; risposte in italiano, codice/notebook/README in inglese; linee guida del PDF in ogni file.
- Ogni correzione o indicazione dell'utente → una riga in `ai/correzioni.md` (serve alla AI usage disclosure).
- Commit e push li fa Claude, annunciandoli; conventional commits; dopo ogni push controllare la CI; `ai/STATO.md`
  aggiornato dopo ogni passo.

## 7. Fase 6: le estensioni sono OBBLIGATORIE (decisione dell'utente, 12:50: "dobbiamo entrare")
Il PDF le ammette solo dopo che le fasi 1–5 sono solide: quindi fasi 1–5 veloci e pulite, poi **tutte e quattro**,
in quest'ordine (dal più utile per la validazione al meno):
1. **Costi di transazione + walk-forward**: i costi sono già nel backtest della fase 2 (`cost_bps` dichiarato); qui si
   sostituisce lo split unico con finestre mobili (stima su N anni, test sull'anno dopo, si avanza), solo dentro
   l'in-sample per tarare, poi una volta sull'OOS.
2. **Seconda coppia INTC/AMD** (già decisa alle 11:05): stessa pipeline, storia della relazione che si rompe;
   fase 4 con short interest su **entrambe** le gambe (Nasdaq).
3. **Hedge ratio dinamico con filtro di Kalman**, confrontato con l'OLS statico della fase 2 (stesse regole, stesso OOS).
4. **Paniere di 3+ titoli** (es. KO, PEP + un terzo titolo delle bevande con storia lunga su Yahoo: da scegliere e
   motivare), spread dalla regressione su tutti.
Ogni estensione: griglia e criterio **dichiarati prima**, ogni prova nel registro SQLite, IS e OOS separati.
Per riusarle, le funzioni della fase 2 prendono la coppia/i prezzi come argomento (niente classi generiche).

## 8. Tabella di marcia (da ~12:50; tutto compresso per fare anche la fase 6)
| Entro | Cosa |
|---|---|
| 13:30 | Fase 1 rifatta (warm-up macro, riusa il codice esistente) |
| 15:15 | **Fase 2** (la più importante): spread OLS con `as_of`, z-score, backtest a due gambe con costi, registro SQLite, griglia dichiarata prima, sensibilità |
| 15:55 | Fase 3 (medium: β mobile e/o ADF mobile; il segnale va peggio nei periodi deboli?) |
| 16:20 | Fase 4 (short interest PEP, 12 mesi tutti nell'OOS: descrittiva, dopo il lancio dell'OOS) |
| 17:10 | Fase 5 (block bootstrap congiunto) + **lancio unico dell'OOS** |
| 17:40 | Est. 1: costi + walk-forward |
| 18:20 | Est. 2: INTC/AMD (con fase 4 su entrambe) |
| 18:55 | Est. 3: Kalman |
| 19:25 | Est. 4: paniere |
| 20:10 | Resoconto di mezza pagina, AI usage disclosure (da `correzioni.md`), file dell'AI committati |
| 20:30 | Ultimo push, CI verde, controllo su GitHub |
Commit e push dopo ogni passo che funziona: alle 21:00 vale l'ultimo commit su `main`, quindi ogni pezzo pushato è al sicuro.

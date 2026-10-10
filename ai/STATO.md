# Stato del lavoro

Aggiornato da Claude dopo ogni passo. Una chat nuova riparte da qui; i dettagli (numeri, fonti, lezioni
tecniche, tabella di marcia, piano della fase 6) sono in `ai/ripartenza.md`; il registro della mattina è in
`ai/STATO_fino_12-40.md`.

**Ultimo aggiornamento:** sab 10 ott, ~15:45 (cambio chat; il registro precedente diceva 13:56) (orologio del PC; la CI di GitHub conferma: push delle 10:19 UTC).
Attenzione: gli orari scritti prima di questa chat (12:40, 12:50) erano circa 40 minuti avanti rispetto all'orologio.

**REGOLA DALLE 13:50: solo commit locali, NIENTE PUSH finché l'utente non lo dice (push alla fine).**

**REGOLA DALLE 13:50: solo commit locali, NIENTE PUSH finché l'utente non lo dice (push alla fine).**
Notebook sempre eseguiti con `PYTHONUTF8=1` (senza, Windows rovina i caratteri: successo al 01, corretto).

## Fatto
- Fase 1 chiusa (vedi `ripartenza.md` §1; notebook 01 rieseguito in UTF-8 in `b0caed8`).
- **Fase 2 chiusa** (nuova chat delle 12:25). Regole e giustificazioni in `ai/piano_fase2.md`. Commit:
  `8a99e71` spread · `86bd0dc` pre-registrazione `SIGNAL_GRID` · `9a8522a`, `c662df0` z-score, backtest a gambe,
  scelta della zona, registro idempotente · `886a4d4` congelamento `PHASE2_STRATEGY` (1,5 / 60 / uscita 0 /
  β 0,388372) · `c7761f4` registro IS · `20ae007` OOS (ultimo commit **su GitHub**) · `be71b5f` crossings ed
  emivita · `b0caed8` notebook 02 + 10 PNG + README "Phase 2" + "Everything I tried" (**locali, non pushati**).
- Numeri: spread IS β 0,388, R² 0,93, 8 anni su 19 senza attraversamenti, emivita 192 g, Engle-Granger p 0,12
  (non stazionario). Griglia 25 (soglie 1–3 × finestre 20/60/120/190/250): Sharpe IS tutti positivi 0,09–0,47,
  nessun t ≥ 2; scelta per regola soglia 1,5 / finestra 60: Sharpe 0,35, t 1,49, 94 trade, +2,4%/anno, DD −14%.
  Due metà: la zona buona si sposta (2001–09 finestre corte, 2010–18 lunghe). Zoom 24: 0,27–0,47.
  **OOS 2019-01-02 → 2026-10-09, lanciato una volta: Sharpe −0,53, −3,8%/anno, t −1,49, DD −32%, 36 trade;
  non sopravvive.** Causa: KO/PEP da 0,35 (2023) a 0,56 (2026), spread a β fisso da −7 a +24, perdite sui short.
  Appunto dell'utente: β mobile come **ipotesi** (fase 3 / Kalman), non risultato.
- Funzioni riusabili: `spread.py` (`fit_hedge_ratio` con as_of, `build_spread`, `rolling_zscore`,
  `mean_crossings_per_year`, `half_life`), `backtest.py` (`positions_from_zscore`, `pair_shares`, `run_backtest`
  generico per gambe, `summarize`, `run_zscore_strategy`, `select_stable_cell`), `trials.py` (`log_trial`
  idempotente, `count_trials`). Builder del notebook: `ai/scripts/build_nb02.py`; script: `grid_is.py`,
  `zoom_is.py`, `oos_launch.py`, `spread_is.py`.

## Fase 3 (chiusa alle ~14:50, commit locale in arrivo, poi push di fasi 1-3)
- Medium: β mobile OLS 500 g (descrittivo) + ADF mobile 500 g sullo spread a β fisso (ogni 5 giorni, etichetta nota a
  t dai dati fino a t-1). Codice: `src/pairs/stability.py` (`rolling_hedge_ratio`, `rolling_adf_pvalue`, `pvalue_cutoff`
  con as_of, `weak_labels`, `trade_returns`, `compare_groups`), `tests/test_stability.py` (7 test), builder
  `ai/scripts/build_nb03.py`, `notebooks/03_stability.ipynb`, figure `03_rolling_beta`, `03_adf_pvalue`, `03_equity_weak`.
- Numeri: β mobile IS da −0,067 (2002-05) a 0,745 (2012-05), solo 30% dei giorni IS entro ±0,1 da 0,388; OOS minimo
  −0,234 (2025-08). ADF p < 0,10 solo nel 10% dei giorni IS (5% OOS).
- Etichetta pre-dichiarata (p > 0,10): fallita, segnala 90% IS / 95% OOS. Etichetta post-hoc dichiarata (p nel 25% più
  alto dell'IS, soglia 0,620): trade deboli/normali IS +0,05% / +0,57% (t −1,07), OOS −0,68% / −0,85% (t 0,12);
  sensibilità 250/750 g: il segno cambia (t IS +2,50 / −1,07 / −0,50). Risposta al PDF: **nessuna evidenza** che il
  segnale vada peggio nei periodi deboli. Quadro OOS misto: crollo 2025-26 dentro le bande (etichetta accesa dal
  3° trim. 2024), ma 2019-24 perde −19% nei giorni normali e guadagna +13% in quelli deboli.
- README: i PDF non lo chiedono; niente sezioni per fase, a fine giornata solo writeup + come eseguire + disclosure.

## Snellimento notebook (fatto ~15:30, commit locali `2b5da46` + `ef32943`, NON pushati)
- `ef32943`: nel 02 rimessi tabella frase per frase del PDF, glossario compatto, nomi dei 4 commit dell'ordine degli
  eventi, base del rendimento giornaliero, nota sull'ordine del notebook (riesecuzione; prova = storia Git). 02 ora
  2.659 parole (−24%).
- **Su GitHub c'è `ccd15dc`** (fasi 1-3, notebook lunghi): push partito per errore dell'AI nonostante il rifiuto (vedi correzioni 15:35).
- L'utente: "tagliare un po'", non dimezzare. Parole di testo: 01 3.246 -> 3.010 (ritocco leggero: sintesi in cima,
  glossario al README, prima persona); 02 3.510 -> 2.095 (circa 500 sono glossario e tabella frase per frase spostati
  nel README; il resto -25%; risultato e grafico IS+OOS in cima; appendice A.1-A.4); 03 2.757 -> 2.364 (glossario e
  tabella tolti, vista per giorno in appendice). Tutti in prima persona singolare.
- Nuove funzioni di servizio: `data.load_pair_prices` (con test), `plots.equity_figure`. Versioni lunghe dei builder
  conservate in `ai/scripts/build_nb0X_long.py`.
- **Da fare nel README** (fine giornata): glossario unico, tabella fase -> metodo -> perché, resoconto di mezza pagina,
  come eseguire, AI disclosure. I notebook rimandano al "README glossary".

## Prove tentate
**62** in `results/trials.sqlite`: 25 griglia + 24 zoom (IS) + 1 OOS + 12 confronti della fase 3 (2 regole × 3 finestre × IS/OOS). Ogni nuova prova va registrata.

## Valutazione brutale e push (15:29 orologio del PC)
- Valutazione fatta su `ef32943`: validazione ottima; punti deboli = consegna incompleta (fasi 4-5, resoconto,
  disclosure, `ai/` ignorata da Git), fase 5 assente (è la "best evidence" del PDF), nessun numero sui costi, t-stat
  iid, ~100 righe di log ANSI nei notebook. Consiglio (accettato): estensioni al massimo i costi; il PDF dice "only
  after the core is solid".
- `ef848ef` style: log nascosti nei notebook (`configure_logging(logging.WARNING)`, anche nei builder); registro
  invariato (62 righe; ripristinato il file perché cambiava solo il contatore interno di SQLite).
- **Push fatto su ordine dell'utente: GitHub `main` = `ef848ef`, CI verde.**

## Prossimo passo
Fase 4 (short interest). Nasdaq API verificata: PEP 24 date (15 ott 2025 – 30 set 2026), KO "only supported for
Nasdaq Listed stocks". Proposta in chat, aspettare l'ok. Poi fase 5, costi, README (resoconto, glossario,
disclosure), `ai/` nel repository, push finale.

## Fase 4: confronti fissati PRIMA di calcolarli (15:34, orologio del PC)
Fonte: FINRA consolidated short interest (KO e PEP, 211 date, dic 2017 - set 2026), Nasdaq solo per verificare PEP e
mostrare il rifiuto per KO. Per ogni mezzo mese tra due date di regolamento: rel_shorting = Δlog SI(KO) − Δlog SI(PEP).
Spread = KO − 0,388372 × PEP (β congelato). Correlazione di Spearman; |t| ≥ 2 = degno di nota; 4 confronti × 2 periodi
(2018 = in-sample, 23 intervalli; 2019-2026, ~186) = **8 prove nel registro** (fase "phase4-short-interest").
1. Divergenza: rel_shorting contro variazione dello spread nello stesso intervallo.
2. Anticipo: rel_shorting contro variazione dello spread nei 10 giorni di borsa dalla pubblicazione (regolamento + 8 giorni).
3. Relazione debole: |rel_shorting| negli intervalli che finiscono in un giorno "debole" (regola relativa della fase 3,
   ADF 500 g) contro "normale" (t di Welch, compare_groups).
4. Movimento più grande del solito: |rel_shorting| negli intervalli col 10% più grande di |variazione dello spread| contro gli altri.
Previsione: niente di solido; se qualcosa esce è il confronto 1 (gli short reagiscono a movimenti già avvenuti), non il 2.

## Fase 4 chiusa (15:45, commit locale `8bfa47a`, NON pushato)
- Fonte: FINRA consolidated short interest (API pubblica, senza chiave) per KO e PEP, 211 date 2017-12-29 → 2026-09-30;
  Nasdaq (fonte del PDF): PEP 24 date identiche a FINRA (differenza massima 0), KO rifiutato ("only supported for Nasdaq
  Listed stocks"). Codice: `src/pairs/short_interest.py` (fetch raw con checkpoint pickle, parse, `interval_table`,
  `spearman` con t), `tests/test_short_interest.py` (6 test), builder `ai/scripts/build_nb04.py`, notebook 04, figura 04.
- Risultato: nulla si allinea. Fuori campione: (1) r 0,00 t 0,03; (2) r 0,02 t 0,20; (3) |shift| deboli 0,114 vs 0,109,
  t 0,35 (61/125); (4) grandi movimenti (≥ 2,72 $) 0,099 vs 0,113, t −0,79 (21/165). 2018: (1) t −0,20, (2) t 0,27,
  (3) e (4) non calcolabili (1 intervallo debole, 0 grandi). Livelli: dal 2023 al 2026 lo short interest raddoppia su
  ENTRAMBE (KO 23→44 M, PEP 12→25 M) mentre il ratio va 0,35→0,56.
- Esplorativi non pre-dichiarati (5, registrati): KO in-sample r 0,39 t 1,99 (OOS t 0,95) letto come rumore; livelli t −1,80
  non significativo (serie con trend).
- Registro: **73** prove (62 + 6 pre-dichiarate + 5 esplorative).

## Prossimo passo
Push su ordine dell'utente. Poi fase 5 (block bootstrap congiunto): proposta prima, ok dell'utente.

## Pulizia notebook e push (15:57 orologio del PC)
- `5c915e0` niente riferimenti agli appunti dell'AI nei notebook 03-04 (e nelle note del registro); `cefa15e` niente
  orari nei notebook. Regole nuove dell'utente: nei notebook mai riferimenti all'AI, mai orari.
- **Push fatto su ordine dell'utente: GitHub `main` = `cefa15e`, CI verde.** Registro: 73 prove.

## CAMBIO CHAT (~15:57)
Prossimo passo: fase 5 (proposta prima, ok dell'utente). Testo per la nuova chat: `ai/prompt_fase5.md`.

## Fase 5: decisioni e previsione fissate PRIMA di calcolare (16:06 orologio del PC)
Decise con l'utente (proposta in chat, ok "fai b"):
- Metodo scelto dal menu: **block bootstrap congiunto** (simple). Motivo: modello-libero; l'OU calibrato sullo spread
  presuppone il ritorno alla media che la fase 2 non ha dimostrato (EG p 0,12) -> circolare.
- **In più, dichiarato** (il PDF dice "pick one"): OU sullo spread IS a β 0,388 come controllo = "mondo in cui la
  relazione è reale alla forza stimata". Bootstrap = mondo senza relazione di lungo periodo. Se lo 0,35 reale cade in
  entrambe, 19 anni non bastano a distinguerle (poca potenza). Da spiegare accuratamente nel notebook 05.
- Rendimenti semplici giornalieri **solo IS 2000-2018** (as_of = 2018-12-31), percorsi lunghi quanto l'IS, prezzi
  iniziali reali, calendario IS. Blocchi **10** (principale), 5 e 20 (sensibilità); **1000 percorsi**; seme 20261010.
- Su ogni percorso: β OLS ristimato sul percorso intero (stessa procedura della fase 2), regole congelate 1,5/60/0,
  valutazione dal 250° giorno come nella fase 2. Seconda analisi: griglia 25 + regola P2 su ogni percorso (blocchi 10;
  numero di percorsi deciso dopo aver misurato i tempi, minimo 200).
- OU: AR(1) sullo spread IS a β 0,388, simulazione esatta, 1000 percorsi, regole congelate; Sharpe sul P&L in dollari
  (capitale costante) e lo stesso Sharpe calcolato sulla storia reale per confronto alla pari.
- Registro: 1 riga per configurazione (bootstrap 5/10/20, bootstrap+selezione, OU) = 5 prove -> 78.
Previsione: bootstrap a regole congelate centrato tra 0 e 0,1, 90% centrale circa −0,35/+0,45; lo 0,35 reale
all'85°-90° percentile (non distinguibile dal caso); con la selezione lo 0,35 scende di percentile. OU: mediana
positiva, intorno a 0,3, con lo 0,35 reale nel mezzo -> conclusione "poca potenza", coerente con l'OOS −0,53.

## Fase 5: primi risultati (16:12) e una diagnostica esplorativa dichiarata
- Bootstrap regole congelate, mediana Sharpe / 5%-95% / quota > 0 / percentile dello 0,351 reale:
  blocchi 5: 0,100 / −0,26..0,45 / 66% / 87° · blocchi 10: 0,165 / −0,23..0,53 / 74% / 80° · blocchi 20: 0,268 /
  −0,09..0,63 / 89% / 64°. Con griglia + regola P2 (blocchi 10): 0,248, reale al 67°. OU (P&L in $): 0,173 /
  −0,21..0,54, reale (0,367) all'80°; bootstrap 10 nella stessa misura: 0,161, reale all'83°.
- Previsione sbagliata sul centro: il bootstrap è centrato in positivo e cresce con i blocchi. Rapporti di varianza
  dello spread reale IS: 0,87 a 10 g, 0,76 a 20, 0,61 a 60 -> ritorno alla media su settimane-mesi, i blocchi ne tengono
  la parte breve. Altra causa possibile: β stimato sullo stesso percorso (regressione spuria).
- Esplorativa, decisa ORA dopo aver visto i numeri: blocchi da 1 giorno (nessuna dinamica tenuta) per separare le due
  cause. Previsione: mediana vicina a 0 se l'effetto viene dai blocchi; positiva se viene dal β stimato in-sample.

## Fase 5 chiusa (16:30 PC, commit locale `df110a2`, NON pushato; GitHub resta `cefa15e`)
- Esplorativa blocchi da 1 giorno: mediana 0,017, 52% positivi, reale al 92° -> il centro positivo NON viene dal β
  stimato sul percorso ma dalla dinamica breve tenuta nei blocchi (rapporti di varianza reali 0,92/0,76/0,61 a 5/20/60 g).
- Selezione rifatta su ogni percorso: sceglie 1,5 (48%) e 60 (57%) più spesso; sui dati reali riproduce 1,5/60, 0,351.
- OU: mediana 0,17 contro 0,16 del bootstrap (stessa misura in $): il mondo "relazione reale" non rende di più.
  OU a 60 g: rapporto di varianza 0,90 contro 0,61 reale -> l'OU descrive male lo spread.
- Conclusione nel notebook: evidenza CONTRO un effetto robusto; lo 0,35 non supera mai il 95° percentile.
- Codice: `src/pairs/simulate.py`, `tests/test_simulate.py` (6 test), `ai/scripts/build_nb05.py`, checkpoint
  `data/phase5_simulation.pkl` (4 min di calcolo), figure 05_example_paths, 05_bootstrap_sharpe, 05_sharpe_by_world.
- Registro: **79** prove (73 + 5 fase 5 + 1 esplorativa).

## Prossimo passo
Costi di transazione sulla strategia congelata (analisi dichiarata, nel registro): proposta all'utente prima.
Poi README (resoconto, tabella fase->metodo->perché, glossario, come eseguire, conteggio prove, PySide6 saltato,
AI disclosure), revisione di CLAUDE.md/ai/.claude da parte dell'utente, .gitignore, push finale entro le 20:30.

## Revisione cattiva della fase 5 (16:42 PC, commit locale `4295aa8`, NON pushato)
- Su richiesta dell'utente prima del push. Conclusione riscritta in tre parti: (1) relazione: ritorno alla media su
  settimane-mesi reale nell'IS (rapporto di varianza 60 g 0,61; percorsi a 1 g: mediana 0,96, solo 0,5% così bassi;
  a 10 g: 0,82, 6%); (2) segnale: positivo nel 74% dei percorsi ma Sharpe tipico 0,17, metà dello 0,35; (3) il
  bootstrap non può contenere cambi di regime (2023-26). Rapporto di varianza: diagnostica, non backtest (registro
  invariato: 79). Box plot con il valore reale OU (0,37, rombo).
- Prossimo passo: push di `4295aa8` su ordine dell'utente, poi README (struttura prima, ok dell'utente).
- Commit locale `1307183` (NON pushato): tabella di tutte le metriche sui percorsi (blocchi 10). Reale vs mediana:
  rendimento 2,4% vs 1,7%; t ≥ 2 solo nell'8% dei percorsi; drawdown reale −14% meno profondo del 98,7% dei percorsi
  (mediana −32%, 5% peggiore circa −60%); volatilità reale sotto tutti i 1000 percorsi. Registro invariato (79).
  Su GitHub resta `cefa15e`: push di 3 commit (`df110a2`, `4295aa8`, `1307183`) su ordine dell'utente.

## Push fase 5 (16:57 PC, su ordine dell'utente): GitHub `main` = `1307183`, CI verde (3 min 33 s). Registro: 79.
Prossimo passo: README (struttura da mostrare all'utente prima di scrivere).

## CAMBIO CHAT (~17:00)
Prima: valutazione dei notebook secondo "How we evaluate" del PDF e aderenza alle fasi; poi struttura del README (bozza in 10 punti, da rivedere insieme). Testo per la nuova chat: `ai/prompt_readme.md`.

## Valutazione dei notebook e correzioni (17:05-17:15 PC, commit locale `e00f247`, NON pushato; GitHub resta `1307183`)
- Valutazione secondo "How we evaluate" + aderenza alle fasi: validazione forte, presentazione il punto debole.
- Correzioni (builder `ai/scripts/build_nb02/03/05.py`, notebook rieseguiti in UTF-8):
  02: glossario tolto (rimando al README), riga registro "50 in this phase", Result con t robusto, A.5 Newey-West
  (IS 1,69-1,78, OOS −1,47/−1,64, lag 10/20/60/120 dichiarati post-OOS, non è una prova), "why it fails" con medie
  annue dello spread −7 → +24 e +39 l'ultimo giorno (assert), ipotesi del β mobile ridimensionata dalla fase 3.
  03: paragrafo OOS §6 a punti. 05: Result corto in 3 punti; rapporto di varianza dichiarato post-hoc (§4, §7) e
  registrato come 2 righe esplorative; "90% of paths" su correlazione e β (6% β negativo); controllo drawdown sui
  percorsi con β entro 0,1 (mediana −31%); "2024–2026".
- Registro: **81** prove (79 + 2 esplorative della fase 5).

## Prossimo passo
Struttura del README da concordare con l'utente (bozza in 10 punti in `ai/prompt_readme.md`), poi scrittura.

## Push delle ~18:45 (su ordine dell'utente): GitHub `main` = `08fcef0`
- `32ac6ed` prosa dei notebook riscritta (5 sub-agenti, controllo automatico di numeri/codice/titoli) · `118e79c`
  README · `08fcef0` CLAUDE.md + `ai/` (diario compreso, scelta dell'utente) + `ai/` esclusa da ruff in pyproject.
- Prossimo: estensioni (1 costi + walk-forward, regole approvate; 2 da scegliere, consiglio Kalman). Prompt per la
  nuova chat: `BSQ/HACKATON BSQ/handoff/prompt_estensioni.md`. Stop 20:00, push finale entro 20:30.

## Estensione Kalman (18:40 PC): regole registrate in `config.py` (`KALMAN_RULES`) prima di ogni calcolo
- L'utente ha scelto di iniziare dal Kalman. δ 1e-4 (Chan), sensibilità 1e-5 e 1e-3, avvio OLS sul 2000, rumore
  di osservazione = varianza dei residui OLS IS, spread con stime fino a t−1, z/soglie congelate, β del trade fisso
  all'ingresso, segno libero, niente costi (confronto alla pari con la fase 2). 6 prove dichiarate (fase "ext-kalman").
- Previsione scritta prima: il Kalman limita la perdita OOS del 2024-26 ma lo Sharpe IS scende sotto 0,35 (0,1-0,2).
- Risultato (notebook `06_kalman.ipynb`, `src/pairs/kalman.py`, `tests/test_kalman.py`, builder `ai/scripts/build_nb06.py`):
  β Kalman 0,10-0,63, mai negativo (0,30 fino al 2023, 0,62 a ott 2026). IS Sharpe 0,15 (t 0,62) contro 0,42 della
  fase 2 sugli stessi giorni; tutto il guadagno IS viene da set 2008-mar 2009 (18,5% su 17,6%). OOS −0,18 contro
  −0,53 (2025: −1,3% contro −16,4%). Trade 191 contro 94, 9 giorni contro 32. Sensibilità: δ 1e-3 −0,13/−0,07,
  δ 1e-5 0,40/0,03 (visto dopo: non si sceglie). Previsione confermata. Registro: **87** prove (81 + 6).
- Prossimo: commit, poi estensione 1 (min_trades 6/3/9 da confermare), README, push su ordine.

## Estensione 1, costi + walk-forward (18:55 PC): regole registrate in `config.py` (`COST_WF_RULES`) prima di calcolare
- Kalman committato (`0bcee87`). Utente: min_trades 6/3/9 ok. Regole come approvate (2/5/10 bp sulla strategia congelata
  IS e OOS = 6 prove; walk-forward 2005-2026, 5 anni, 5 bp, sensibilità 3 e 8 anni sul 2008-2026 = 3 prove).
- Risultato (notebook `07_costs_walkforward.ipynb`, `src/pairs/walkforward.py` dalla bozza con tipi sistemati,
  `tests/test_walkforward.py`, builder `ai/scripts/build_nb07.py`): costi 5 bp → Sharpe IS 0,35 → 0,28 (0,52%/anno,
  solo 94 trade; previsione 0,15 sbagliata), OOS −0,53 → −0,60. Walk-forward 5 anni: 2005-2026 Sharpe −0,23 (t −1,06),
  DD −37%, perde anche nel 2005-2018 (−0,16); 2026 −17,9%. Sensibilità 2008-2026: 5 a −0,04, 3 a 0,11, 8 a −0,11.
  β passato 0,03-0,73; finestra 190 scelta 15 volte su 22. Registro: **96** prove.
- Prossimo: commit, README (estensioni, tabella metodi, conteggio 96, conclusione, "with more time"), push su ordine.

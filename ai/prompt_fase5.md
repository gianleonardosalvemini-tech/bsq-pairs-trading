Hackathon BSQ, cambio chat delle ~15:57 (orologio del PC). Riparti esattamente da qui.

Codice: C:\Users\gianl\Documents\bsq-pairs-trading. Leggi SUBITO, in quest'ordine:
1. ai/STATO.md (fasi 1-4 chiuse, valutazione brutale delle 15:29, registro a 73 prove, prossimo passo)
2. ai/correzioni.md (ogni mia correzione e ogni errore dell'AI: serve per la AI disclosure)
3. ai/piano_fase2.md (perché la strategia è congelata) e ai/regole_oro.md
4. ai/ripartenza.md §5 (accorgimenti tecnici: notebook con PYTHONUTF8=1, commit uno alla volta)
PDF originali: HACKATON BSQ/consegna progetto/ (tema e linee guida): rileggi TU la fase 5 alla lettera.

STATO DI GIT: locale = GitHub `main` = `cefa15e`, CI verde. Il prossimo push SOLO quando lo dico io, mai nella
stessa chiamata di altri comandi.

FASE 5 ("Simulating the pair", PDF p.5). Prima di scrivere codice: cosa chiede la fase frase per frase, cosa proponi
(un metodo dal menu, con il perché), poi aspetti il mio ok. Il PDF chiede: simulare le due serie INSIEME preservando
come si muovono insieme; simple = joint block bootstrap (stessi blocchi di 5-20 giorni per entrambe, con
reinserimento); medium = Ornstein-Uhlenbeck sullo spread; rieseguire il segnale della fase 2 su ogni percorso e
"report how your signal's performance varies across the simulated paths ... your best evidence for or against the
relationship being a real, robust effect rather than a fit to one specific historical accident".
Punti da decidere con me (proponi tu, io scelgo):
- Da quali rendimenti ricampionare: solo in-sample 2000-2018 (raccomandato: l'OOS non si tocca) o tutto.
- Lunghezza dei blocchi (es. 10 giorni, con 5 e 20 come sensibilità), numero di percorsi, seme fisso.
- Sul percorso: β ristimato con l'OLS come nella fase 2 (stesse regole: soglia 1,5, finestra 60, uscita 0) oppure β
  congelato 0,388. E se rieseguire anche la scelta sulla griglia di 25 su ogni percorso (tiene conto della selezione).
- Lettura chiave da spiegare: blocchi di 5-20 giorni conservano la correlazione giornaliera ma distruggono il
  ritorno alla media su scale di mesi (emivita 192 g): i percorsi sono quasi un "mondo senza relazione di lungo
  periodo". Se lo Sharpe in-sample 0,35 cade dentro la distribuzione simulata, l'edge in-sample è
  indistinguibile dal caso.
- Confronti e previsione scritti PRIMA di calcolarli; ogni configurazione nel registro results/trials.sqlite.

REGOLE GIÀ DECISE (non rimetterle in discussione):
- Commit locali uno alla volta, annunciati, controllando l'esito (il pre-commit esegue ruff, mypy, tutti i test e i
  notebook). Push solo su mio ordine esplicito.
- Strategia della fase 2 CONGELATA (PHASE2_STRATEGY: soglia 1,5, finestra 60, uscita 0, β 0,388372); OOS lanciato una
  volta (Sharpe −0,53). Tutto ciò che riguarda il 2019-2026 è analisi post-lancio, da dichiarare.
- Il β mobile / Kalman è un'IPOTESI, non un risultato.
- Notebook: in INGLESE, PRIMA PERSONA SINGOLARE ("I"), risultato in cima, ragionamento per ogni scelta e ogni numero,
  dettagli in appendice, `assert` su ogni numero del testo, verificare ogni frase con i dati prima di scriverla.
  NIENTE riferimenti all'uso dell'AI (ai/, chat, appunti) e NIENTE orari nei notebook. Log nascosti:
  `configure_logging(logging.WARNING)`. Notebook generati da ai/scripts/build_nb0X.py, poi ruff format, poi
  `jupyter execute --inplace` con PYTHONUTF8=1.
- Riusa: spread.py (fit_hedge_ratio con as_of, build_spread, rolling_zscore), backtest.py (run_zscore_strategy,
  summarize, select_stable_cell), trials.py (log_trial idempotente, count_trials), data.load_pair_prices,
  plots (finish_figure, save_figure, equity_figure). Funzioni nuove in src/pairs/simulate.py con `as_of: date`
  (filtro <= as_of come prima operazione, con assert), docstring NumPy con Notes, test pytest.
- Ogni mia correzione in ai/correzioni.md; ai/STATO.md aggiornato dopo ogni passo.

COME VOGLIO CHE LAVORI: molto critico con me e mai apprensivo; il mio obiettivo è capire se la strategia è davvero
vincente; chiarezza per un non addetto ai lavori.

DOPO LA FASE 5 (tempo stimato ~1 h): costi di transazione sulla strategia congelata come analisi dichiarata (~15 min,
nel registro); README: resoconto di mezza pagina, tabella fase -> metodo -> perché, glossario unico (i notebook vi
rimandano), come eseguire, conteggio di tutte le prove, PySide6 saltato dichiarato, AI usage disclosure (da
ai/correzioni.md); togliere CLAUDE.md, ai/, .claude/ dal .gitignore DOPO che li ho rivisti io; push finale entro le
20:30, CI verde, controllo su GitHub. Estensioni della fase 6: al massimo i costi.

PARTIAMO: in 3 righe dimmi dove siamo, poi la proposta della fase 5 frase per frase dal PDF, e aspetta il mio ok.

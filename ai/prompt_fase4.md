Hackathon BSQ, cambio chat delle ~15:45 (ora del PC). Riparti esattamente da qui.

Codice: C:\Users\gianl\Documents\bsq-pairs-trading. Leggi SUBITO, in quest'ordine:
1. ai/STATO.md (fasi 1, 2, 3 chiuse; snellimento dei notebook; stato di Git; prossimo passo)
2. ai/correzioni.md (ogni mia correzione e ogni errore dell'AI: serve anche per la AI disclosure)
3. ai/piano_fase2.md (perché la strategia è congelata) e ai/regole_oro.md
4. ai/ripartenza.md §5 (accorgimenti tecnici: notebook sempre con PYTHONUTF8=1, commit uno alla volta)
PDF originali: HACKATON BSQ/consegna progetto/ (tema e linee guida): rileggili TU, alla lettera.

STATO DI GIT (attenzione):
- HEAD locale = `ef32943` (2 commit NON pushati: `2b5da46` snellimento notebook, `ef32943` ripristino spiegazioni del 02).
- GitHub `main` = `ccd15dc` (fase 3 con i notebook lunghi): push partito per errore dell'AI nonostante il mio rifiuto.
  Non fare force push. Il prossimo push si fa SOLO quando lo dico io.

PRIMA COSA DA FARE (prima di qualunque codice): una VALUTAZIONE BRUTALE, senza apprensione, del lavoro sul commit
LOCALE `ef32943` (notebook 01, 02, 03 eseguiti, src/, tests/, results/), NON sulla versione su GitHub. Dimmi:
1) quanto vale il lavoro e dove è debole; 2) se è meticolosamente attinente alla consegna, fase per fase e frase per
frase del PDF (compresi "What to deliver" e le linee guida di programmazione), elencando cosa manca o è fuori traccia;
3) se il notebook 02 (dove si concentra la valutazione) contiene tutti i ragionamenti e le giustificazioni, e se i
notebook sono leggibili da uno sconosciuto; 4) i rischi principali per entrare nei 7 su 14 e come ridurli col tempo
che resta (consegna: ultimo commit su main entro le 21:00, obiettivo 20:30). Poi aspetta il mio ok.

REGOLE GIÀ DECISE (non rimetterle in discussione):
- Commit locali uno alla volta, annunciati, controllando l'esito (il pre-commit esegue ruff, mypy, tutti i test e i
  notebook). Push solo su mio ordine esplicito, mai nella stessa chiamata di altri comandi.
- Strategia della fase 2 CONGELATA (PHASE2_STRATEGY: soglia 1,5, finestra 60, uscita 0, β 0,388372); OOS già lanciato
  una volta (Sharpe −0,53). Tutto ciò che riguarda il 2019-2026 è analisi post-lancio, da dichiarare. Ogni nuovo
  backtest o confronto va nel registro results/trials.sqlite (oggi 62 righe).
- Il β mobile / Kalman è un'IPOTESI, non un risultato.
- Compartimenti stagni: una fase alla volta. Prima di scrivere codice: cosa chiede la fase (frase per frase dal PDF),
  cosa propongo (un metodo dal menu, con il perché), poi aspetti il mio ok.
- Notebook: in INGLESE, in PRIMA PERSONA SINGOLARE ("I", mai "we"), risultato in cima, ragionamento per ogni scelta e
  ogni numero, dettagli in appendice, `assert` su ogni numero del testo. Snelli ma completi: tagliare "un po'", mai a
  scapito delle giustificazioni. Verificare ogni frase con i dati prima di scriverla.
- Il README non è chiesto dai PDF: conterrà a fine giornata il resoconto di mezza pagina, la tabella fase -> metodo ->
  perché, il glossario unico (i notebook 01 e 03 già rimandano al "README glossary"), come eseguire, la AI disclosure
  (da ai/correzioni.md).
- Riusa le funzioni esistenti (spread.py, backtest.py, trials.py, stability.py, data.load_pair_prices,
  plots.equity_figure). Ogni mia correzione in ai/correzioni.md; ai/STATO.md aggiornato dopo ogni passo.

COME VOGLIO CHE LAVORI: molto critico con me e mai apprensivo; il mio obiettivo è capire se la strategia è davvero
vincente; chiarezza per un non addetto ai lavori.

DOPO LA VALUTAZIONE: push (se lo dico io), poi fase 4 (short interest; solo PEP è su Nasdaq: dichiararlo), poi fase 5
(simulazione: "Report how your signal's performance varies across the simulated paths ... your best evidence for or
against the relationship being a real, robust effect"), poi resoconto, disclosure e push finale.

PARTIAMO: in 3 righe dimmi dove siamo, poi fai la valutazione brutale sul commit locale `ef32943` e aspetta il mio ok.

Hackathon BSQ, cambio chat delle ~17:00 (orologio del PC). Riparti esattamente da qui.

Codice: C:\Users\gianl\Documents\bsq-pairs-trading. Leggi SUBITO, in quest'ordine:
1. ai/STATO.md (fasi 1-5 chiuse; in fondo: fase 5, revisione cattiva, tabella metriche, push delle 16:57)
2. ai/correzioni.md (ogni mia correzione e ogni errore dell'AI: è la base della AI usage disclosure)
3. ai/regole_oro.md (caselle da spuntare) e ai/ripartenza.md §5 (accorgimenti tecnici)
4. README.md attuale (848 parole, fermo alla mattina: va riscritto) e i 5 notebook (solo il "Result" in cima a ciascuno)
PDF originali: HACKATON BSQ/consegna progetto/ (tema e linee guida): rileggi tu cosa chiedono per le consegne.

STATO DI GIT: locale = GitHub `main` = `1307183`, CI verde. Registro results/trials.sqlite: 79 prove.
Il prossimo push SOLO quando lo dico io, mai nella stessa chiamata di altri comandi.

FASE 5 CHIUSA (conclusione in tre parti): (1) il ritorno alla media dello spread su settimane-mesi è reale
nell'in-sample (rapporto di varianza 60 g 0,61; percorsi senza dinamica: mediana 0,96, solo 0,5% così bassi);
(2) il segnale è positivo nel 74% dei percorsi ma lo Sharpe tipico è 0,17, metà dello 0,35; drawdown reale −14%
meno profondo del 98,7% dei percorsi (mediana −32%); (3) il bootstrap non può contenere cambi di regime (2023-26).
Metodo: block bootstrap congiunto; OU come controllo in più dichiarato (il PDF dice "pick one").

PRIMA DI TUTTO (prima del README): una VALUTAZIONE dei 5 notebook seguendo la sezione "How we evaluate" del PDF
(rileggila alla lettera e usa i suoi criteri come griglia) e l'ADERENZA di ciascun notebook alla propria fase,
frase per frase rispetto al testo del PDF. Schietta e cattiva: per ogni notebook cosa regge, cosa manca, cosa è
fuori traccia, cosa correggere (con priorità e tempo stimato). Non correggere nulla finché non decidiamo insieme.

POI il README. Bozza di struttura (ORDINE E CONTENUTI LI ANALIZZIAMO INSIEME prima di scrivere):
1. Titolo + risultato in 3 righe (KO/PEP; IS Sharpe 0,35 t 1,49; OOS −0,53; fase 5).
2. Resoconto di mezza pagina (richiesto dal PDF): fasi completate 1-5, metodo scelto e perché, cosa ho imparato,
   cosa farei con più tempo (β mobile/Kalman come ipotesi, walk-forward, seconda coppia INTC/AMD).
3. Tabella fase -> metodo -> perché -> cosa ho saltato, con link ai notebook (OU in più dichiarato; FINRA per KO).
4. Risultati chiave: tabella IS vs OOS + 2-3 figure già salvate in results/figures.
5. Everything I tried: 79 prove per fase dal registro (query SQLite, esplorative separate) e perché conta.
6. Validazione: split fissato in un commit prima dei prezzi, regole pre-registrate, OOS lanciato una volta,
   as_of, assert su ogni numero.
7. Glossario unico (i notebook vi rimandano): spread, hedge ratio, z-score, Sharpe, t-stat, drawdown, emivita,
   ADF/Engle-Granger, short interest, days to cover, block bootstrap, OU, rapporto di varianza, in/out-of-sample.
8. How to run: uv sync, notebook (PYTHONUTF8=1 su Windows), pytest, checkpoint data/phase5_simulation.pkl,
   struttura del repository.
9. Engineering standards: linee guida rispettate; PySide6 saltato e dichiarato.
10. AI usage disclosure: cosa ho chiesto, cosa ho controllato, dove l'ho corretta (da ai/correzioni.md, compresi gli
    errori dell'AI: push partito nonostante il rifiuto, numeri a memoria presi dagli assert, conclusione della fase 5
    corretta dopo la mia richiesta di revisione) + rimando alla cartella ai/ lasciata "as-is".
La vecchia sezione "Phase 1" del README sparisce (resta la riga della tabella; dettagli nel notebook 01).

DOPO IL README, IN ORDINE (anche questo lo rivediamo insieme): (a) mia revisione di CLAUDE.md, ai/, .claude/ (preparami un elenco di cosa contiene ogni file);
(b) togliere le 3 righe dal .gitignore e committarli; (c) controllo finale sulle regole d'oro; (d) push su mio
ordine, CI verde, controllo su GitHub. Solo DOPO, se c'è tempo: estensione costi + walk-forward (completa, analisi
post-OOS dichiarata), stop alle 20:00, push finale entro le 20:30. Ogni prova nel registro.

REGOLE GIÀ DECISE (non rimetterle in discussione):
- Commit locali uno alla volta, annunciati, controllando l'esito (il pre-commit esegue ruff, mypy, tutti i test e
  i notebook). Push solo su mio ordine esplicito, in una chiamata a sé.
- Strategia della fase 2 CONGELATA (soglia 1,5, finestra 60, uscita 0, β 0,388372); OOS lanciato una volta.
- README e notebook in INGLESE, PRIMA PERSONA SINGOLARE ("I"), ogni numero verificato sui dati prima di scriverlo.
  Nei notebook NIENTE riferimenti all'uso dell'AI e NIENTE orari (nel README l'AI sta solo nella disclosure).
- Patch lunghe: scriverle in un file col tool Write (gli heredoc con molte virgolette si rompono).
- Ogni mia correzione in ai/correzioni.md; ai/STATO.md aggiornato dopo ogni passo.

COME VOGLIO CHE LAVORI: molto critico con me e mai apprensivo; chiarezza per un non addetto ai lavori.

PARTIAMO: in 3 righe dimmi dove siamo, poi la valutazione dei notebook secondo "How we evaluate" e l'aderenza
alle fasi, e aspetta che decidiamo insieme cosa correggere.

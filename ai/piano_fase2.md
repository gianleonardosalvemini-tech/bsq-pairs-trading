# Fase 2: proposta da confermare con l'utente (scritta sab 10 ott, 12:25 ora del PC)

La fase che pesa di più (PDF p.4: "Validation: the part we weight most heavily"). Obiettivo dell'utente: **capire se
la strategia è vincente**, quindi validazione onesta prima di tutto. Le scelte P1–P4 sono **raccomandate ma non
ancora confermate**: chiederle all'utente in 3 righe ciascuna prima di scrivere codice.

## Regola sulle modifiche (discussa con l'utente alle 12:25)
- **Prima del lancio dell'OOS** si può cambiare tutto (griglia, criterio, costi, codice): ogni backtest finisce nel
  registro SQLite e il conteggio totale va nel README.
- **Dopo il lancio dell'OOS** la strategia non si tocca più: solo analisi in più, dichiarate come tali.
- Lo split 2000-01-03 / 2018-12-31 / 2026-10-09 non si sposta (vedi correzioni.md, 12:05).

## Codice (funzioni esplicite, riusabili per le estensioni: niente classi generiche)
- `src/pairs/spread.py`
  - `fit_hedge_ratio(prices, a, b, as_of) -> HedgeFit` (dataclass frozen: alpha, beta, as_of, n_obs): OLS
    `A = alpha + beta * B` sui **prezzi** (Adj Close corretti, come dice il PDF), **prima riga**: filtro
    `<= as_of` con `assert`. Con statsmodels (già nelle dipendenze) o numpy.
  - `build_spread(prices, a, b, beta) -> Series` = A − beta·B (beta float; per Kalman servirà una Series).
  - `rolling_zscore(spread, window) -> Series`: media e deviazione mobili su `window` giorni che finiscono oggi
    (solo passato + oggi); NaN finché la finestra non è piena.
- `src/pairs/backtest.py`
  - `positions_from_zscore(z, entry, exit=0.0) -> Series` in {-1, 0, +1}: long spread se z < −entry, short se
    z > +entry, chiude quando z torna a `exit` (attraversa lo zero). Ciclo esplicito (macchina a stati), chiaro da
    spiegare.
  - `run_backtest(prices, shares, cost_bps) -> DataFrame` **generico per gambe**: `shares` = azioni per gamba
    (coppia: +1 A, −beta B moltiplicato per la posizione; Kalman: beta variabile; paniere: n gambe). Posizione
    decisa alla chiusura di t, guadagna da t+1 (`shift(1)`). Rendimento giornaliero = PnL in dollari / capitale
    lordo del giorno prima (somma dei valori assoluti delle gambe). Costo = cost_bps × valore scambiato a ogni
    cambio di azioni. Convenzione nelle Notes della docstring.
  - `summarize(returns) -> dict`: rendimento annuo, volatilità annua, Sharpe (×√252), t-stat della media
    giornaliera, max drawdown, numero di trade, % giorni in posizione, durata media di un trade.
- `src/pairs/trials.py`: `results/trials.sqlite`, tabella `trials` (id, ora, fase, coppia, parametri JSON, periodo
  IS/OOS, metriche, nota). `log_trial(...)`, `count_trials(...)`. Ogni backtest passa da qui. Nei test e nel test
  dei notebook il database va nella cartella temporanea (`PAIRS_RESULTS_DIR`), per non gonfiare il conteggio.
- Test pytest: (1) cambiare un prezzo dopo `as_of` non cambia beta; (2) cambiare un dato futuro non cambia lo
  z-score di oggi; (3) la posizione di t si applica al rendimento di t+1; (4) PnL a due gambe su un esempio a mano;
  (5) costi addebitati solo ai cambi di posizione; (6) il registro conta le righe.

## Regole fissate PRIMA del primo backtest (commit `docs: pre-register phase 2 grid and selection rule`)
- **P1. Griglia**: soglia di ingresso {1.0, 1.5, 2.0, 2.5, 3.0} × finestra dello z-score {20, 40, 60, 90, 120}
  giorni = **25 prove**, tutte sull'in-sample. Uscita fissa a z = 0 (PDF: "close the position when it reverts
  back toward zero"), non tarata.
- **P2. Criterio di scelta**: la cella con il più alto **Sharpe netto medio su sé stessa e le celle vicine** (centro
  della zona più stabile), non la migliore isolata; a parità, meno trade. È la difesa contro il risultato fortunato.
- **P3. Costi**: 5 bps (0,05%) del valore scambiato per gamba, a ogni ingresso e uscita (prudente per due titoli
  tra i più liquidi degli USA). Altri valori nell'estensione 1.
- **P4. beta**: stimato una volta sul 2000–2018 e **congelato** per l'OOS. Da dichiarare: il backtest in-sample usa
  un beta stimato sugli stessi dati, quindi è ottimista; il beta mobile è nella fase 3.
- Dataclass in `config.py` (es. `SignalGrid`) con questi valori, validata (fallisce subito se vuota o incoerente).

## Notebook `02_signal.ipynb` (generato da `ai/scripts/build_nb02.py`, come il 01)
1. Cosa fa, glossario (spread, hedge ratio, z-score, Sharpe, drawdown, costi, look-ahead).
2. Regole pre-registrate (tabella) e commit che le fissa.
3. Hedge ratio sull'in-sample (alpha, beta, R²) e **spread verificato**: grafico IS con media, attraversamenti della
   media per anno, emivita (AR(1)), ADF come numero indicativo.
4. z-score e segnale su un periodo d'esempio (grafico con ingressi/uscite).
5. Le 25 prove: heatmap dello Sharpe netto IS (sensibilità), tabella completa, criterio applicato, variante scelta.
6. Equity curve IS della variante scelta, lorda e netta; metriche.
7. "Out-of-sample: lanciato una volta alle ~17:10" (sezione riempita solo allora; ora annotata nel registro).
8. Cosa impariamo / cosa non so; `assert` su ogni numero.
Poi paragrafo "Phase 2" del README + tabella "Everything I tried" alimentata dal registro.

## Critica onesta (da dire all'utente, senza ansia)
- Aspettativa a priori: uno z-score semplice su KO/PEP con beta statico sui prezzi probabilmente rende poco al
  netto dei costi; il 2000–2005 (in-sample) è una deriva del rapporto 0,73 → 0,33. Un risultato nullo spiegato bene
  è "in great shape" (PDF p.2). Se è troppo bello: prima cercare l'errore (look-ahead, segni, costi).
- OLS sui prezzi (come il PDF) e non sui log: con prezzi cresciuti ×2,8 e ×4,7 nell'IS la regressione pesa di più
  gli anni recenti. Da dire come limite.

## Passi e commit (uno alla volta, annunciati, CI controllata)
1. pre-registrazione (`config.py` + test) → 2. `spread.py` + test → 3. `backtest.py` + `trials.py` + test →
4. notebook 02 + PNG + README. Tabella di marcia: fase 2 entro le 15:15 (ora del PC: c'è ~40 min di margine).

## Decisione 12:41: finestre della media mobile = B {20, 60, 120, 190, 250} giorni (scelta dell'utente)
Giustificazione (da riportare nel notebook 02 e nel README):
1. Il PDF lascia libera la finestra ("lookback window for the rolling mean") e la nomina come parametro di sensibilità.
2. Misura sull'in-sample (nessun backtest): spread KO − 0,388·PEP, emivita AR(1) 192 giorni. Finestre molto più corte
   dell'emivita fanno inseguire lo spread alla media mobile (z = oscillazioni brevi); vicine all'emivita, z misura
   gli scostamenti alla scala a cui lo spread rientra davvero. La griglia copre le due ipotesi.
3. Punti: 20 (un mese), 60 e 120 (trimestre, semestre: l'intervallo che il PDF usa nella fase 3), 190 (≈ emivita),
   250 (un anno di borsa). Candidate (celle interne): 60, 120, 190.
4. Deciso prima di qualunque backtest, solo con dati IS; le prove restano 25.
5. Limiti da dichiarare: l'emivita è stimata con lo stesso β dell'IS ed è imprecisa; scegliere la griglia guardando
   l'IS è "tuning" lecito ma va detto.
Conseguenza: tutte le celle si confrontano sullo **stesso periodo** (dal primo giorno in cui la finestra di 250 giorni
è piena, ~fine 2000), altrimenti le finestre corte avrebbero un anno di storia in più.

## Decisione 12:43: soglie di ingresso {1.0, 1.5, 2.0, 2.5, 3.0}, uscita a z = 0 (confermate dall'utente)
Ragionamento da scrivere **sempre** nel notebook e nel README (richiesta dell'utente): le tre soglie interne sono
l'esempio del PDF ("1.5, 2, 2.5"); 1.0 e 3.0 sono bordi che servono solo a misurare la sensibilità (1.0 non è "very
negative": entra quasi a ogni oscillazione). Le celle di bordo non si possono scegliere perché la loro media sulle
vicine usa meno celle (5 o 3 invece di 8) ed è più rumorosa; togliere 1.0 renderebbe 1.5 un bordo non sceglibile.

## Decisione 12:45: regola di scelta P2 (confermata dall'utente; da spiegare SEMPRE nel notebook, in parole semplici)
1. Candidate solo le 9 celle interne (soglie 1.5/2/2.5 × finestre 60/120/190); per ognuna media dello Sharpe su sé
   stessa + 8 vicine; vince la media più alta (centro della zona stabile). 2. Almeno 20 trade IS per essere scelta.
3. A parità, meno trade. 4. L'OOS si lancia comunque sulla cella scelta, anche se tutto è negativo: si riporta il
   fallimento. 5. Prove dichiarate: 25. Scritto prima: con 19 anni uno Sharpe IS sotto ~0,4 è compatibile col caso.

## Decisione 12:58: diagnostica aggiunta prima del primo backtest (ok dell'utente)
Oltre a heatmap 5×5 (IS intero) e zoom 5×5: stessa heatmap dello Sharpe sulle **due metà dell'IS** (fine 2000–2009,
2010–2018) per vedere se la zona buona resta nello stesso punto (stabilità nel tempo dei parametri), e heatmap del
**numero di trade**. Stessi backtest: nessuna prova in più (totale 49). La scelta resta sull'IS intero.
Ogni cella parte piatta dallo stesso giorno (primo giorno con la finestra di 250 piena).

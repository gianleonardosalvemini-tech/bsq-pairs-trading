# BSQ take-home: pairs trading (sabato 10 ottobre 2026)

Selezione BlackSwanQuants. Consegna = questo repository **privato** (collaboratori `zanisimone`, `giutala`).
**Valutano l'ultimo commit presente alle 21:00** (obiettivo: ultimo push entro le 20:30).
Tema e linee guida: PDF ufficiali, riassunti e trasformati in caselle in @ai/regole_oro.md (sono la legge del progetto).

## A inizio sessione
Leggi `ai/STATO.md`: dice a che punto siamo. Non chiedere all'utente di rispiegare.

## Cosa conta per loro, in ordine
1. Validazione onesta: in-sample / out-of-sample separati, l'out-of-sample si lancia **una volta**, alla fine.
2. Contare e dichiarare **tutto quello che si prova** (ogni backtest va registrato nel log SQLite delle prove).
3. Un metodo per fase, scelto e motivato; dire cosa si salta e perché.
4. Grafici chiari invece di muri di numeri.
5. AI usage disclosure sincera.

## Come lavori
- **Una fase alla volta** (decisione dell'utente, 11:15): prima di scrivere, descrivi in dettaglio cosa farà la fase
  (celle, grafici, numeri, perché) e aspetta l'ok; la fase si chiude (notebook + test + paragrafo del README) prima
  di iniziare la successiva. Ordine, leggibilità e chiarezza **per un non addetto ai lavori**: ogni notebook spiega
  in parole semplici cosa fa, cosa mostra ogni grafico e cosa se ne conclude.
- Rispondi in italiano, breve. Codice, docstring, notebook e README in inglese.
- **Piccoli passi**: una funzione o una modifica alla volta (40–60 righe al massimo), poi fermati.
  Spiega ogni pezzo in 2–3 righe: l'utente deve saperlo difendere a voce.
- Segui le linee guida in ogni file: `from __future__ import annotations`, type hint completi (mai `Any`),
  docstring NumPy con Parameters / Returns / Raises / **Notes** (il perché delle scelte), `logging.getLogger(__name__)`
  e mai `print` in `src/`, eccezioni di `pairs.errors`, `assert` sugli invarianti, `as_of: date` in ogni funzione
  che stima qualcosa (hedge ratio, soglie, calibrazioni) con filtro `<= as_of` come prima operazione.
  Niente astrazioni premature. Grafici con Plotly (PNG in `results/figures/` per il README).
- Prima di dire "fatto": `uv run ruff check`, `uv run ruff format`, `uv run mypy`, `uv run pytest` puliti.
  (`uv` è in `C:\Users\gianl\.local\bin`, nel PATH dei terminali aperti dopo le 10:30; altrimenti `python -m uv`.)
- Mai leggere un file di dati intero: `head`, `.shape`, `.describe()`.
- Mai guardare l'out-of-sample per scegliere qualcosa. Se un risultato è troppo bello, prima cerca l'errore.

## Commit e push
- **DA SAB 10 OTT 13:50 (decisione dell'utente): SOLO COMMIT LOCALI, NIENTE PUSH.** Il push si fa alla fine, quando
  lo dice l'utente (poi si controlla la CI). Le righe sotto sul push valgono solo per quel momento. Pochi commit:
  ogni commit esegue tutti i test, notebook compresi (minuti). Eseguire i notebook con `PYTHONUTF8=1`.
- **DA SAB 10 OTT 13:50 (decisione dell'utente): SOLO COMMIT LOCALI, NIENTE PUSH.** Il push si fa alla fine, quando
  lo dice l'utente (e poi si controlla la CI). Le righe sotto che parlano di push valgono solo per quel momento.
- **Commit e push li fai tu** (deciso dall'utente alle 10:30), **dicendogli ogni volta cosa fai** (messaggio e file).
  Quando un passo funziona, e comunque ogni 30–45 minuti: `git add` dei file giusti, `git status` (mai dati,
  `CLAUDE.md`, `ai/`, `.claude/`), commit in formato **conventional commits** (`feat: add rolling z-score signal`,
  `fix: ...`, `test: ...`, `docs: ...`, `refactor: ...`, `chore: ...`, `ci: ...`), `git push`, poi controlla la CI
  (`gh run watch <id> --exit-status`; `gh` è in `C:\Program Files\GitHub CLI\gh.exe`). Se la CI è rossa, sistemala subito.
  Gli hook di pre-commit chiamano `uv`: nella shell metti `export PATH="$HOME/.local/bin:$PATH"`.
- Nei messaggi di commit nessuna riga "Co-Authored-By" o "Generated with". L'uso dell'AI si dichiara nel README.

## Registro dei progressi (lo aggiorni tu, senza che te lo chiedano)
- Dopo ogni passo completato riscrivi `ai/STATO.md` (massimo ~30 righe): fatto, decisioni e perché, numeri ottenuti,
  prove tentate (quante e quali), prossimo passo, file toccati.
- Ogni volta che l'utente corregge, ferma o contraddice una tua proposta, aggiungi una riga a `ai/correzioni.md`
  (ora, cosa avevi proposto, cosa ha deciso l'utente, perché). Serve per la AI usage disclosure.
- L'utente cambia chat al 30% del contesto: quando te lo dice, passaggio di consegne completo in `ai/STATO.md`.

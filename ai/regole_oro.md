# Regole d'oro: ogni richiesta dei due documenti, come casella da spuntare

Ogni riga è una cosa che BSQ ha scritto nero su bianco. Colonna "Dove si vede" = la prova nel repository.
**Si rilegge alle 13:00, alle 17:00 e si spunta tutta alle 20:00.** Fonte: P = `2026_takehome_pairs-trading.pdf`
(pagina), G = `programming-guidelines.pdf`, M = mail.

## 1. Il criterio n.1: validazione onesta (P2, P3, P4)
| ✓ | Richiesta | Dove si vede |
|---|---|---|
| ☐ | Split **in-sample / out-of-sample** per data; risultati riportati **separatamente** | tabella IS vs OOS nel README e nel notebook 02 |
| ☐ | L'out-of-sample **non si guarda fino alla fine**; mai tornare indietro a cambiare dopo averlo visto | `Config` con data di split fissata prima; OOS lanciato **una volta**, ora annotata in `trials.sqlite` |
| ☐ | **Dire quante cose ho provato** (soglie, finestre, varianti): il numero conta più del risultato mostrato | `results/trials.sqlite` + tabella "everything I tried" nel README con il conteggio totale |
| ☐ | **Sensibilità** a soglia e finestra: se un piccolo cambio fa crollare il risultato, dirlo | heatmap soglia × finestra (solo in-sample) |
| ☐ | Dire **come ho scelto** la variante mostrata (criterio deciso prima) | README, sezione Phase 2 |
| ☐ | Lo standard della fase 2 vale **per ogni fase in cui si testa qualcosa** (3, 5, estensioni) | ogni notebook distingue IS e OOS |
| ☐ | Spread verificato, non presunto: grafico, attraversamenti della media | notebook 02 |
| ☐ | Dire **quando non sono sicuro**; un risultato nullo spiegato bene è "in great shape" | sezione "Limitations / what I am unsure about" |

## 2. Scelte e ragionamento (P1, P3)
| ✓ | Richiesta | Dove si vede |
|---|---|---|
| ☐ | **Una coppia** per tutto il progetto | `Config` |
| ☐ | **Un metodo per fase** dove c'è un menu, con il **perché** | README: tabella fase → metodo scelto → motivo → cosa ho saltato |
| ☐ | Meglio metà delle fasi fatte bene che tutte a metà; dire **cosa ho scelto di saltare** | README |
| ☐ | **Estensioni della fase 6: OBBLIGATORIE per noi (12:50)**, tutte e quattro dopo un nucleo solido: costi + walk-forward, seconda coppia INTC/AMD, Kalman, paniere | README + notebook 06 |
| ☐ | Fase 4: vincolo Nasdaq-only dichiarato; anche "dati non disponibili / nulla trovato, ecco perché" è valido | notebook 04 + README |
| ☐ | Fase 5: le due serie simulate **insieme**; riportare **la variazione** della performance tra i percorsi | notebook 05: distribuzione (istogramma) dello Sharpe sui percorsi |

## 3. Presentazione (P2)
| ✓ | Richiesta | Dove si vede |
|---|---|---|
| ☐ | "A clear plot beats a wall of numbers": pochi grafici chiari, con titolo, assi, unità | `results/figures/*.png` nel README |
| ☐ | Organizzato **fase per fase**, seguibile da uno sconosciuto | `notebooks/01…05`, README con un paragrafo per fase |
| ☐ | Tabelle leggibili, output curato (G) | README + log `rich` |

## 4. Le tre consegne (P2, P6, M)
| ✓ | Richiesta | Dove si vede |
|---|---|---|
| ☐ | **Notebook o cartella ordinata di script**, fase per fase | `notebooks/`, `src/` |
| ☐ | **Resoconto di mezza pagina**: fasi completate, metodo scelto e perché, cosa farei con più tempo | inizio del README |
| ☐ | **AI usage disclosure**, sempre: cosa ho chiesto, cosa ho controllato, **dove l'ho corretta** | README, sezione "AI usage disclosure" (vera, dal diario `ai/correzioni.md`) |
| ☐ | File di supporto prodotti dall'AI lasciati nella cartella così come sono | cartella `ai/`. **Alle 20:00**: l'utente li rivede, poi si tolgono `CLAUDE.md`, `ai/`, `.claude/` dal `.gitignore` e si committano |
| ☐ | Repository **privato**, collaboratori `zanisimone` e `giutala` invitati | GitHub → Settings → Collaborators |
| ☐ | **Ultimo commit prima delle 21:00** (obiettivo 20:30), push fatto e visibile su GitHub | `git log -1`, pagina GitHub |

## 5. Linee guida di programmazione (G)
| ✓ | Richiesta | Dove si vede |
|---|---|---|
| ☐ | Docstring **NumPy** con Parameters, Returns, Raises, **Notes** su tutte le funzioni pubbliche | `src/pairs/*.py` |
| ☐ | **PEP 8** + type hint completi e precisi, niente `Any`, `from __future__ import annotations` | `ruff check` e `mypy src` puliti |
| ☐ | **ruff** e **mypy** automatici (pre-commit e CI) | `.pre-commit-config.yaml`, `.github/workflows/ci.yml` verde |
| ☐ | Niente astrazioni premature: funzioni esplicite | codice |
| ☐ | Configurazione strutturata (dataclass), fallisce subito se manca qualcosa | `src/pairs/config.py`, `.env.example` |
| ☐ | **Eccezioni personalizzate**, mai `Exception` generica | `src/pairs/errors.py` |
| ☐ | Invarianti con **`assert`** che falliscono rumorosamente | codice + notebook |
| ☐ | Ogni numero **verificato con `assert`** nei notebook end-to-end | ultima cella di ogni notebook + `tests/test_notebooks.py` che li esegue tutti |
| ☐ | **pytest** per le singole funzioni | `tests/`, `uv run pytest` verde |
| ☐ | **`as_of: date`** in hedging/calibrazione, filtro `<= as_of` come prima operazione con `assert` | `spread.py`, `simulate.py` |
| ☐ | **Checkpoint pickle** raw / parsed / validated, **versionati** (Yahoo riscrive gli Adj Close) | `data/*.pkl` nel repository + funzione che li rigenera |
| ☐ | **SQLite** per l'output | `results/trials.sqlite` |
| ☐ | **Python 3.12 + uv + `uv.lock`**; gira con solo `uv sync` | `.python-version`, `uv.lock`, istruzioni nel README |
| ☐ | **Conventional commits** | `git log --oneline` |
| ☐ | **logging.getLogger(__name__)** in ogni modulo, niente `print`; **rich** | `grep -rn "print(" src` vuoto |
| ☐ | **Plotly** per i grafici | notebook + PNG esportati |
| ☐ | PySide6: saltato **dichiarandolo**; **se avanza tempo a fine giornata** (dopo le fasi 1–5, verso le 19) visualizzatore opzionale dei checkpoint | README, "Engineering standards" |

## 6. Le trappole che uccidono un progetto di pairs trading (nostre)
| ✓ | Controllo | Come |
|---|---|---|
| ☐ | β stimato **solo sull'in-sample** (o su finestre passate) | `as_of` + test pytest |
| ☐ | z-score con media e deviazione **mobili e passate** | test: cambiare un dato futuro non cambia lo z-score di oggi |
| ☐ | Posizione decisa in t applicata al rendimento di **t+1** | test pytest |
| ☐ | PnL calcolato **sulle due gambe** (long A, short β B), convenzione scritta | `backtest.py`, Notes della docstring |
| ☐ | Adj Close vs Close: scelta scritta | README, Data |
| ☐ | Costi di transazione inclusi (almeno una stima dichiarata) | `Config.cost_bps` |
| ☐ | Risultato troppo bello → **prima cerco l'errore** | — |

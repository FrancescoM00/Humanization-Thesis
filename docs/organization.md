# Organizzazione e responsabilità dei moduli

Il progetto distingue la produzione dei dati dalla loro valutazione. Un modello restituisce un testo o score individuali; `evaluation/` usa questi risultati per calcolare le misure dell'esperimento. Le entry point in `experiments/` leggono un file YAML e collegano i moduli nell'ordine richiesto.

```text
Humanization/
├── README.md
├── requirements.txt
├── pyproject.toml
├── Humanization.code-workspace
├── .vscode/
├── configs/
├── data/
│   ├── raw/mgtbench/
│   ├── raw/opai/
│   └── processed/
├── src/humanization/
│   ├── datasets/
│   ├── stylometry/
│   ├── semantics/
│   ├── rewriting/
│   ├── detection/
│   ├── naturalness/
│   └── evaluation/
├── experiments/
├── results/
│   ├── published/
│   └── new_runs/
├── docs/
└── tests/
```

## Dati e produzione degli score

- `datasets/` legge i CSV MGTBench e i dati OpAI, prepara campioni e mantiene riconoscibili gli identificatori di fonte. La pulizia produce dati separati: i file originali restano inalterati.
- `stylometry/` estrae caratteristiche linguistiche, come POS, punteggiatura e diversità lessicale. Il vettore delle feature non è una probabilità di origine umana.
- `semantics/` produce similarità ed evidenze NLI. Gli score delle due direzioni rimangono distinti e conservano il legame con frasi o unità confrontate.
- `rewriting/` contiene i sistemi che generano il testo riscritto. I preset Qwen e DIPPER hanno parametri e procedure espliciti.
- `detection/` esegue i detector e salva uno score per testo. La convenzione del punteggio è registrata: uno score basso può indicare maggiore evidenza artificiale per alcuni detector.
- `naturalness/` prepara coppie e ottiene giudizi di naturalezza. L'eventuale client esterno appartiene a questo modulo, non all'aggregazione statistica.

## Evaluation

Le responsabilità principali sono:

| Modulo | Valutazione |
| --- | --- |
| `evaluation/stylometric.py` | Differenze fra feature e capacità discriminativa |
| `evaluation/semantic.py` | Confronto fra condizioni e metriche semantiche |
| `evaluation/detection.py` | Calibrazione, FPR, TPR, ASR e confronti appaiati |
| `evaluation/naturalness.py` | Aggregazione dei giudizi, instabilità all'ordine e concordanza |
| `evaluation/statistics.py` | Strumenti statistici condivisi |
| `evaluation/reporting.py` | Tabelle e report |

Moduli aggiuntivi mantengono separate le analisi più specifiche, come shortlist, robustezza, errori distribuiti e studio appaiato. Non duplicano i pesi dei modelli né introducono un nuovo riscrittore.

## Esempio: Qwen 14B e DIPPER

1. Il dataset fornisce gli stessi originali ai due sistemi.
2. `rewriting/` produce una riscrittura per sistema e fonte.
3. `detection/` assegna score agli originali e alle riscritture. `semantics/` produce supporti NLI e similarità; `stylometry/` produce feature.
4. `evaluation/` confronta i risultati collegati alla stessa fonte. Calcola ASR, variazioni di TPR, aggregazioni semantiche, distanze di stile e intervalli.
5. Report, tabelle e metadati sono salvati sotto `results/new_runs/`.

La trasformazione di una fonte non deve cambiare il suo identificatore per diventare una nuova osservazione indipendente. I confronti mantengono insieme le versioni collegate.

## Risultati di riferimento e nuove esecuzioni

`results/published/` contiene i risultati di riferimento. `data/processed/` contiene input e punteggi necessari alla loro rianalisi. I nuovi output sono separati. Le configurazioni eseguibili sono relative alla cartella del progetto; i manifest di esecuzione registrano gli input e i parametri effettivamente usati. Ricalcolare una tabella dai punteggi salvati è distinto dall'eseguire nuovamente un modello.

Le entry point operative sono i file numerati di `experiments/`. I parametri dei controlli specifici sono in `configs/protocols/`.

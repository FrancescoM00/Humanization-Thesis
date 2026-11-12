# 08 — Scelta della configurazione NLI contestuale

**Riferimento: §1.9 del PDF.** Si sceglie come costruire le premesse NLI e aggregare gli score, usando un pannello di sviluppo separato dal test già esaminato. Si confrontano unità singole, finestre fino a due o tre unità contigue e documento intero, con media, q10, minimo e quota non supportata.

## Input e risultato

[`inputs/nli_calibration`](../../inputs/nli_calibration) contiene 139 riscritture OpAI presumibilmente conservative, 297 alterazioni intenzionali e 137 identità: 573 coppie su 46 fonti. Positivi e perturbazioni sono proxy, non gold umano. Le 141 coppie positive e le 265 alterate del test precedente non vengono usate per scegliere la configurazione.

La policy richiede copertura completa, media sulle identità almeno 0,95 e q10 almeno 0,90. Si seleziona poi la massima AUC di calibrazione fra i candidati ammissibili, con regola di parità prefissata.

| Modalità con minimo bidirezionale | AUC di calibrazione |
| --- | ---: |
| Unità singole | 0,9470 |
| Finestre fino a due unità | 0,9735 |
| Finestre fino a tre unità | 0,9712 |

È selezionata **window2 + min**, con supporto medio sulle identità 0,9710. Il documento intero non ha copertura sufficiente. Il distacco da window3 è piccolo e non dimostra una superiorità generale su dati nuovi. AUC 0,9735 non significa accuratezza del 97,35% né validazione umana.

## Ricalcolare la scelta dai punteggi ricevuti

Questo comando usa soltanto dati e score già inclusi, senza nuove inferenze:

```bash
python scripts/calibrate_contextual_nli.py analyze --input inputs/nli_calibration --scores results/08_nli_calibration/calibration/contextual_nli.csv --output runs/nli_calibration_analysis --bootstrap 2000 --seed 42
```

Produce `candidate_summary.csv`, `auc_by_condition.csv`, `calibration_scores.csv`, `selection.json` e `report.md`. I risultati congelati sono in [`results/08_nli_calibration/calibration`](../../results/08_nli_calibration/calibration); il piccolo pannello esplicativo precedente è in `context_pilot` e non è una stima casuale dell'accuratezza.

## Rifare preparazione e inferenza

Per ricostruire il pannello a partire dall'estensione semantica congelata:

```bash
python scripts/calibrate_contextual_nli.py prepare --pairs inputs/semantic_extension/pairs.jsonl --alignment inputs/semantic_extension/alignment.json --output runs/nli_calibration_inputs
```

Con il checkpoint NLI già scaricato in `data/models/nli`, la nuova inferenza usa:

```bash
python scripts/score_contextual_nli.py --pairs inputs/nli_calibration/pairs.jsonl --alignment inputs/nli_calibration/alignment.json --model data/models/nli --output runs/nli_calibration_scores --device cuda:0 --batch-size 4 --threads 2
```

È disponibile anche il runner [notebooks/run_nli_calibration.py](../../notebooks/run_nli_calibration.py), descritto nella [guida notebook](../../notebooks/README_NLI_Calibration.md). Il run storico ha completato 323.336 evidenze DeBERTa-v3 base float32 in circa 53,7 minuti di GPU, esclusi setup e pianificazione. La CPU è supportata ma richiede molto più tempo. Ricalcolare la scelta dai punteggi salvati è invece un'analisi leggera.

La scelta è successiva allo studio su 126 fonti e precedente alle nuove annotazioni umane, ancora in raccolta. Si applica come score continuo: non seleziona una nuova soglia e non autorizza il riuso di 0,04075, che appartiene alla configurazione storica MiniLM top-3.

# 03 — Conservazione semantica

**Riferimento: §1.4 del PDF.** Si verifica se le metriche riconoscono cambiamenti di significato, distinguendoli da riscritture conservative. Le condizioni generate e le operazioni OpAI forniscono etichette intenzionali o presunte, non giudizi umani indipendenti.

## Dati, codice e risultati

| Controllo | Script principali | Risultati già calcolati |
| --- | --- | --- |
| Identità, negazioni, entità, frasi mescolate, troncamento, cancellazioni; Compress/Expand OpAI | `run_semantic_experiment.py` | [baseline](../../results/03_semantics/baseline/interpretazione.md) |
| Positivi Polish/Paraphrase/Style, alterazioni sottili, scelta dell'aggregatore | `prepare_subtle_perturbations.py`, `prepare_semantic_extension.py`, `analyze_semantic_extension.py` | [extension](../../results/03_semantics/extension/interpretazione.md) |
| Entità: 145 sostituzioni applicate e sottoinsieme di 34 date/importi/percentuali | `compare_entity_policy.py`, `analyze_clean_numeric_subset.py` | [entity_clean](../../results/03_semantics/entity_clean/interpretazione.md), [numeric34](../../results/03_semantics/entity_clean/numeric34/report.md) |
| Similarità allineate, errori su 1/2/3/5 frasi, lunghezza e guardie lessicali | `score_aligned_similarity.py`, `analyze_semantic_fairness.py`, `analyze_minimum_multiplicity.py`, `evaluate_lexical_guards.py` | [fairness](../../results/03_semantics/fairness/interpretazione.md) |

Le implementazioni condivise sono in `src/semantics/`. Gli input congelati si trovano in `inputs/semantic_baseline/`, `inputs/semantic_extension/`, `inputs/semantic_clean/` e `inputs/semantic_fairness/`. Conservano testi, ID, split ed edit originali. I manifest storici si riferiscono al vecchio repository, commit `d2868b3`; la nuova organizzazione non riscrive quegli hash.

## Ricalcolare le analisi, senza eseguire modelli

Dalla radice, dopo l'installazione indicata nel README generale:

```bash
python scripts/analyze_semantic_extension.py \
  --pairs inputs/semantic_extension/pairs.jsonl \
  --scores results/03_semantics/extension/scores.csv \
  --output runs/03_semantics/extension --bootstrap 1000

python scripts/analyze_clean_numeric_subset.py \
  --data-dir inputs/semantic_clean --baseline-data inputs/semantic_baseline \
  --results results/03_semantics/entity_clean \
  --baseline-results results/03_semantics/baseline \
  --output runs/03_semantics/numeric34 --bootstrap 1000
```

Per il confronto completo con aggregazioni uguali:

```bash
python scripts/analyze_semantic_fairness.py \
  --pairs inputs/semantic_extension/pairs.jsonl \
  --distributed-pairs inputs/semantic_fairness/distributed_pairs.jsonl \
  --extra-pairs inputs/semantic_fairness/expanded_positive_pairs.jsonl \
  --extra-pairs inputs/semantic_fairness/expanded_distributed_pairs.jsonl \
  --old-scores results/03_semantics/extension/scores.csv \
  --score results/03_semantics/fairness/aligned_old/aligned_cosine.csv \
  --score results/03_semantics/fairness/aligned_old/aligned_bertscore.csv \
  --score results/03_semantics/fairness/new_nli.csv \
  --score results/03_semantics/fairness/original_panel/embedding.csv \
  --score results/03_semantics/fairness/expanded_panel/embedding.csv \
  --score results/03_semantics/fairness/aligned_new/aligned_cosine.csv \
  --score results/03_semantics/fairness/aligned_new/aligned_bertscore.csv \
  --output runs/03_semantics/fairness --bootstrap 1000
```

Questi comandi usano CPU e punteggi inclusi. Producono nuove tabelle, intervalli bootstrap e report in `runs/`, mantenendo i risultati consegnati. Il controllo del minimo entro la stessa coppia richiede inoltre probabilità NLI e allineamenti storici: il relativo CSV finale è incluso, la cache neurale completa non è distribuita.

## Eseguire nuovamente le metriche

Servono `requirements/semantic.txt`, spaCy small e i modelli in `data/models/`. Le revisioni originali sono:

| `--model` | Checkpoint | Revisione |
| --- | --- | --- |
| `embedding` | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| `bertscore` | `FacebookAI/roberta-base` | `e2da8e2f811d1448a5b465c236feacd80ffbac7b` |
| `nli` | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | `6f5cf0a2b59cabb106aca4c287eed12e357e90eb` |

`download_semantic_models.py --model NOME --revision SHA --output data/models` scarica il checkpoint indicato. Per rivalutare le coppie baseline già congelate:

```bash
mkdir -p data/reruns/semantic_baseline
cp inputs/semantic_baseline/pairs.jsonl data/reruns/semantic_baseline/pairs.jsonl
for stage in lexical embedding bertscore nli report; do
  python scripts/run_semantic_experiment.py --stage "$stage" \
    --data-dir data/reruns/semantic_baseline \
    --output runs/03_semantics/baseline --models data/models --device cpu
done
```

L'NLI richiede prima gli embedding per recuperare le premesse. La GPU accelera le inferenze; la CPU è supportata ma il run completo può essere lungo. Gli stadi caricano un modello per processo. Per nuove coppie, `--stage prepare --input data/clean/samples.jsonl --n 200` costruisce il pannello; `--entity-policy clean` modifica soltanto la condizione entità, lasciando espliciti i casi saltati.

## Lettura dei risultati

La negazione lascia coseno e BERTScore documentali circa a **0,998/0,999**, mentre l'entailment della frase modificata scende a **0,001**. Nel confronto OpAI appaiato il minimo NLI raggiunge **AUC 0,958**, contro **0,607** del coseno allineato e **0,534** di BERTScore allineato. Con errori distribuiti la media può superare il minimo; quest'ultimo diminuisce anche all'aumentare delle opportunità di trovare una frase difficile. Sul sottoinsieme numerico pulito l'entailment locale medio è **0,00257**, a fronte di similarità documentali elevate.

La soglia storica **0,04075** riguarda uno specifico minimo calibrato su OpAI: non è applicata automaticamente ad altri sistemi. La configurazione contestuale successiva è nella [sezione 08](../08_calibrazione_nli/README.md). L'ulteriore test sulle cautele, successivo al PDF, è un [supplemento separato](../../supplements/hedging/README.md).

# Confronto equo e robustezza degli aggregatori

Risultati eseguiti: [interpretazione](../../results/03_semantics/fairness/interpretazione.md), [report completo](../../results/03_semantics/fairness/report.md), [stile sulle frasi toccate](../../results/04_opai_style/touched/report.md). Questa fase supera la raccomandazione precedente di adottare il minimo calibrato come unico indicatore: l'output della suite è un profilo continuo, senza la soglia OpAI 0,04075 applicata ad altri humanizer.

## Cosa è stato confrontato

- Coseno e BERTScore con massimo allineato per ogni frase/chunk e media, q10 o minimo, nelle stesse unità dell'NLI. BERTScore è anche calcolato sugli esatti tre candidati NLI come sensibilità. Non si confronta più soltanto una media documentale con un minimo locale.
- Corruzioni annidate su 1, 2, 3 e 5 frasi, con sorgenti e testi fissi per ogni curva. Primo pannello esplorativo: 392 negativi. Pannello principale aggiuntivo: 87 nuovi positivi e 400 negativi, 50 sorgenti per famiglia e 25 per split. Le fonti aggiuntive sono distinte dalle 99 dei vecchi controlli. Le regole e l'applicabilità non usano punteggi NLI.
- Lunghezza: correlazioni sui positivi, controllo del numero di opportunità entro la stessa coppia e percentili di riferimento per tre fasce apprese solo sulla calibrazione.
- Quattro guardie lessicali con conteggi e cambiamenti di valori; confronto con NLI e combinazione L2. I limiti del lessico v1 e dei conteggi documentali rimangono dichiarati.
- Stile: lineage delle array native OpAI verificata su 200 traiettorie, feature sui passaggi toccati e distanze RMS/Jensen–Shannon. Intensità per frase e copertura sono riportate separatamente; l'ordine atteso non è forzato.

Tutte le classi semantiche restano proxy operative, non gold umane. I nuovi stress test sottili sono prevalentemente numerici; non sono una stima degli errori di DIPPER o di humanizer commerciali.

## Consultare senza inferenze

Leggere i report e i CSV già presenti. `results_semantic_fairness/continuous_suite/continuous_scores.csv` contiene score per coppia e `humanizer_distributions.csv` quantili per sistema/condizione. Il generatore/editor viene usato come etichetta se manca il campo esplicito `humanizer`.

La suite non emette un verdetto semantico, non applica soglie e non applica implicitamente i riferimenti di lunghezza OpAI a una nuova popolazione. Le metriche di supporto/similarità hanno orientamento alto=maggiore conservazione; le guardie sono conteggi/differenze descrittive. Nessuno score è una probabilità certificata di correttezza del documento.

## Riprodurre dai dati ripristinati

Da `/workspace/Humanization`, con modelli e venv Linux già disponibili. I file preparati rifiutano sovrascritture implicite. I dati del run contengono le coppie già congelate: non rigenerarle per leggere risultati.

Per ricreare le coppie solo se assenti:

```bash
.venv/bin/python scripts/prepare_distributed_experiment.py
.venv/bin/python scripts/prepare_expanded_distributed.py
.venv/bin/python scripts/prepare_fairness_inputs.py
```

Se mancano gli allineamenti nuovi, eseguire separatamente sui due pannelli:

```bash
.venv/bin/python scripts/score_semantic_extension.py --stage embedding --data-dir data/semantic_fairness/original_panel --output results_semantic_fairness/original_panel --previous-data data/semantic_extension --previous-output results_semantic_extension --threads 4
.venv/bin/python scripts/score_semantic_extension.py --stage embedding --data-dir data/semantic_fairness/expanded_panel --output results_semantic_fairness/expanded_panel --previous-data data/semantic_extension --previous-output results_semantic_extension --threads 4
.venv/bin/python scripts/prepare_fairness_inputs.py
```

Le metriche allineate usano `scripts/score_aligned_similarity.py`, stadi `cosine` e `bertscore`. Il vecchio pannello ha default già corretti. Per i soli nuovi esempi:

```bash
.venv/bin/python scripts/score_aligned_similarity.py --stage cosine --pairs data/semantic_fairness/new_pairs.jsonl --alignment data/semantic_fairness/alignment.json --output results_semantic_fairness/aligned_new
.venv/bin/python scripts/score_aligned_similarity.py --stage bertscore --pairs data/semantic_fairness/new_pairs.jsonl --alignment data/semantic_fairness/alignment.json --output results_semantic_fairness/aligned_new --threads 4
```

NLI conserva il checkpoint DeBERTa-v3 base float32 e le probabilità precedenti:

```bash
.venv/bin/python scripts/plan_nli_jobs.py --data-dir data/semantic_fairness --output results_semantic_fairness
.venv/bin/python scripts/precompute_nli_onnx.py --data-dir data/semantic_fairness --output results_semantic_fairness --threads 4 --batch-size 8
.venv/bin/python scripts/run_semantic_experiment.py --stage nli --data-dir data/semantic_fairness --output results_semantic_fairness --threads 4 --require-nli-cache
```

Le cache NLI verificano percorso/configurazione dei modelli e metadati dei pesi: spostarle su un'altra macchina o percorso può richiedere una riconciliazione verificata della provenienza. Non cancellarle o rigenerare ore di inferenze per questo motivo; i CSV dei risultati restano utilizzabili indipendentemente dalle cache.

Per rifare soltanto l'analisi, `new_nli.csv` è la selezione delle 879 nuove coppie da `nli.csv`; le vecchie righe non vengono riunite con contatori cumulativi di run diversi:

```bash
.venv/bin/python scripts/analyze_semantic_fairness.py --extra-pairs data/semantic_fairness/expanded_positive_pairs.jsonl --extra-pairs data/semantic_fairness/expanded_distributed_pairs.jsonl --score results_semantic_fairness/new_nli.csv --score results_semantic_fairness/original_panel/embedding.csv --score results_semantic_fairness/expanded_panel/embedding.csv --score results_semantic_fairness/aligned_new/aligned_cosine.csv --score results_semantic_fairness/aligned_new/aligned_bertscore.csv --bootstrap 1000
.venv/bin/python scripts/analyze_minimum_multiplicity.py
.venv/bin/python scripts/evaluate_lexical_guards.py --pairs data/semantic_fairness/pairs.jsonl --scores results_semantic_fairness/nli.csv --output results_semantic_fairness/guards_full --bootstrap 1000
.venv/bin/python scripts/run_opai_touched_style.py --reuse-features
```

Il controllo del numero di opportunità usa soltanto probabilità salvate, senza modelli neurali nuovi. L'analisi stilometrica può riusare la cache locale. I test mirati di allineamento e regole sottili sono passati (8 test); il risultato scientifico è distinto dai test software.

## Esportare score continui di una suite

Il nuovo `scripts/export_semantic_suite.py` accetta coppie JSONL con `id`, `original`, `modified`, e opzionalmente `humanizer`, `source_group_id`, `dataset`, `perturbation`. Riunisce CSV già calcolati con lo stesso ID, senza richiedere etichette di conservazione o uno split di benchmark. Esempio del run eseguito:

```bash
.venv/bin/python scripts/export_semantic_suite.py --pairs data/semantic_fairness/pairs.jsonl --score results_semantic_fairness/nli.csv --score results_semantic_fairness/aligned_old/aligned_cosine.csv --score results_semantic_fairness/aligned_old/aligned_bertscore.csv --score results_semantic_fairness/aligned_new/aligned_cosine.csv --score results_semantic_fairness/aligned_new/aligned_bertscore.csv --score results_semantic_fairness/guards_full/guard_scores.csv --output results_semantic_fairness/continuous_suite
```

Per un nuovo humanizer, i suoi output devono prima essere realmente prodotti e confrontati; qui non sono stati eseguiti DIPPER o nuovi servizi. Una soglia futura è solo secondaria e va ricalibrata su riferimenti rappresentativi del nuovo sistema.

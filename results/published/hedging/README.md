# Estensione delle alterazioni di cautela

Il [report](report.md) e il [PDF](report.pdf) discutono il test mirato. Il [protocollo](protocol.json) mantiene la soglia documentale storica esatta **0,040749866515398**, senza ricalibrazione. Questo controllo non modifica gli esperimenti precedenti o la configurazione contestuale corrente della suite.

## Materiali

- `pairs.jsonl`: pannello finale, 60 originali, 120 modifiche con intento di alterazione e 60 parafrasi con intento conservativo. Un originale contribuisce tre coppie, non tre fonti indipendenti.
- `sources.jsonl`, `preparation_manifest.json`: provenienza e composizione finali; 22 Essay, 16 Reuters, 22 WritingPrompts.
- `assistant_review.csv`, `review_pairs.csv`: lettura dell'assistente prima degli score; 90 giudizi validi, 90 qualificati. Non sono annotazioni umane.
- `replacements.csv`, `review_freeze.json`, `pre_review/`: quattro sostituzioni anteriori all'inferenza, motivazioni, dati iniziali e revisioni anche della riserva. Il pannello congelato è nel commit `f5170b9`.
- `preparation_failure_*.json`, `candidate_pool_debug_*.jsonl`, `screening_attempt_*.csv`: documentazione degli screening iniziali incompleti; **non sono pannelli di test aggiuntivi**.
- `source_overlap_audit.json`: confronto con esperimenti precedenti. Cinque gruppi sono condivisi con lo study Kaggle, ma con generatori differenti e nessuna coincidenza di originale macchina. Nessuna coincidenza con gli originali OpAI della calibrazione NLI contestuale.
- `nli.csv`, `pair_scores.csv`, `metrics.csv`, `directional_summary.csv`: risultati del modello, punteggi derivati e intervalli per originale.
- `human_annotation.csv`, `annotation_instructions.md`: modulo umano **vuoto** e istruzioni. La chiave `human_annotation_key.csv` deve restare separata dagli annotatori. L'archivio `annotazione_umana.zip` contiene soltanto modulo e istruzioni.
- `report_cautele.zip`: documenti, risultati e dati finali; non contiene pesi dei modelli, ambiente virtuale o cache probabilistica.

## Codice e riproduzione

`semantics/hedging_controls.py` costruisce le triplette da cautele native. `scripts/prepare_hedging_controls.py` applica il protocollo senza utilizzare punteggi NLI. `scripts/finalize_hedging_controls.py` applica le revisioni testuali, archivia il pannello iniziale e sostituisce intere triplette invalide dalla riserva deterministica. La revisione umana non è prodotta dal codice.

L'inferenza riutilizza `scripts/run_semantic_experiment.py`, `scripts/plan_nli_jobs.py`, `scripts/initialize_nli_reference_cache.py` e `scripts/precompute_nli_onnx.py`. MiniLM recupera le premesse; DeBERTa-v3 base float32 valuta entailment. L'export ONNX non quantizzato accelera l'esecuzione su CPU e viene confrontato con predizioni native del medesimo checkpoint. Non si ricalibra una soglia durante questo controllo numerico.

Per rigenerare l'analisi dai punteggi già inclusi, dalla radice del progetto:

```bash
.venv/bin/python scripts/analyze_hedging_controls.py \
    --pairs results_hedging_extension/pairs.jsonl \
    --scores results_hedging_extension/nli.csv \
    --protocol results_hedging_extension/protocol.json \
    --review results_hedging_extension/assistant_review.csv \
    --output results_hedging_extension
```

Per una nuova inferenza basta copiare le coppie finali nella directory di lavoro dedicata `data/semantic_hedging`, eseguire lo stage `embedding`, pianificare i job NLI, inizializzare il riferimento nativo, precomputare le probabilità ONNX e infine eseguire lo stage `nli` con `--require-nli-cache`. I modelli devono essere disponibili in `data/models`; cache e allineamenti restano in `data/semantic_hedging`, ignorata da Git. Le versioni iniziali e finali del pannello non vanno mescolate.

`semantics/hedging_analysis.py` calcola AUC con rischio uguale a meno il supporto, sensibilità e falsi allarmi alla soglia fissa, 2.000 bootstrap per originale, sottoinsiemi revisionati e diagnostiche locali. Gli intervalli di Wilson sono secondari e applicati soltanto alle classi con una singola osservazione per originale.

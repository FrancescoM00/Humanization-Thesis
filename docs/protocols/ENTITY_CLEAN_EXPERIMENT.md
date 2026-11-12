# Sostituzioni di entità: politica clean v2

La patch fornita dall'utente aggiunge una condizione distinta, `entity_substitution_clean`, mantenendo la politica v1 come default. È stata applicata, rivista e collegata a `scripts/run_semantic_experiment.py` tramite `--entity-policy {v1,clean}`. I risultati storici rimangono in `results_semantic_full/`; il nuovo campione e i risultati usano `data/semantic_clean/` e `results_semantic_clean/`.

Copie scaricabili dei dati sono in [exports/semantic_clean/](https://github.com/FrancescoM00/Humanization/blob/d2868b323938a2a9371c2a872ad86b5596d0f097/exports/semantic_clean/README.md) sul branch `main`; il backup dati di `cloud-backup` conserva l'intero stato scientifico del run. Sono disponibili sia la [patch originale](https://github.com/FrancescoM00/Humanization/blob/d2868b323938a2a9371c2a872ad86b5596d0f097/patches/entity_clean_v2.original.patch) sia la [patch corretta](https://github.com/FrancescoM00/Humanization/blob/d2868b323938a2a9371c2a872ad86b5596d0f097/entity_clean_v2_corrected.patch).

## Revisione della patch

I 15 test forniti passavano già con la patch originale. La revisione ha trovato casi non coperti dai test: omografi come il mese May e il modale may; cognomi contenuti nel nome completo di un'altra persona; collisione con un cognome di destinazione già presente; possessivi inglobati dal NER; simboli monetari esterni allo span. Sono state aggiunte guardie conservative solo alla nuova politica e regressioni dedicate.

Verifica finale: **15 test originali + 19 regressioni = 34 passati**; altri cinque test riguardano il comparatore e il riuso degli allineamenti. Tutte le funzioni preesistenti hanno lo stesso AST della baseline, salvo l'estensione di `generate_perturbation_pairs` per selezionare le condizioni. Il campione definitivo usa `guard_revision=review_fixes_3`: **145 entity applicate, 55 saltate, 2.297 coppie totali**, con **2.007 coppie non interessate identiche**. L'[audit testuale indipendente dagli score](../../results/03_semantics/entity_clean/clean_entity_text_audit.md) descrive le verifiche meccaniche e i limiti NER residui.

Il nuovo parametro `conditions` viene materializzato per funzionare anche con un iteratore su più documenti. Una condizione sconosciuta produce un errore, anziché ricadere silenziosamente nella cancellazione casuale. Con la chiamata preesistente, le sei condizioni v1 restano le stesse. Le righe clean registrano la politica e la revisione delle guardie nei metadati.

La coorte viene selezionata con l'idoneità v1, come nell'esperimento storico. Nella modalità clean viene sostituita solo la condizione entity: stessa posizione nella lista, nomi compatibili con la pipeline e ID distinti che conservano `entity_substitution_clean`. Le fonti senza un candidato applicabile sono registrate in `skipped.jsonl`; non vengono rimpiazzate con altre fonti. Il controllo identità locale delle entità segue la nuova frase bersaglio e può cambiare; gli altri controlli rimangono identici.

## Esecuzione

Dalla radice del progetto, il comando completo è:

```bash
.venv/bin/python scripts/run_semantic_experiment.py --entity-policy clean --data-dir data/semantic_clean --output results_semantic_clean --opai-file data/opai/records.jsonl --threads 4
```

Il parametro OpAI conserva anche le stesse 800 coppie dello studio storico; senza questo parametro la preparazione non le include. Le directory del nuovo esperimento devono essere distinte da quelle dei risultati storici.

L'esecuzione effettuata usa gli stessi stadi e modelli del comando completo, ma li avvia separatamente per usare il percorso CPU ONNX float32 già validato:

```bash
.venv/bin/python scripts/run_semantic_experiment.py --stage prepare --entity-policy clean --data-dir data/semantic_clean --output results_semantic_clean --opai-file data/opai/records.jsonl --threads 4
.venv/bin/python scripts/run_semantic_experiment.py --stage lexical --entity-policy clean --data-dir data/semantic_clean --output results_semantic_clean --threads 4
.venv/bin/python scripts/run_semantic_experiment.py --stage embedding --entity-policy clean --data-dir data/semantic_clean --output results_semantic_clean --threads 4
.venv/bin/python scripts/run_semantic_experiment.py --stage bertscore --entity-policy clean --data-dir data/semantic_clean --output results_semantic_clean --threads 4
```

Embedding e BERTScore vengono ricalcolati sull'intero nuovo campione. Prima dello stadio NLI sono copiate nella nuova directory le cache `nli_probability_cache.jsonl` e `nli_pytorch_reference.jsonl` del run storico, senza modificarne gli originali. Le probabilità vengono riusate soltanto per la stessa coppia di stringhe e il medesimo checkpoint; i nuovi confronti sono inferiti con DeBERTa-v3 **base float32**, senza quantizzazione:

```bash
.venv/bin/python scripts/compare_entity_policy.py --freeze-unchanged-alignment
.venv/bin/python scripts/plan_nli_jobs.py --data-dir data/semantic_clean --output results_semantic_clean
.venv/bin/python scripts/precompute_nli_onnx.py --data-dir data/semantic_clean --output results_semantic_clean --threads 4 --batch-size 8
.venv/bin/python scripts/run_semantic_experiment.py --stage nli --entity-policy clean --data-dir data/semantic_clean --output results_semantic_clean --threads 4 --require-nli-cache
.venv/bin/python scripts/run_semantic_experiment.py --stage report --entity-policy clean --data-dir data/semantic_clean --output results_semantic_clean
.venv/bin/python scripts/compare_entity_policy.py
```

I risultati numerici e la copertura finale sono in [entity_policy_comparison.md](../../results/03_semantics/entity_clean/entity_policy_comparison.md). Il comparatore richiede identità integrale delle coppie non interessate, incluse fixture e OpAI, e confronta le metriche con tolleranza numerica `atol=rtol=1e-6`. I contatori di lavoro cumulativo vengono esclusi. Non copia i vecchi risultati per imporre l'invarianza.

Il flag di congelamento è un passaggio preparatorio esplicito, da eseguire una sola volta dopo l'embedding e prima di pianificare/inferire l'NLI. Sulle 2.007 coppie identiche riusa il retrieval storico, dopo verifica di modello, hash dei testi, chunk e budget; le entità clean e i loro controlli mantengono il retrieval nuovo. Il coseno resta quello ricalcolato. Il ranking di tre coppie aveva piccole differenze numeriche, di cui una con un'evidenza diversa: mantenere fisso il piano delle evidenze evita di confondere questo effetto con la politica delle entità. L'allineamento appena ricalcolato e il manifest `unchanged_alignment_reuse.json` sono conservati per il confronto.

Il file delle metriche era stato esteso fra l'embedding storico e l'NLI storico: i suoi hash globali dei due embedding non coincidono. Non si dichiara un'identità del sorgente storico non disponibile. Il controllo usa la coincidenza fra codice NLI storico e codice corrente, la medesima directory dei modelli, l'identità delle unità e il confronto numerico degli embedding sulle coppie invariate. I manifest storici rimangono intatti.

## Interpretazione

Il confronto principale riguarda esclusivamente `controlled/entity_substitution` sulle fonti applicabili a entrambe le politiche. Sono riportate separatamente le medie complessive e la composizione per tipo, perché la maggiore selettività della clean cambia il campione. Anche a fonte fissa possono cambiare entità selezionata, frase bersaglio e quantità di menzioni modificate: il delta non isola causalmente una singola correzione.

Il nome clean descrive guardie più conservative, non annotazioni gold. Rimangono errori NER, granularità geografica non risolta e coreferenza incompleta. Non si usa l'NLI per certificare che una perturbazione sia valida. L'eventuale lettura testuale dell'assistente è distinta dalle annotazioni manuali dell'utente; queste ultime non sono incluse nella patch caricata.

# Dal capitolo sperimentale agli script

La consegna segue **Sezione_sperimentalev2 (2).pdf**, fornito dall'autore, sezioni **1.2–1.9**. Il documento di riferimento ha SHA-256 `a66c7dfc6a03a6347a12e0f6218439048784fd9ea5634f2972b5021336341447`. La valutazione della naturalezza (§1.10) è fuori dal perimetro attuale.

Il codice proviene dal repository `FrancescoM00/Humanization`, stato scientifico al commit **`d2868b3`**. Il repository originale rimane separato; la riorganizzazione offre un percorso di lettura e di esecuzione più breve. Non rappresenta un nuovo esperimento.

| Sezione del PDF | Guida di esecuzione | Script principali | Risultati congelati |
| --- | --- | --- | --- |
| 1.2 Dataset e lunghezza | [01_dataset](../experiments/01_dataset/README.md) | `import_mgtbench.py`, `run_style_experiment.py`, `run_length_matched_style.py`, `download_opai.py` | `results/02_stylometry/baseline`, `length_matched` |
| 1.3 Feature, classificazione, shortlist e robustezza | [02_stilometria](../experiments/02_stilometria/README.md) | `run_style_experiment.py`, `run_length_matched_style.py`, `run_shortlist_validation.py`, `run_shortlist_length_robustness.py`, relativi `analyze_*` | `results/02_stylometry/` |
| 1.4.1–1.4.2 Metriche semantiche e perturbazioni iniziali | [03_conservazione_semantica](../experiments/03_conservazione_semantica/README.md) | `run_semantic_experiment.py`, `download_semantic_models.py` | `results/03_semantics/baseline` |
| 1.4.3 Controlli conservativi e alterazioni sottili | [03_conservazione_semantica](../experiments/03_conservazione_semantica/README.md) | `prepare_semantic_extension.py`, `prepare_subtle_perturbations.py`, `analyze_semantic_extension.py`, `score_aligned_similarity.py` | `results/03_semantics/extension`, `fairness/aligned_old` |
| 1.4.4 Entità e quantità | [03_conservazione_semantica](../experiments/03_conservazione_semantica/README.md) | `run_semantic_experiment.py --entity-policy clean`, `compare_entity_policy.py`, `analyze_clean_numeric_subset.py` | `results/03_semantics/entity_clean`, sottocartella `numeric34` |
| 1.4.5 Errori distribuiti, lunghezza e guardie | [03_conservazione_semantica](../experiments/03_conservazione_semantica/README.md) | `prepare_distributed_experiment.py`, `prepare_expanded_distributed.py`, `analyze_semantic_fairness.py`, `analyze_minimum_multiplicity.py`, `evaluate_lexical_guards.py` | `results/03_semantics/fairness` |
| 1.5 Stile nelle frasi OpAI modificate | [04_stile_opai](../experiments/04_stile_opai/README.md) | `run_opai_style.py`, `run_opai_touched_style.py` | `results/04_opai_style/document`, `touched` |
| 1.6 Riscrittori e detector | [05_riscrittori_detector](../experiments/05_riscrittori_detector/README.md) | `run_humanizers.py`, `run_detectors.py`, `run_zero_shot_detectors.py` | preset in `configs/`; risultati nelle due fasi seguenti |
| 1.7 Pilot Qwen 3B/DIPPER, 42 fonti | [06_pilot_42](../experiments/06_pilot_42/README.md) | notebook Pilot e `analyze_humanizer_results.py pilot` | `results/06_pilot_42/qwen`, `dipper` |
| 1.8 Studio con quattro preset, 126 fonti | [07_studio_126](../experiments/07_studio_126/README.md) | notebook Study e `analyze_humanizer_results.py study` | `results/07_study_126` |
| 1.9 Scelta della configurazione NLI | [08_calibrazione_nli](../experiments/08_calibrazione_nli/README.md) | `calibrate_contextual_nli.py`, `score_contextual_nli.py`, runner NLI | `results/08_nli_calibration/calibration` |

I nomi degli script nella tabella sono relativi a `scripts/`; i notebook sono sotto `notebooks/`. Le librerie comuni si trovano in `src/stylometry`, `src/semantics`, `src/humanizers`: estrattori e metriche non vengono duplicati per ogni capitolo.

## Tre percorsi distinti

1. **Leggere i risultati:** aprire il README dell'esperimento e i report/CSV collegati. Non richiede GPU.
2. **Ricalcolare le analisi:** eseguire i comandi sugli input e sui punteggi congelati, con output sotto `runs/`. Le guide distinguono i casi che richiedono anche gli allineamenti o altri input.
3. **Ripetere modelli ed estrazione:** usare i comandi completi o i notebook, scaricando i checkpoint e predisponendo le risorse indicate. Questo percorso è più costoso e non è necessario per controllare conteggi e tabelle già disponibili.

I report e i manifest congelati mantengono i riferimenti al progetto originale. Un hash del codice storico identifica l'implementazione eseguita allora: non va riscritto come se fosse l'hash della nuova organizzazione. I percorsi d'ingresso dei protocolli sono risolti esplicitamente verso le cartelle nuove. Le verifiche di riproduzione segnalano eventuali differenze invece di modificare i risultati pubblicati.

## Materiale successivo al PDF

Il PDF include già le verifiche sulla shortlist (C1/C2/C3/casuali) e il controllo con 100 candidate. L'ultima estensione sulle espressioni di cautela non è ancora inclusa: è documentata nel [supplemento dedicato](../supplements/hedging/README.md). Non viene aggiunta implicitamente al testo del capitolo.

Le annotazioni umane reference sono ancora in raccolta. Le etichette intenzionali delle perturbazioni, i positivi OpAI presunti e le revisioni dell'assistente rimangono distinti da giudizi umani indipendenti.

# Humanization — codice e risultati degli esperimenti

Questo ZIP raccoglie il codice usato per la parte sperimentale della tesi, i dati e i risultati delle esecuzioni già svolte. Il lavoro studia se la riscrittura di un testo generato da un modello linguistico ne riduce la rilevabilità, quanto ne conserva il significato e come ne modifica lo stile.


## 1. Contenuto dello ZIP

```text
Humanization/
├── README.md                  Guida
├── requirements.txt           Dipendenze Python
├── pyproject.toml             Configurazione del package Python
├── Humanization.code-workspace
├── .vscode/                   Per aprire ed eseguire il progetto in VS Code
├── configs/                   Parametri degli esperimenti e percorsi dei file
│   └── protocols/             Regole fissate per shortlist e controllo delle cautele
├── data/
│   ├── raw/
│   │   ├── mgtbench/          Essay_LLMs.csv, Reuters_LLMs.csv e WP_LLMs.csv originali
│   │   └── opai/              Parquet ufficiali di OpAI-Bench
│   ├── processed/             Campioni usati, coppie, riscritture e score individuali
│   └── resources/             Risorse linguistiche locali
├── src/humanization/          Implementazione dei metodi, divisa per componente
├── experiments/               Script numerati per avviare le analisi
├── results/
│   ├── published/             Risultati delle esecuzioni già svolte
│   └── new_runs/              Creata quando si esegue una nuova analisi
├── tools/                     Script per costruire lo ZIP
├── requirements/              Profili delle dipendenze CPU e GPU
├── docs/                      Registri tecnici di controllo e provenienza
└── THIRD_PARTY_NOTICES.txt      Attribuzioni e riferimenti ai progetti esterni
```

I CSV MGTBench contengono testi umani e testi prodotti da diversi generatori nei domini Essay, Reuters e WritingPrompts. I testi umani servono per il confronto stilistico e per calibrare i detector. **Negli esperimenti di riscrittura si parte invece da un testo macchina e se ne valuta la versione riscritta.** 

`data/processed/` conserva gli input effettivamente impiegati: non è necessario ricostruire i campioni per leggere i risultati. In particolare, `study_126/` contiene le riscritture e gli score dello studio su 126 originali; `pilot_42/` quelli delle prove preliminari su 42 originali. 

## 2. Cosa fanno gli script principali

Gli script di `experiments/` sono i punti di avvio. Leggono i parametri da `configs/` e richiamano le funzioni di `src/humanization/`.

| Script | Cosa fa | Risultati già disponibili |
| --- | --- | --- |
| `00_data.py` | Controlla i tre CSV MGTBench e prepara i campioni quando richiesto. | Input utilizzati in `data/processed/`. |
| `01_stylometry.py` | Estrae le feature stilistiche; esegue confronti umano/macchina, regressione logistica e analisi della selezione delle feature. | `results/published/02_stylometry/` |
| `02_semantic_validation.py` | Calcola le metriche semantiche o rianalizza gli score salvati delle coppie originale/modificato. | `results/published/03_semantics/` |
| `03_opai_editing.py` | Misura come cambia lo stile nelle operazioni OpAI, sul documento intero e sulle sole frasi modificate. | `results/published/04_opai_style/` |
| `04_rewriting.py` | Produce le riscritture con Qwen e DIPPER, oppure recupera quelle già salvate. Registra anche i fallimenti. | Testi in `data/processed/study_126/rewriting/rewrites.jsonl`. |
| `05_detection.py` | Calcola gli score dei detector sugli originali, sulle riscritture e sugli umani di calibrazione. | Score in `data/processed/study_126/detection/` e `zero_shot/`. |
| `06_naturalness.py` | Prepara e analizza confronti di naturalezza; può interrogare un giudice remoto solo con attivazione esplicita. | Prove esplorative in `data/processed/naturalness_reference/`; non costituiscono validazione umana. |
| `07_evaluation.py` | Riunisce gli score e calcola prestazioni dei detector, confronti fra riscrittori, intervalli e riepiloghi semantici/stilistici. | `results/published/07_study_126/` e confronto fra pilot. Gestisce anche le analisi NLI e cautele già salvate. |

La separazione è questa: **riscrittura → misure sui testi → aggregazione e confronto dei risultati**. I primi due passaggi producono testi e score individuali; `evaluation` costruisce le tabelle finali.

### Dove sono implementati i metodi

| Componente in `src/humanization/` | File principali e funzione |
| --- | --- |
| `datasets/` | `mgtbench.py` legge e prepara MGTBench; `opai.py` legge i dati di editing. |
| `stylometry/` | `features.py` calcola le 105 feature con spaCy, textstat e MATTR; `touched.py` e `opai.py` gestiscono lo stile nelle porzioni modificate. |
| `semantics/` | `perturbations.py`, `subtle_perturbations.py` e `hedging_controls.py` costruiscono le alterazioni; `positive_controls.py` prepara i controlli di significato conservato. `metrics.py`, `nli.py`, `contextual_nli.py` e `aligned_similarity.py` calcolano le misure. |
| `rewriting/` | `qwen.py` e `dipper.py` gestiscono le due famiglie di riscrittori; `pipeline.py` organizza generazione e salvataggio; `controls.py` prepara i controlli nulli. |
| `detection/` | `roberta.py` gestisce i classificatori; `binoculars.py` implementa il detector zero-shot. |
| `naturalness/` | `pairwise.py` prepara i confronti nei due ordini; `groq.py` gestisce il servizio remoto. Modulo opzionale. |
| `evaluation/` | `study.py` analizza lo studio; `detection.py` calcola le metriche di rilevazione; `shortlist*.py` analizza selezione e stabilità delle feature; `hedging.py` analizza le cautele. Gli altri moduli gestiscono aggregazioni, statistica e report. |

Le sottocartelle `workflows/` collegano questi metodi per le singole analisi: ad esempio `run_semantic_experiment.py` esegue le fasi semantiche e `analyze_semantic_extension.py` ne analizza i risultati. Tutti i percorsi della tabella sono relativi alla componente indicata.

I modelli principali sono Qwen2.5-3B-Instruct, Qwen2.5-14B-Instruct e DIPPER XXL per la riscrittura; OpenAI RoBERTa, Fakespot, RADAR e Binoculars per la detection. Per la semantica si usano ROUGE-L, BLEU, embedding MiniLM, BERTScore con RoBERTa-base e DeBERTa-v3-base addestrato su NLI. Revisioni dei checkpoint e parametri sono conservati nelle configurazioni e nei manifest. «DIPPER a tre frasi» indica finestre di tre frasi, non tre applicazioni successive del modello; «identità» indica il testo lasciato invariato come controllo.

## 3. Dove leggere i risultati già ottenuti

**Tutti i percorsi di questa tabella partono da `results/published/`.** Ogni riga identifica il tipo di esperimento e indica le tabelle e i grafici da aprire. I nomi tecnici delle cartelle collegano codice, input e risultati; i file JSON conservano parametri e informazioni delle esecuzioni.

| Esperimento | Cartella | Tabelle e grafici da consultare |
| --- | --- | --- |
| Confronto stilometrico iniziale umano/macchina | `02_stylometry/baseline/` | [Differenze fra feature](results/published/02_stylometry/baseline/feature_effects.csv), [prestazioni del classificatore](results/published/02_stylometry/baseline/classifier_metrics.csv), [grafico delle differenze](results/published/02_stylometry/baseline/feature_effects.pdf). |
| Stilometria con lunghezza abbinata: 300 umani e 300 macchina per ciascuno dei tre domini | `02_stylometry/length_matched/` | [Prestazioni](results/published/02_stylometry/length_matched/classifier_metrics.csv), [controllo delle lunghezze](results/published/02_stylometry/length_matched/length_balance.csv), [grafico delle differenze](results/published/02_stylometry/length_matched/feature_effects.pdf). |
| Verifica delle quattro feature: tutte le feature, shortlist fissa, selezione sul training e quaterne casuali | `02_stylometry/shortlist/` | [AUC e intervalli](results/published/02_stylometry/shortlist/metrics.csv), [frequenze di selezione](results/published/02_stylometry/shortlist/selection_frequency.csv), [grafico delle AUC](results/published/02_stylometry/shortlist/auc_comparison.pdf). |
| Robustezza della selezione escludendo cinque indicatori sensibili alla lunghezza: 105 contro 100 candidati | `02_stylometry/shortlist_length_robustness/` | [AUC e intervalli](results/published/02_stylometry/shortlist_length_robustness/metrics.csv), [confronto delle selezioni](results/published/02_stylometry/shortlist_length_robustness/selection_comparison.csv), [grafico delle AUC](results/published/02_stylometry/shortlist_length_robustness/auc_comparison.pdf). |
| Validazione semantica iniziale: negazioni, entità, frasi mescolate, troncamento, cancellazione e asimmetria OpAI | `03_semantics/baseline/` | [Riepilogo delle metriche](results/published/03_semantics/baseline/metric_summary.csv), [grafico delle perturbazioni](results/published/03_semantics/baseline/semantic_perturbations.pdf), [asimmetria OpAI](results/published/03_semantics/baseline/opai_asymmetry.csv). |
| Sostituzioni di entità corrette dopo la revisione degli esempi | `03_semantics/entity_clean/` | [Confronto fra le due politiche](results/published/03_semantics/entity_clean/entity_policy_summary.csv), [differenze appaiate](results/published/03_semantics/entity_clean/entity_policy_paired_deltas.csv). |
| Controllo sulle sole 34 sostituzioni di date, importi e percentuali | `03_semantics/entity_clean/numeric34/` | [Riepilogo delle metriche](results/published/03_semantics/entity_clean/numeric34/summary.csv), [risultati per categoria](results/published/03_semantics/entity_clean/numeric34/by_category.csv). |
| Controlli positivi OpAI, alterazioni sottili e confronto degli aggregatori semantici | `03_semantics/extension/` | [AUC delle metriche](results/published/03_semantics/extension/metric_auc.csv), [AUC per gruppo](results/published/03_semantics/extension/auc_by_stratum.csv), [grafico degli aggregatori](results/published/03_semantics/extension/aggregator_auc.pdf). |
| Confronto equo fra metriche: allineamento per frase, errori su più frasi, effetto della lunghezza e guardie lessicali | `03_semantics/fairness/` | [Confronto delle metriche](results/published/03_semantics/fairness/fair_metric_auc.csv), [alterazioni su più frasi](results/published/03_semantics/fairness/distributed_auc.csv), [grafico degli aggregatori](results/published/03_semantics/fairness/fair_aggregation_auc.pdf), [guardie lessicali](results/published/03_semantics/fairness/guards_full/metric_auc.csv). |
| Variazione dello stile dopo Polish, Paraphrase, Style Rewrite, Compress ed Expand | `04_opai_style/` | [Documento intero](results/published/04_opai_style/document/summary.csv), [distanze sulle sole frasi modificate](results/published/04_opai_style/touched/distance_summary.csv), [grafico delle distanze](results/published/04_opai_style/touched/distance_rms_coverage.pdf). |
| Prove preliminari su 42 originali: baseline 3B e DIPPER sul passaggio intero | `06_pilot_42/` | [Detection della baseline 3B](results/published/06_pilot_42/qwen/detection_summary.csv), [confronto appaiato delle ASR](results/published/06_pilot_42/dipper/paired_asr_comparison.csv), [confronto fra riscrittori](results/published/06_pilot_42/dipper/comparison_with_qwen.csv). |
| Studio su 126 originali: quattro configurazioni di riscrittura, controlli nulli e quattro detector | `07_study_126/` | [Detection](results/published/07_study_126/detection_overall.csv), [supporto NLI](results/published/07_study_126/nli_primary_summary.csv), [stile](results/published/07_study_126/style_summary.csv), [controlli nulli](results/published/07_study_126/null_comparisons.csv). |
| Scelta della modalità NLI su controlli positivi, alterazioni e identità | `08_nli_calibration/calibration/` | [Confronto delle modalità](results/published/08_nli_calibration/calibration/candidate_summary.csv), [AUC per alterazione](results/published/08_nli_calibration/calibration/auc_by_condition.csv). |
| Estensione mirata delle alterazioni di cautela, con soglia storica invariata | `hedging/` | [Metriche e intervalli](results/published/hedging/metrics.csv), [supporto nelle due direzioni](results/published/hedging/directional_summary.csv), [grafico delle distribuzioni](results/published/hedging/support_distributions.pdf). |

Nello studio su 126 originali, 42 erano già presenti nei pilot e 84 sono aggiuntivi. Le quattro configurazioni di riscrittura sono la baseline Qwen 3B, Qwen 14B, DIPPER sul passaggio intero e DIPPER a finestre di tre frasi; l'identità serve da riferimento. Le annotazioni indipendenti richieste ai colleghi non sono ancora incluse: le revisioni dell'assistente sono indicate come tali e non valgono come giudizi umani indipendenti.

Nei risultati di detection, **TPR** è la quota di testi macchina rilevati, **FPR** la quota di umani erroneamente rilevati e **ASR** la quota di originali già rilevati che, dopo riscrittura, eludono il detector. Nelle analisi delle feature e delle metriche semantiche, **AUC** misura la capacità di separare i due gruppi dell’esperimento, per esempio umano/macchina o significato conservato/alterato.

I CSV riportano i valori numerici, i PDF delle figure mostrano i grafici e i JSON conservano parametri e informazioni dell'esecuzione. `predictions.npz` contiene predizioni numeriche compresse. Le coppie e le riscritture in formato JSONL hanno una registrazione per riga.

## 4. Comandi facoltativi per riprodurre alcune analisi

Questa parte serve solo a chi desidera eseguire il codice. Usare **Python 3.12**, aprire il terminale nella cartella estratta e predisporre l'ambiente:

```bash
python -m venv .venv
```

Attivazione: `source .venv/bin/activate` su Linux/macOS oppure `.venv\Scripts\Activate.ps1` in PowerShell. Poi:

```bash
python -m pip install -r requirements.txt
```

**Verificare i CSV ufficiali**, senza modificarli:

```bash
python experiments/00_data.py --mode reproduce
```

**Ricalcolare le tabelle dello studio sui riscrittori e il confronto dei pilot** dai testi e dagli score salvati:

```bash
python experiments/07_evaluation.py --mode reproduce --output results/new_runs/esempio_evaluation
```

I risultati sono in `esempio_evaluation/study_126/` e `esempio_evaluation/pilot_42/`; `reproduction_check.csv` confronta i valori con le tabelle di riferimento.

**Ricalcolare il confronto fra selezione da 105 e da 100 feature candidate** dalle predizioni salvate:

```bash
python experiments/01_stylometry.py --mode reproduce --analysis shortlist_100 --output results/new_runs/esempio_stilometria
```

Si ottengono `metrics.csv`, `paired_differences.csv`, grafici e tabelle delle selezioni.

**Rianalizzare i controlli positivi e le alterazioni semantiche** dagli score salvati:

```bash
python experiments/02_semantic_validation.py --mode reproduce --output results/new_runs/esempio_semantica
```

Questi esempi usano CPU e dati inclusi: ricalcolano le analisi senza eseguire nuovamente i modelli. Gli output vanno in `results/new_runs/` e non sostituiscono i risultati già consegnati. Le opzioni di ogni script si consultano con `python experiments/NOME_SCRIPT.py --help`.

Per nuove inferenze sono disponibili modalità `compute`, configurazioni GPU in `requirements/gpu.txt` e parametri nei file YAML. I pesi dei modelli, l'ambiente virtuale e le credenziali non sono nello ZIP; generare nuovi testi o score richiede download e risorse adeguate. La naturalezza remota richiede anche una chiave e l'attivazione esplicita delle chiamate. Tutto questo non è necessario per consultare i risultati o per eseguire gli esempi sopra.

Attribuzioni scientifiche e condizioni delle risorse esterne sono in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt). Per ricostruire lo ZIP della consegna: `python tools/build_delivery.py` produce `dist/Humanization.zip`.

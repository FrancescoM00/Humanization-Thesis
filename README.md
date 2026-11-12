# Humanization — codice e risultati degli esperimenti

Questo ZIP raccoglie il codice usato per la parte sperimentale della tesi, i dati e i risultati delle esecuzioni già svolte. Il lavoro studia se la riscrittura di un testo generato da un modello linguistico ne riduce la rilevabilità, quanto ne conserva il significato e come ne modifica lo stile.

**Per consultare il lavoro basta questo README e i report collegati sotto.** Non occorre installare Python o eseguire gli script per leggere i risultati. La cartella può essere aperta in Visual Studio Code; tabelle CSV e figure PDF si possono aprire anche separatamente.

## 1. Contenuto dello ZIP

```text
Humanization/
├── README.md                  Questa guida alla consegna
├── requirements.txt           Dipendenze Python
├── pyproject.toml             Configurazione del package Python
├── Humanization.code-workspace
├── .vscode/                   Configurazioni per aprire ed eseguire il progetto in VS Code
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
├── tests/                     Controlli sul codice
├── tools/                     Script per costruire lo ZIP
├── requirements/              Profili delle dipendenze CPU e GPU
├── docs/                      Registri tecnici di controllo e provenienza
└── THIRD_PARTY_NOTICES.md      Attribuzioni e riferimenti ai progetti esterni
```

I CSV MGTBench contengono testi umani e testi prodotti da diversi generatori nei domini Essay, Reuters e WritingPrompts. I testi umani servono per il confronto stilistico e per calibrare i detector. **Negli esperimenti di riscrittura si parte invece da un testo macchina e se ne valuta la versione riscritta.** Per «fonte» si intende quel testo originale: più riscritture dello stesso originale non costituiscono fonti indipendenti.

`data/processed/` conserva gli input effettivamente impiegati: non è necessario ricostruire i campioni per leggere i risultati. In particolare, `study_126/` contiene le riscritture e gli score dello studio su 126 originali; `pilot_42/` quelli delle prove preliminari su 42 originali. I file `SOURCE.json` documentano la provenienza dei dati ufficiali.

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

**Tutti i percorsi di questa tabella partono da `results/published/`.** Ogni riga identifica il tipo di esperimento; il report collegato spiega campione, metodo, risultati e limiti. I nomi tecnici delle cartelle sono mantenuti per collegare senza ambiguità codice, input e risultati.

| Esperimento | Cartella | Report e tabelle da consultare |
| --- | --- | --- |
| Confronto stilometrico iniziale umano/macchina | `02_stylometry/baseline/` | [Report](results/published/02_stylometry/baseline/report.md), `feature_effects.csv`, `classifier_metrics.csv`. |
| Stilometria con lunghezza abbinata: 300 umani e 300 macchina per ciascuno dei tre domini | `02_stylometry/length_matched/` | [Report](results/published/02_stylometry/length_matched/report.md), `features.csv`, `classifier_metrics.csv`. |
| Verifica delle quattro feature: tutte le feature, shortlist fissa, selezione sul training e quaterne casuali | `02_stylometry/shortlist/` | [Report PDF](results/published/02_stylometry/shortlist/report.pdf), `metrics.csv`, `paired_differences.csv`, `selection_frequency.csv`. |
| Robustezza della selezione escludendo cinque indicatori sensibili alla lunghezza: 105 contro 100 candidati | `02_stylometry/shortlist_length_robustness/` | [Report PDF](results/published/02_stylometry/shortlist_length_robustness/report.pdf), `metrics.csv`, `selection_comparison.csv`. |
| Validazione semantica iniziale: negazioni, entità, frasi mescolate, troncamento, cancellazione e asimmetria OpAI | `03_semantics/baseline/` | [Report](results/published/03_semantics/baseline/report.md), `metric_summary.csv` e figure PDF. |
| Sostituzioni di entità corrette dopo la revisione degli esempi | `03_semantics/entity_clean/` | [Confronto fra le due politiche](results/published/03_semantics/entity_clean/entity_policy_comparison.md), `entity_policy_summary.csv`. |
| Controllo sulle sole 34 sostituzioni di date, importi e percentuali | `03_semantics/entity_clean/numeric34/` | [Report](results/published/03_semantics/entity_clean/numeric34/report.md), `summary.csv`. |
| Controlli positivi OpAI, alterazioni sottili e confronto degli aggregatori semantici | `03_semantics/extension/` | [Confronto degli aggregatori](results/published/03_semantics/extension/aggregator_report.md), `metric_auc.csv`, `auc_by_stratum.csv`. |
| Confronto equo fra metriche: allineamento per frase, errori su più frasi, effetto della lunghezza e guardie lessicali | `03_semantics/fairness/` | [Report](results/published/03_semantics/fairness/report.md), `fair_metric_auc.csv`, `distributed_auc.csv`; analisi delle guardie in `guards_full/`. |
| Variazione dello stile dopo Polish, Paraphrase, Style Rewrite, Compress ed Expand | `04_opai_style/` | [Documento intero](results/published/04_opai_style/document/report.md) e [sole frasi modificate](results/published/04_opai_style/touched/report.md); `touched/distance_summary.csv`. |
| Prove preliminari su 42 originali: baseline 3B e DIPPER sul passaggio intero | `06_pilot_42/` | [Pilot 3B](results/published/06_pilot_42/qwen/report.md), [pilot DIPPER e confronto appaiato](results/published/06_pilot_42/dipper/paired_comparison.md). |
| Studio su 126 originali: quattro configurazioni di riscrittura, controlli nulli e quattro detector | `07_study_126/` | [Detection](results/published/07_study_126/detection_analysis.md), [semantica e stile](results/published/07_study_126/semantic_style.md), `detection_overall.csv`, `nli_primary_summary.csv`, `style_summary.csv`. |
| Scelta della modalità NLI su controlli positivi, alterazioni e identità | `08_nli_calibration/calibration/` | [Report](results/published/08_nli_calibration/calibration/report.md), `candidate_summary.csv`, `auc_by_condition.csv`. |
| Estensione mirata delle alterazioni di cautela, con soglia storica invariata | `hedging/` | [Report PDF](results/published/hedging/report.pdf), `metrics.csv`, `directional_summary.csv`. |

Nello studio su 126 originali, 42 erano già presenti nei pilot e 84 sono aggiuntivi. Le quattro configurazioni di riscrittura sono la baseline Qwen 3B, Qwen 14B, DIPPER sul passaggio intero e DIPPER a finestre di tre frasi; l'identità serve da riferimento. Le annotazioni indipendenti richieste ai colleghi non sono ancora incluse: le revisioni dell'assistente sono indicate come tali e non valgono come giudizi umani indipendenti.

Nei risultati di detection, **TPR** è la quota di testi macchina rilevati, **FPR** la quota di umani erroneamente rilevati e **ASR** la quota di originali già rilevati che, dopo riscrittura, eludono il detector. Nelle analisi delle feature e delle metriche semantiche, **AUC** misura la capacità di separare i due gruppi dichiarati nel report.

I file `report.md` sono leggibili in VS Code con l'anteprima Markdown; dove disponibile, `report.pdf` contiene la versione impaginata. I CSV riportano i valori numerici, i PDF delle figure mostrano i grafici e i JSON conservano parametri e informazioni dell'esecuzione. `predictions.npz` contiene predizioni numeriche compresse. Le coppie e le riscritture in formato JSONL hanno una registrazione per riga.

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

Si ottengono `report.md`, `metrics.csv`, `paired_differences.csv` e le tabelle delle selezioni.

**Rianalizzare i controlli positivi e le alterazioni semantiche** dagli score salvati:

```bash
python experiments/02_semantic_validation.py --mode reproduce --output results/new_runs/esempio_semantica
```

Questi esempi usano CPU e dati inclusi: ricalcolano le analisi senza eseguire nuovamente i modelli. Gli output vanno in `results/new_runs/` e non sostituiscono i risultati già consegnati. Le opzioni di ogni script si consultano con `python experiments/NOME_SCRIPT.py --help`.

Per nuove inferenze sono disponibili modalità `compute`, configurazioni GPU in `requirements/gpu.txt` e parametri nei file YAML. I pesi dei modelli, l'ambiente virtuale e le credenziali non sono nello ZIP; generare nuovi testi o score richiede download e risorse adeguate. La naturalezza remota richiede anche una chiave e l'attivazione esplicita delle chiamate. Tutto questo non è necessario per consultare i risultati o per eseguire gli esempi sopra.

Attribuzioni scientifiche e condizioni delle risorse esterne sono in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Per ricostruire lo ZIP della consegna: `python tools/build_delivery.py` produce `dist/Humanization.zip`.

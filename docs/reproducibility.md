# Riproduzione ed esecuzione

Il progetto offre due percorsi distinti: ricalcolare le analisi a partire dai punteggi inclusi oppure produrre nuovi testi e score con i modelli. Il primo serve a verificare le tabelle e non richiede di ripetere le inferenze.

## 1. Preparare il progetto locale

Estrarre lo ZIP e aprire la cartella `Humanization/` in VS Code. Creare una `.venv` con Python 3.12 e installare `requirements.txt`, come nel [README](../README.md). L'installazione delle dipendenze richiede accesso alle rispettive fonti. I pesi dei modelli non sono nello ZIP.

Per una macchina con GPU installare un PyTorch CUDA compatibile con l'hardware e usare il profilo `requirements/gpu.txt`. Non sostituire una build CUDA funzionante con il profilo CPU. Le opzioni di esecuzione e il dispositivo sono espliciti nei file YAML e nei comandi.

In VS Code selezionare l'interprete della `.venv`. Task e debugger usano l'interprete selezionato e la cartella di lavoro del progetto; non contengono percorsi assoluti della macchina dello sviluppatore.

## 2. Verificare i dati

```bash
python experiments/00_data.py --config configs/datasets.yaml --mode reproduce
```

I CSV ufficiali MGTBench sono in `data/raw/mgtbench/`. I loro byte e l'ordine delle righe devono restare invariati: gli identificatori e i campioni congelati dipendono da questi dati. Pulizia, matching e nuove preparazioni hanno output separati. La directory `data/raw/opai/` ospita i file originali OpAI; le coppie congelate utilizzate dalle analisi sono sotto `data/processed/`.

## 3. Riprodurre le tabelle già ottenute

```bash
python experiments/07_evaluation.py --config configs/evaluation.yaml --mode reproduce
```

La modalità usa i punteggi individuali congelati, ricalcola le metriche e le confronta con `results/published/`. Gli output sono in `results/new_runs/`. Il confronto appaiato dei pilot usa 42 fonti; lo studio ne comprende 126 e distingue anche le 84 fonti aggiuntive.

La riproduzione non chiama i riscrittori, non ricalibra le soglie dei detector e non esegue il giudice di naturalezza. I valori di riferimento sono usati per il confronto finale, non copiati come se fossero nuovi calcoli. Consultare il manifest e i controlli numerici prodotti dall'esecuzione per conoscere file, parametri e differenze.

Le altre entry point consentono di riesaminare le singole componenti. Le opzioni effettive di ogni comando sono disponibili con `--help`:

```bash
python experiments/01_stylometry.py --help
python experiments/02_semantic_validation.py --help
python experiments/03_opai_editing.py --help
```

## 4. Eseguire un nuovo esperimento con i modelli

I file YAML dichiarano quali dati leggere, quali preset applicare e dove scrivere. Per un confronto fra riscrittori, il flusso è:

1. Preparare una coorte e mantenere fissi gli identificatori di fonte.
2. Eseguire `04_rewriting.py` per produrre le riscritture.
3. Eseguire la detection con `05_detection.py` e le misure semantiche con `02_semantic_validation.py` sugli stessi originali e sulle nuove riscritture. Estrarre anche le feature stilometriche richieste.
4. Passare gli score ottenuti a `07_evaluation.py` per le metriche e i confronti finali.
5. Eseguire eventualmente `06_naturalness.py` come valutazione aggiuntiva.

Prima del nuovo run verificare che i percorsi YAML colleghino queste fasi: gli scorer devono leggere le riscritture appena prodotte, e l'evaluation gli score appena calcolati. Una modalità di replay degli output salvati non va presentata come una nuova inferenza; il manifest deve dichiararlo. Non si riutilizza la cache di uno score per un testo differente.

L'analisi delle riscritture OpAI con `03_opai_editing.py` è invece un esperimento di validazione su un dataset di editing: non è una fase necessaria per generare con Qwen o DIPPER.

## Risorse necessarie

| Attività | Risorse |
| --- | --- |
| Controllo dei CSV e ricalcolo delle metriche | CPU; nessun checkpoint necessario |
| Estrazione stilometrica | CPU, spaCy piccolo e dati linguistici richiesti |
| Similarità e NLI | Checkpoint locali o download; CPU supportata, GPU utile per tempi inferiori |
| Qwen 14B e DIPPER XXL | Memoria considerevole; configurazioni e precisione devono essere adatte a GPU/RAM disponibili |
| Giudice di naturalezza remoto | Connessione al provider e chiave in una variabile d'ambiente, soltanto per richieste esplicite |

L'ambiente di lavoro usato per verificare questa consegna disponeva di circa 32 GB di RAM. Questa disponibilità non è una misura del requisito minimo di ogni script. La memoria dei modelli dipende anche da precisione, quantizzazione, batch e lunghezza: una GPU non compatibile con il preset non rende scientificamente equivalente un cambio automatico di modello.

Per la naturalezza configurare `GROQ_API_KEY` nell'ambiente locale quando necessario. Non scrivere la chiave negli YAML o nei file versionati. Le richieste reali richiedono l'opzione esplicita `--execute`; preparare un piano o valutare risposte salvate non richiede la chiave.

## Verifiche e limiti

```bash
python -m pytest -q
```

Le verifiche del refactoring e delle tabelle vanno riferite alla versione effettivamente consegnata. Un controllo sintattico non equivale a una nuova esecuzione dei modelli. Un buon risultato su perturbazioni sintetiche non sostituisce la validazione su riscritture reali e annotazioni umane.

`results/published/` conserva i risultati di riferimento; `results/new_runs/` contiene le nuove esecuzioni. Le vecchie strutture e gli hash storici restano identificabili nell'archivio e nella cronologia. Il repository originale è separato e non viene modificato da questi comandi.

## Comandi dei singoli moduli

```bash
python experiments/01_stylometry.py --mode reproduce
python experiments/02_semantic_validation.py --mode reproduce
python experiments/03_opai_editing.py --mode reproduce
python experiments/04_rewriting.py --mode replay
python experiments/05_detection.py --mode replay
python experiments/06_naturalness.py --mode reproduce
python experiments/07_evaluation.py --mode reproduce
```

Il replay conserva i testi e gli score storici; il controllo degli hash rifiuta score di testi diversi. Per un nuovo confronto sui testi prodotti ora, dopo aver predisposto i checkpoint e la GPU configurata:

```bash
python experiments/04_rewriting.py --mode compute
python experiments/05_detection.py --mode compute
python experiments/02_semantic_validation.py --mode compute --download-models
python experiments/01_stylometry.py --mode compute --analysis rewrites
python experiments/07_evaluation.py --mode compute
```

Questi preset condividono `results/new_runs/custom_study`. Non si devono usare output di un altro run. La calibrazione NLI e il controllo aggiuntivo sulle cautele si ricalcolano con `07_evaluation.py --mode reproduce --analysis nli_calibration --output results/new_runs/nli_calibration` e `--analysis hedging --output results/new_runs/hedging`.

Per preparare il giudizio di naturalezza usare `06_naturalness.py --mode prepare`: produce il piano senza richieste. Gli eventuali fallimenti dei riscrittori sono registrati separatamente; la politica esplicita `report_and_exclude` limita i giudizi alle coppie con entrambe le versioni disponibili. Per effettuare richieste reali impostare `enabled: true`, configurare `GROQ_API_KEY` nell'ambiente ed eseguire `--mode compute --execute`. I giudizi non sono etichette umane.

I Parquet ufficiali OpAI sono già inclusi in `data/raw/opai/official_parquet`; CMUdict è in `data/resources/nltk_data`, con le attribuzioni originali. I pesi restano esterni: una nuova inferenza completa può richiedere molta memoria, decine di GB di cache e tempi lunghi. La nuova esecuzione completa dei modelli non è stata effettuata durante il refactoring.

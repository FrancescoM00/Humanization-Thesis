# Humanization

Progetto Python per valutare le riscritture di testi generati da modelli linguistici. La cartella può essere estratta dallo ZIP e aperta direttamente in **Visual Studio Code**. Comprende codice, configurazioni, CSV originali e dati intermedi per riprodurre le analisi già eseguite.

Il flusso è: **dati → riscrittura → score individuali di stile, semantica e detection → evaluation → risultati**. La naturalezza è un modulo opzionale. La produzione di testi e score è separata dal calcolo delle metriche finali.

## Aprire ed eseguire il progetto

1. Estrarre lo ZIP in una cartella, per esempio `Humanization/`.
2. In VS Code scegliere **File → Apri cartella** e selezionarla, oppure aprire `Humanization.code-workspace`.
3. Installare le estensioni Python e Python Debugger suggerite. Usare **Python 3.12**.
4. Aprire il terminale nella cartella del progetto e creare l'ambiente:

```bash
python -m venv .venv
```

Attivarlo con `source .venv/bin/activate` su Linux/macOS oppure `.venv\Scripts\Activate.ps1` in PowerShell. Poi:

```bash
python -m pip install -r requirements.txt
```

In **Python: Select Interpreter** scegliere l'interprete della `.venv` appena creata. I comandi seguenti partono dalla radice del progetto:

```bash
python experiments/00_data.py --config configs/datasets.yaml --mode reproduce
python experiments/07_evaluation.py --config configs/evaluation.yaml --mode reproduce
```

Il secondo comando ricalcola le metriche dai punteggi individuali salvati, confrontandole con i risultati pubblicati. Non esegue nuovamente riscrittori o detector e non effettua chiamate al giudice. In VS Code si può anche scegliere il task **Humanization: riproduci evaluation** oppure avviare la corrispondente configurazione con F5.

## Organizzazione

| Cartella | Funzione |
| --- | --- |
| `configs/` | Parametri YAML, percorsi di input/output e preset dei modelli |
| `data/raw/mgtbench/` | I tre CSV originali, conservati senza modifiche |
| `data/raw/opai/` | File originali OpAI quando disponibili o scaricati |
| `data/processed/` | Campioni, coppie e punteggi intermedi congelati |
| `src/humanization/` | Package con dataset, stilometria, semantica, riscrittura, detection, naturalezza ed evaluation |
| `experiments/` | Comandi che collegano configurazioni e moduli |
| `results/published/` | Risultati di riferimento già ottenuti |
| `results/new_runs/` | Output delle nuove esecuzioni |
| `docs/` | Metodo, flusso e istruzioni di riproduzione |
| `archive/legacy_experiments/` | Materiale storico, separato dal percorso operativo |
| `tests/` | Verifiche del codice e dei protocolli |

La [guida all'organizzazione](docs/organization.md) descrive quali moduli producono gli score e quali calcolano le metriche.

## Esperimenti e flusso completo

| Comando | Scopo |
| --- | --- |
| `00_data.py` | Verificare e preparare i dati |
| `01_stylometry.py` | Estrarre feature e valutare i segnali stilistici |
| `02_semantic_validation.py` | Calcolare e verificare le misure semantiche |
| `03_opai_editing.py` | Studiare le modifiche OpAI, anche nelle porzioni riscritte |
| `04_rewriting.py` | Produrre riscritture con i preset configurati |
| `05_detection.py` | Produrre score dei detector per originali e riscritture |
| `06_naturalness.py` | Preparare, eseguire o valutare confronti di naturalezza |
| `07_evaluation.py` | Aggregare score, calcolare ASR, intervalli e confronti |

`--mode reproduce` utilizza gli input congelati; le modalità di calcolo e riproduzione degli output salvati sono indicate da `--help`. Per una nuova esecuzione, i percorsi YAML devono collegare gli output della riscrittura agli input degli scorer e poi a quelli di evaluation. La [guida di riproduzione](docs/reproducibility.md) spiega l'ordine e i requisiti.

La riproduzione delle tabelle usa CPU e punteggi inclusi. Eseguire nuovamente i modelli richiede i checkpoint e risorse adeguate; i pesi non sono nello ZIP. Per GPU usare `requirements/gpu.txt` con un PyTorch CUDA compatibile. La naturalezza tramite servizio esterno richiede una chiave in una variabile d'ambiente e un'esecuzione esplicita: non è necessaria per il percorso di riproduzione principale.

Il [metodo](docs/methodology.md) distingue etichette intenzionali, revisioni dell'assistente e annotazioni umane. Il repository originale rimane separato; questa riorganizzazione non modifica i risultati storici. Attribuzioni e termini delle fonti: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Consegna ZIP

Da questa repository si può scaricare lo ZIP dal pulsante **Code → Download ZIP**. Per generare una cartella radice chiamata esattamente `Humanization/`, eseguire `python tools/build_delivery.py`: crea `dist/Humanization.zip`, includendo anche i dati ufficiali OpAI e la risorsa linguistica CMUdict. Pesi, ambiente virtuale, credenziali e nuovi run rimangono fuori dall'archivio.

Verifica di questa versione: **209 test passati, 1 saltato perché richiede CUDA**. La riproduzione dello studio controlla 209 colonne numeriche e quella del pilot 15, con tolleranza 1e-12. I comandi di riproduzione sono stati eseguiti senza nuovi modelli o richieste al provider.

Lo ZIP completo è di circa **160 MiB** perché comprende anche i Parquet ufficiali OpAI e gli input congelati. Una nuova esecuzione dei modelli richiede ulteriori download di checkpoint, non inclusi in questa dimensione.

# Humanization — codice della sezione sperimentale

Questa repository accompagna la **sezione sperimentale della tesi** e ne segue l'ordine. Contiene script, dati d'ingresso, risultati e istruzioni per studiare tre aspetti delle riscritture di testi macchina: **rilevabilità, conservazione del significato e cambiamento stilistico**.

È una consegna separata da [Humanization](https://github.com/FrancescoM00/Humanization), che conserva l'intero lavoro precedente. La riorganizzazione parte dal commit [`d2868b3`](https://github.com/FrancescoM00/Humanization/tree/d2868b323938a2a9371c2a872ad86b5596d0f097). Il repository originale rimane invariato: per tornare alla vecchia organizzazione basta riaprire quello. La valutazione della naturalezza è esclusa dalla consegna attuale.

## Da dove iniziare

Ogni guida spiega obiettivo, input, comandi, output, risultati e risorse richieste. Tutti i comandi si eseguono dalla radice del progetto.

| PDF | Esperimento |
| --- | --- |
| §1.2 | [01 — Dataset e controllo della lunghezza](experiments/01_dataset/README.md) |
| §1.3 | [02 — Stilometria, shortlist e robustezza](experiments/02_stilometria/README.md) |
| §1.4 | [03 — Conservazione semantica e perturbazioni](experiments/03_conservazione_semantica/README.md) |
| §1.5 | [04 — Stile delle porzioni OpAI riscritte](experiments/04_stile_opai/README.md) |
| §1.6 | [05 — Riscrittori e detector](experiments/05_riscrittori_detector/README.md) |
| §1.7 | [06 — Pilot su 42 fonti](experiments/06_pilot_42/README.md) |
| §1.8 | [07 — Studio su 126 fonti](experiments/07_studio_126/README.md) |
| §1.9 | [08 — Calibrazione NLI contestuale](experiments/08_calibrazione_nli/README.md) |

La [mappa dettagliata](docs/MAPPATURA_TESI.md) collega sezioni, script e tabelle. L'ultima estensione sulle cautele, successiva al PDF di riferimento, è un [supplemento separato](supplements/hedging/README.md).

## Dati ufficiali inclusi

In [`data/raw/`](data/raw) sono presenti le tre copie originali fornite dall'autore: `Essay_LLMs.csv`, `Reuters_LLMs.csv` e `WP_LLMs.csv`. Sono identiche byte per byte ai dati degli esperimenti. Il caricamento principale è locale; i CSV restano inalterati e la pulizia produce file separati.

Il controllo dei dati non richiede dipendenze esterne:

```bash
python scripts/import_mgtbench.py --verify-only
```

Provenienza, importazione e distinzione da OpAI sono spiegate nella [guida dati](data/README.md).

## Installazione

Ambiente di riferimento verificato: **Python 3.12 su Linux**. Per stilometria e rianalisi delle tabelle, da Linux/macOS:

```bash
git clone https://github.com/FrancescoM00/Humanization-Thesis.git
cd Humanization-Thesis
bash scripts/setup.sh style
source .venv/bin/activate
```

In PowerShell, dalla cartella del progetto:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements\style-lock.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe -m nltk.downloader -q -e -d .venv\nltk_data cmudict
.venv\Scripts\Activate.ps1
```

Il setup installa anche il modello spaCy piccolo. Le nuove inferenze semantiche su CPU usano il profilo `bash scripts/setup.sh semantic` con PyTorch CPU. Per i riscrittori e Binoculars su GPU seguire invece i [notebook Kaggle](notebooks/README.md), che conservano il PyTorch CUDA della piattaforma. I checkpoint vengono scaricati quando necessari; non sono inclusi nella repository.

## Verificare i risultati senza rilanciare i modelli

Le tabelle e le figure consegnate sono in `results/`. I punteggi e le coppie necessari alla loro rianalisi sono in `inputs/`. Per esempio:

```bash
python scripts/analyze_humanizer_results.py study --run inputs/study_126 --pilot-cohort inputs/study_126/cohorts/pilot.jsonl --output runs/study_126 --reference results/07_study_126
python scripts/analyze_humanizer_results.py pilot --qwen-run inputs/pilot_42/qwen --dipper-run inputs/pilot_42/dipper --output runs/pilot_42 --reference results/06_pilot_42/dipper
```

Questi comandi usano i punteggi salvati e scrivono sotto `runs/`. Le guide distinguono questo percorso dalla nuova estrazione delle feature e dalle inferenze LLM, più costose. Gli output rigenerati non sostituiscono i risultati storici.

Il codice condiviso è in `src/`, i comandi eseguibili in `scripts/`, i preset in `configs/`. Le verifiche della migrazione e della riproduzione sono in [VERIFICA_CONSEGNA.md](docs/VERIFICA_CONSEGNA.md). Per i test:

```bash
python -m pytest -q
```

Le etichette delle perturbazioni e le revisioni dell'assistente non sono annotazioni umane indipendenti; le reference umane dello studio sono ancora in raccolta. I manifest storici mantengono la provenienza originale, documentata nella [migrazione](docs/MIGRAZIONE.md). Le attribuzioni del codice e i termini delle fonti sono in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

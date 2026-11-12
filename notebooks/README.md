# Notebook di riproduzione su Kaggle

Questi notebook eseguono le tecniche di riscrittura e le misure descritte nel PDF. I risultati già calcolati si trovano in `results/`: per leggerli o ricalcolarne i riepiloghi non occorre avviare i modelli.

| Parte del PDF | File | Scopo |
| --- | --- | --- |
| §§1.6–1.7 | [Humanization_Pilot.ipynb](Humanization_Pilot.ipynb) e [guida](README_Pilot.md) | Pilot di 42 fonti, baseline 3B e DIPPER, detector supervisionati, semantica e stile |
| §1.8 | [Humanization_Study_v2.ipynb](Humanization_Study_v2.ipynb) e [guida](README_Study_v2.md) | Quattro riscrittori sulle stesse 126 fonti, Binoculars e controlli nulli |
| §1.9 | [run_nli_calibration.py](run_nli_calibration.py) e [guida](README_NLI_Calibration.md) | Confronto fra le configurazioni NLI sul pannello di calibrazione congelato |

## Prima di eseguire

Su Kaggle importare il file `.ipynb`, abilitare Internet e selezionare la GPU. Per lo studio con Binoculars servono **due T4**: i due Falcon 7B vengono caricati separatamente, uno su ciascuna GPU. Il pilot con un solo riscrittore può usare una GPU.

I notebook clonano `FrancescoM00/Humanization-Thesis` e registrano il commit effettivo. `REPO_REF="main"` può essere sostituito con il commit da riprodurre. Se il codice o la configurazione cambiano rispetto a un run già iniziato, usare un nuovo `RUN_SUFFIX`: non sovrascrivere risultati precedenti. Il checkout deve essere accessibile dalla sessione Kaggle.

I tre CSV ufficiali sono letti da `data/raw` nel checkout. Per usare una cartella caricata come Kaggle Dataset, prima della sezione dei dati impostare:

```python
os.environ["MGTBENCH_RAW_DIR"] = "/kaggle/input/nome-dataset"
```

La directory deve contenere direttamente `Essay_LLMs.csv`, `Reuters_LLMs.csv` e `WP_LLMs.csv`. Il controllo degli hash e degli identificatori ricostruisce lo stesso campione degli esperimenti; i CSV non vengono modificati o scaricati da un mirror.

## Risorse e tempi

La prova `MODE="smoke"` usa tre fonti e verifica l'esecuzione tecnica. Non produce una stima affidabile della qualità. Il pilot usa 42 fonti; lo studio ne usa 126 più 300 umani di calibrazione. Una sessione completa include download, riscritture e scoring, quindi richiede diverse ore e dipende dalla cache e dalla quota GPU disponibile. Non è un'elaborazione da lanciare per consultare una tabella già disponibile.

DIPPER XXL richiede circa 45 GB di download; Qwen 14B e i due Falcon hanno checkpoint ulteriori. Il caricamento NF4 riduce la memoria GPU dei riscrittori, senza ridurre nella stessa proporzione i file da scaricare. Venv e pesi rimangono fuori dalle cartelle dei risultati e dagli ZIP.

## Output da conservare

Le cartelle sotto `/kaggle/working` contengono coorti, configurazioni, riscritture, fallimenti, punteggi, soglie, metadati e moduli di annotazione semantica. L'ultima cella produce uno ZIP. Scaricarlo prima di chiudere la sessione: la disponibilità futura delle cache temporanee non è garantita.

Le cartelle originali del repository precedente non vengono toccate da questi notebook: clone e nuove destinazioni usano nomi distinti. La directory dei risultati di ogni nuovo run è separata dai risultati congelati consegnati con il progetto.

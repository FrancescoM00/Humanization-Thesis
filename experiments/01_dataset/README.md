# 01 — Dataset e preparazione

**Riferimento: §1.2 del PDF.** Questa fase prepara testi umani e macchina in inglese e mantiene riconoscibili i gruppi di origine. Le coppie della stessa riga non sono necessariamente parafrasi equivalenti.

## Input

I tre CSV originali sono in [`data/raw`](../../data/raw): `Essay_LLMs.csv`, `Reuters_LLMs.csv`, `WP_LLMs.csv`. Ciascuno contiene 1.000 righe, un testo umano e versioni di sette generatori. Reuters comprende anche due colonne di autore. Questi file sono le copie ufficiali fornite dall'autore; coincidono byte per byte con quelli degli esperimenti pubblicati.

Controllo locale, dalla radice del progetto:

```bash
python scripts/import_mgtbench.py --input-dir data/raw --verify-only
```

Per importare nuovamente le copie originali da un'altra cartella:

```bash
python scripts/import_mgtbench.py --input-dir /percorso/csv_ufficiali --output data/raw
```

I CSV restano inalterati. La pulizia avviene durante la preparazione: spaziatura normalizzata, controllo delle code incomplete, lunghezza minima, duplicati e gruppi collegati. Ordine delle righe e identificatori originali non vanno cambiati.

## Preparare i campioni

Con l'ambiente stilometrico e `en_core_web_sm` installati, i due comandi seguenti preparano il campione e completano l'analisi di [§1.3](../02_stilometria/README.md):

```bash
python scripts/run_style_experiment.py --raw-dir data/raw --output runs/baseline --pairs 300 --seed 42
python scripts/run_length_matched_style.py --raw-dir data/raw --output runs/length_matched --pairs 300 --seed 42
```

Si ottengono 300 coppie per dominio: **900 coppie, 1.800 testi**, con un solo generatore macchina assegnato a ciascuna coppia. Umani e versioni collegate restano nello stesso gruppo di validazione.

Il secondo comando conserva prefissi a fine frase fra 90 e 180 parole lessicali, vicini a 150, con conteggio esattamente uguale nella coppia. Le medie umani/macchine coincidono: 150,053 in Essay, 146,607 in Reuters e 150,240 in WritingPrompts. Il classificatore che usa soltanto il numero di parole ha AUC 0,500. Caratteri e numero di frasi possono ancora differire.

Le cache di preparazione sono sotto `data/`; tabelle, manifest e report del nuovo run sono sotto `runs/`. I risultati storici da consultare sono in [`results/02_stylometry`](../../results/02_stylometry).

## OpAI-Bench

Il secondo dataset contiene versioni Polish, Paraphrase, Style Rewrite, Compress ed Expand. Serve a costruire confronti semantici e a misurare lo stile delle frasi effettivamente riscritte. La preparazione è descritta nelle guide [semantica](../03_conservazione_semantica/README.md) e [stile OpAI](../04_stile_opai/README.md). I tre CSV MGTBench non sostituiscono i Parquet OpAI.

Questa fase usa CPU; non richiede GPU. I manifest storici conservano hash e percorsi del repository originale al commit `d2868b3`; la nuova organizzazione non riscrive la provenienza degli esperimenti.

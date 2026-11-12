# 04 — Stile nelle riscritture OpAI

**Riferimento: §1.5 del PDF.** Si misura come cambiano le feature stilometriche fra versioni adiacenti: Polish, Paraphrase, Style Rewrite, Compress ed Expand. L'analisi distingue l'intensità della modifica nelle frasi toccate dalla sua copertura nel documento.

## Input e risultati già disponibili

Il campione comprende **200 traiettorie**, **188 originali distinti** e **1.000 coppie adiacenti**. Lo stesso originale può avere più editor; gli editor vengono mediati prima dell'inferenza statistica.

- [Documento intero](../../results/04_opai_style/document/report.md): `paired_feature_values.csv`, `summary.csv`, `length_changes.csv` e controllo con prefissi di lunghezza identica.
- [Frasi toccate](../../results/04_opai_style/touched/report.md): `touched_feature_values.csv`, `aligned_sentence_distances.csv`, `distance_summary.csv`, `ordinal_summary.csv` e confronto appaiato con il documento intero.

Le 105 feature sono quelle di `src/stylometry/features.py`. Il codice specifico è in `scripts/run_opai_style.py`, `scripts/run_opai_touched_style.py` e `src/semantics/touched_style.py`. Le tabelle incluse possono essere lette senza caricare modelli o scaricare OpAI.

## Ripetere estrazione e analisi

Servono l'ambiente stilometrico (`requirements/style.txt`), spaCy `en_core_web_sm`, textstat e `pyarrow` per i Parquet. È un esperimento **CPU**, senza modelli generativi o detector.

Il caricamento richiede i 16 Parquet ufficiali OpAI e `records.provenance.json` nella directory `data/opai/`. Per recuperarli alla revisione usata nel progetto:

```bash
python scripts/download_opai.py \
  --revision 072c03c7051409346aaba05168d183cf6d863ee9 \
  --output data/opai/records.jsonl
```

Poi, dalla radice:

```bash
python scripts/run_opai_style.py \
  --opai-dir data/opai --data-dir data/reruns/opai_style \
  --output runs/04_opai_style/document --n 200 --seed 42 --bootstrap 1000

python scripts/run_opai_touched_style.py \
  --pairs data/reruns/opai_style/pairs.jsonl --opai-dir data/opai \
  --split-pairs inputs/semantic_extension/positive_pairs.jsonl \
  --data-dir data/reruns/opai_touched_style \
  --output runs/04_opai_style/touched --seed 42 --bootstrap 1000
```

Il primo comando costruisce le coppie e le feature su documenti interi; il secondo recupera `sentences` e `sentence_labels` native e verifica la corrispondenza delle frasi fra versioni. Il file `--split-pairs` conserva gli split di calibrazione/test già adottati: non vengono ricreati guardando i risultati. Aggiungere `--reuse-features` per riprendere una cache della stessa esecuzione. Le nuove elaborazioni restano in `data/reruns/` e `runs/`.

## Risultati da confrontare con il PDF

Le scale della distanza RMS sono apprese soltanto dalle frasi native di calibrazione. La misura primaria usa **91 dimensioni** standardizzate e le sole frasi marcate; il test comprende **103 traiettorie su 99 fonti**.

| Distanza media nel test | Polish | Paraphrase | Style Rewrite |
| --- | ---: | ---: | ---: |
| RMS delle frasi marcate, primaria | 0,366 | 0,717 | 0,656 |
| RMS pesata per la copertura, diagnostica | 0,152 | 0,369 | 0,425 |

L'ordine Polish < Paraphrase < Style non emerge dalla misura primaria: Style−Paraphrase è −0,0615, IC 95% [−0,1257; 0,0157]. La diagnostica per copertura restituisce l'ordine delle medie, ma risente del numero crescente di frasi coinvolte. Non sostituisce la metrica primaria dopo aver visto l'esito.

La diluizione documentale è visibile nelle stesse coppie: per Paraphrase, ΔMATTR è circa **+0,0220 nelle porzioni cambiate**, contro **+0,00747 nel documento**, su 92 coppie e 88 fonti. MATTR resta indefinita sotto 50 parole. Un cambiamento maggiore delle feature descrive una riscrittura più distante su quelle dimensioni; non certifica un miglioramento qualitativo.

I manifest e i risultati storici conservano i riferimenti al repository originale, commit `d2868b3`. La riorganizzazione non cambia il campione, le definizioni delle feature o i risultati riportati.

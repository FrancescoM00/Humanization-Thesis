# 07 — Studio esteso su 126 fonti

**Riferimento: §1.8 del PDF.** Si confrontano quattro preset sugli stessi originali: Qwen 3B, Qwen 14B, DIPPER passaggio e DIPPER a finestre di tre frasi. L'identità è un controllo aggiuntivo.

Le 126 fonti comprendono le 42 del pilot e **84 aggiuntive**; i risultati sulle 84 sono riportati separatamente. Ogni combinazione dominio–generatore contiene sei fonti. Le versioni dello stesso originale non aumentano il numero di osservazioni indipendenti.

## Risultati pubblicati

Sono disponibili 630 tentativi, dei quali 626 validi e quattro troncati. I fallimenti restano espliciti; non diventano evasioni riuscite. Semantica e stile descrivono i soli output validi.

| Preset | ASR OpenAI | ASR Fakespot | ASR RADAR | ASR Binoculars |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3B | 45/73 | 1/109 | 8/90 | 17/120 |
| Qwen 14B | 60/73 | 1/109 | 8/90 | 29/120 |
| DIPPER passaggio | 10/73 | 11/109 | 7/90 | 0/120 |
| DIPPER tre frasi | 6/73 | 8/109 | 11/90 | 1/120 |

Il successo dipende dal detector. Sulle 84 fonti nuove Fakespot rileva tutti gli output validi dei due preset Qwen; su Binoculars il 14B evade 21/80 originali inizialmente rilevati. Il risultato non dimostra un effetto della sola dimensione del modello: 3B e 14B differiscono anche per precisione.

I controlli nulli confrontano le riscritture con prefissi della stessa lunghezza e con un rumore lessicale minimo, usando i medesimi sottoinsiemi eleggibili. Non si confrontano percentuali con denominatori diversi. I report, i punteggi e le distribuzioni sono in [`results/07_study_126`](../../results/07_study_126).

## Rigenerare le tabelle senza modelli

Coorti, riscritture e score individuali congelati sono in `inputs/study_126`. Per ricalcolare gli aggregati del pilot incluso, delle 84 fonti nuove e del campione completo:

```bash
python scripts/analyze_humanizer_results.py study --run inputs/study_126 --pilot-cohort inputs/study_126/cohorts/pilot.jsonl --output runs/reproduced_study_126 --reference results/07_study_126
```

Il comando produce le tabelle detector, i riepiloghi semantici e stilometrici, i confronti appaiati e il controllo rispetto agli aggregati pubblicati. Non carica checkpoint e non effettua generazioni o chiamate API. I conteggi, gli score medi e le decisioni devono essere distinti dagli intervalli: se una procedura bootstrap storica non è ricostruibile esattamente, il confronto la segnala e documenta quella esplicita usata nella riproduzione.

La misura NLI primaria attuale è `window2 + min`, selezionata in una fase di calibrazione successiva descritta nella [guida 08](../08_calibrazione_nli/README.md). La rielaborazione degli score già disponibili non è una preregistrazione retroattiva. Non viene trasferita la soglia storica 0,04075 ai veri riscrittori.

## Ripetere generazioni e inferenze

Usare [`notebooks/Humanization_Study_v2.ipynb`](../../notebooks/Humanization_Study_v2.ipynb), `MODE = "study"` e `configs/study_v2.json`. Il notebook usa il pannello congelato e mantiene gli output sotto una nuova directory. La [guida operativa](../../notebooks/README_Study_v2.md) descrive installazione, GPU, cache e salvataggio.

Il run originario usava due Tesla T4; 14B e DIPPER richiedono memoria e download consistenti, e Binoculars distribuisce observer e performer su due GPU. Per ricalcolare le sole tabelle dai file inclusi basta la CPU.

Le annotazioni umane sulle 84 coppie reference sono ancora in raccolta. I punteggi automatici, i proxy di sviluppo e le letture dell'assistente non vanno presentati come etichette umane di conservazione semantica.

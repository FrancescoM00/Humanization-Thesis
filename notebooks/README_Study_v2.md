# Studio su 126 fonti — PDF §1.8

Notebook: [Humanization_Study_v2.ipynb](Humanization_Study_v2.ipynb). Prima leggere la [guida generale](README.md).

## Obiettivo e condizioni

Si applicano quattro configurazioni alle medesime 126 fonti, sei per ogni combinazione fra dominio e generatore. Le 42 fonti del pilot sono incluse; le 84 aggiuntive sono analizzate anche separatamente. I 300 umani di calibrazione rimangono disgiunti da tutto il pannello.

[`configs/study_v2.json`](../configs/study_v2.json) conserva:

- Qwen2.5-3B-Instruct, prompt fisso e generazione greedy in FP16;
- Qwen2.5-14B-Instruct con lo stesso prompt, greedy e quantizzazione NF4;
- DIPPER XXL NF4 sull'intero passaggio, diversità lessicale 60 e d'ordine 0;
- DIPPER XXL con finestre di tre frasi e prefisso cumulativo delle porzioni già riscritte;
- identità, controllo che restituisce l'originale senza riscriverlo.

Il preset a tre frasi non applica DIPPER tre volte all'intero testo. Cambiano segmentazione, contesto e budget totale rispetto al preset sull'intero passaggio.

## Esecuzione

Selezionare **T4 ×2** e Internet. Iniziare con `MODE="smoke"` e verificare gli output. Il run esteso usa:

```python
MODE = "study"
RUN_SUFFIX = "_thesis_study"
ACTIVE_SYSTEMS = ["identity", "qwen_humanize", "qwen14b_humanize", "dipper", "dipper_context3"]
RUN_BINOCULARS = True
```

Il notebook controlla che la coorte ricostruita dai CSV ufficiali coincida con `inputs/study_126/study_126_protocol_v2.jsonl`. Sono attesi 126 originali di test, 300 umani di calibrazione e 630 esiti fra i cinque sistemi, con eventuali fallimenti espliciti. Le consegne di scrittura presenti fra gli input rimangono nel risultato primario; il modulo del tipo di fonte usa soltanto l'originale.

Le sezioni eseguono riscritture, controlli nulli, tre classificatori RoBERTa, Binoculars, caratteristiche di stile, metriche semantiche storiche, NLI contestuale, esportazione e preparazione delle annotazioni semantiche. I controlli nulli sono prefissi con conteggio alfabetico abbinato, rumore lessicale minimo e loro combinazione. Binoculars mantiene un Falcon per GPU in FP16; non viene incluso nel voto dei tre RoBERTa.

## Risorse e cache

Il codice distingue gli asset leggibili dalle nuove scritture. Può riusare una venv con dipendenze compatibili e checkpoint fissati già presenti in `/kaggle/temp/humanization_runtime`, anche se la directory è di sola lettura. Clone, nuovi risultati e temporanei sono sotto `/kaggle/working`.

`REUSE_CACHED_MODELS=None` rileva automaticamente il vecchio runtime non scrivibile. In modalità di solo riuso, un checkpoint mancante provoca un errore esplicito. `REUSE_CACHED_MODELS=False` permette nuovi download soltanto in una cache scrivibile. Non cancellare il run o sostituire una venv funzionante per risolvere un download: leggere quale directory è stata scelta e se vi sono spazio e permessi di scrittura.

I modelli semantici sono conservati in `/kaggle/working/.humanization_semantic_models`, fuori dallo ZIP del run. I modelli vengono eseguiti in subprocess distinti per rilasciare la memoria tra le fasi. Il tempo complessivo dipende soprattutto dai checkpoint già presenti e dalle riscritture: mettere in conto diverse ore, non pochi minuti.

## Output e limiti

L'ultima cella esporta coorti, configurazioni, manifest, riscritture, fallimenti, soglie e punteggi. Scaricare lo ZIP prima di chiudere la sessione. I moduli reference e diagnostici per l'annotazione semantica sono distinti: il campione arricchito per score non stima direttamente l'accuratezza nella popolazione.

Il confronto 3B–14B è anche un confronto FP16–NF4. I due preset DIPPER differiscono per più fattori. La prova a tre fonti serve al collaudo, mentre le tabelle del PDF si riferiscono al run esteso. Lo scoring contestuale produce le modalità singola, fino a due/tre frasi e documento soltanto quando rientra nel budget; la scelta primaria è documentata separatamente nella [calibrazione NLI](README_NLI_Calibration.md).

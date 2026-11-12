# 05 — Sistemi di riscrittura e detector

**Riferimento: §1.6 del PDF.** Si parte da testi macchina e si chiede ai sistemi di riscriverli. I testi umani sono riferimenti per stile e calibrazione dei detector; non sono gli input da umanizzare.

## Configurazioni confrontate

- **Identità:** mantiene invariato il testo macchina e controlla la pipeline.
- **Baseline 3B:** Qwen2.5-3B-Instruct, prompt fisso, una generazione greedy.
- **14B:** Qwen2.5-14B-Instruct con lo stesso prompt; il preset usa NF4, mentre il 3B usa FP16.
- **DIPPER passaggio:** DIPPER XXL, diversità lessicale 60 e d'ordine 0, una chiamata sul passaggio intero.
- **DIPPER tre frasi:** stessi pesi e diversità, finestre di tre frasi e prefisso delle finestre già riscritte. Non significa applicare tre volte DIPPER allo stesso testo.

Non si selezionano riscritture interrogando un detector. Il confronto 3B/14B non isola il solo effetto della dimensione: cambia anche la quantizzazione. I due DIPPER differiscono per contesto, numero di chiamate e budget complessivo.

## Detector e definizioni

OpenAI RoBERTa-base è un riferimento storico addestrato su GPT-2; Fakespot è usato sull'input grezzo; RADAR-Vicuna è un classificatore RoBERTa-large. Binoculars è zero-shot e usa Falcon-7B e Falcon-7B-Instruct.

Le soglie usano 300 gruppi umani di calibrazione separati dal test, con FPR obiettivo 5% sulla calibrazione. L'ASR divide le evasioni per i soli originali già rilevati dal singolo detector. Il TPR descrive invece le rilevazioni sull'intera coorte considerata. Un fallimento di generazione non è un'evasione riuscita. Il voto di maggioranza riguarda i tre RoBERTa, non Binoculars, e non presume indipendenza fra i classificatori.

## Comandi della pipeline

I preset sono in [`configs`](../../configs); i notebook delle guide [pilot](../06_pilot_42/README.md) e [studio](../07_studio_126/README.md) coordinano gli stessi CLI. Esempio sul pannello congelato di 126 fonti:

```bash
python scripts/run_humanizers.py --cohort inputs/study_126/cohorts/study.jsonl --config configs/study_v2.json --output runs/study_126/rewriting --device cuda:0
python scripts/run_detectors.py --cohort inputs/study_126/cohorts/study.jsonl --rewrites runs/study_126/rewriting/rewrites.jsonl --config configs/study_v2.json --output runs/study_126/detection --device cuda:0
python scripts/run_zero_shot_detectors.py --cohort inputs/study_126/cohorts/study.jsonl --rewrites runs/study_126/rewriting/rewrites.jsonl --config configs/study_v2.json --output runs/study_126/zero_shot --observer-device cuda:0 --performer-device cuda:1
```

I file principali sono `rewrites.jsonl`, `pairs.jsonl`, `detector_scores.csv`, le soglie e le tabelle di riepilogo. Gli stati falliti sono conservati esplicitamente. I checkpoint possono richiedere download consistenti: per l'esecuzione completa usare una macchina GPU e l'ambiente `requirements/kaggle.txt`. Binoculars è configurato su due GPU; i preset DIPPER XXL richiedono un download di circa 45 GB. Non viene effettuato un ripiego implicito su un modello più piccolo.

Questi comandi effettuano nuove generazioni e inferenze. Per il professore, il percorso più rapido è l'[analisi del pilot](../06_pilot_42/README.md) o dello [studio già eseguito](../07_studio_126/README.md), che usa i file congelati senza GPU.

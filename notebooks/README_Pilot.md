# Pilot su 42 fonti — PDF §§1.6–1.7

Notebook: [Humanization_Pilot.ipynb](Humanization_Pilot.ipynb). La [guida generale](README.md) spiega accesso a Kaggle, CSV, risorse e download degli output.

## Obiettivo

Applicare la baseline Qwen2.5-3B e DIPPER XXL agli stessi 42 originali macchina, due per ciascuna delle 21 combinazioni dominio–generatore. I 300 umani di calibrazione sono separati dalle fonti di test. Le soglie dei tre detector supervisionati sono stimate su questi umani con FPR nominale del 5%.

## Due esecuzioni separate

Per il controllo tecnico iniziare con `MODE="smoke"`. Dopo aver controllato i tre esiti di ogni sistema, ripetere su `MODE="pilot"` usando un suffisso distinto per ciascun preset:

```python
MODE = "pilot"
RUN_SUFFIX = "_thesis_qwen3b"
ACTIVE_SYSTEMS = ["identity", "qwen_humanize"]
```

Per DIPPER:

```python
MODE = "pilot"
RUN_SUFFIX = "_thesis_dipper"
ACTIVE_SYSTEMS = ["identity", "dipper"]
```

Configurazione: [`configs/pilot.json`](../configs/pilot.json). L'opzione esplicita `--only dipper` attiva il preset anche se è disabilitato per default nel file. Il prompt, la generazione greedy, i checkpoint fissati e i parametri DIPPER rimangono quelli del pilot: intero passaggio, diversità lessicale 60 e diversità d'ordine 0. L'identità è il controllo che restituisce il testo iniziale.

## Ordine delle celle

Le sezioni verificano GPU e dipendenze, leggono i CSV ufficiali, ricostruiscono e verificano la coorte congelata, producono riscritture, eseguono detector, stile e metriche semantiche, quindi esportano i punteggi e l'archivio. Non eseguire la generazione più volte per scegliere una riscrittura in base ai detector: il protocollo prevede una sola generazione per sistema e fonte.

## Interpretazione

L'ASR si calcola sugli originali che il singolo detector rilevava già prima della riscrittura. I fallimenti restano registrati. I tre classificatori RoBERTa condividono una famiglia architetturale; il voto di maggioranza è descrittivo. Coseno, BERTScore e NLI misurano aspetti diversi; la soglia semantica storica 0,04075 non viene applicata a questi nuovi riscrittori.

Per consultare i risultati del PDF usare `results/` e il relativo README. I punteggi già calcolati consentono i confronti senza ripetere download e inferenze GPU.

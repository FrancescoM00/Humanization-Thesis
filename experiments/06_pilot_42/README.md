# 06 — Pilot appaiato su 42 fonti

**Riferimento: §1.7 del PDF.** Qwen 3B e DIPPER sull'intero passaggio riscrivono gli stessi 42 testi macchina: due per ciascuna combinazione di tre domini e sette generatori. Sono passaggi brevi del campione a lunghezza controllata, non documenti completi.

## Risultati pubblicati

| Detector | Qwen 3B | DIPPER passaggio |
| --- | ---: | ---: |
| OpenAI RoBERTa-base | 17/24 = 70,8% | 4/24 = 16,7% |
| Fakespot | 1/37 = 2,7% | 7/37 = 18,9% |
| RADAR | 2/28 = 7,1% | 3/28 = 10,7% |
| Maggioranza RoBERTa | 2/31 = 6,5% | 2/31 = 6,5% |

Le percentuali sono ASR: i denominatori contengono soltanto gli originali già rilevati dal singolo detector. Il vantaggio sul detector storico OpenAI non si trasferisce agli altri. Dopo Qwen, Fakespot rileva 41/42 testi contro 37 prima e RADAR 35/42 contro 28: una riscrittura può aumentare la rilevabilità.

I report congelati sono in [`results/06_pilot_42/qwen`](../../results/06_pilot_42/qwen) e [`dipper`](../../results/06_pilot_42/dipper). DIPPER accorcia 41/42 passaggi, con rapporto parole medio 0,739. Il coseno documentale resta elevato (0,928 Qwen, 0,922 DIPPER), senza garantire fedeltà. Le letture dei casi sono dell'assistente e non costituiscono gold umano.

## Rigenerare le tabelle senza GPU

Gli input ricevuti sono conservati separatamente in `inputs/pilot_42/qwen` e `inputs/pilot_42/dipper`: coorti, riscritture, score detector, semantica e stile. Dalla radice, con l'ambiente Python attivo:

```bash
python scripts/analyze_humanizer_results.py pilot --qwen-run inputs/pilot_42/qwen --dipper-run inputs/pilot_42/dipper --output runs/reproduced_pilot_42 --reference results/06_pilot_42/dipper
```

Il comando legge gli score individuali e produce aggregati e confronti appaiati, senza generare testi, scaricare modelli o interrogare servizi. `reproduction_check.csv` confronta i risultati rigenerati con le tabelle pubblicate. Le eventuali differenze negli intervalli storici sono segnalate, non nascoste: conteggi ed endpoint vanno distinti dall'ordine delle estrazioni bootstrap.

## Ripetere l'esperimento completo

Usare [`notebooks/Humanization_Pilot.ipynb`](../../notebooks/Humanization_Pilot.ipynb), ambiente GPU e configurazione `configs/pilot.json`. Impostare `MODE = "pilot"`. Il preset Qwen usa identità e `qwen_humanize`; il secondo run seleziona identità e `dipper`, mantenendo la stessa coorte e una cartella di output distinta. Per i dettagli operativi seguire la [guida del notebook](../../notebooks/README_Pilot.md).

La preparazione delle coorti da CSV, se necessaria, usa:

```bash
python scripts/prepare_humanizer_pilot.py --raw-dir data/raw --output runs/pilot_cohorts --per-cell 2 --calibration-per-domain 100 --seed 42
```

Per confrontare esattamente i risultati già pubblicati si usano le coorti congelate incluse. La preparazione nuova non sostituisce silenziosamente gli input storici. Le soglie usano 300 gruppi umani separati dal test; il controllo identità mantiene il testo macchina invariato.

Un run completo richiede GPU e download dei checkpoint; DIPPER XXL scarica circa 45 GB anche quando viene caricato in 4 bit. Per consultare o ricalcolare le tabelle precedenti basta la CPU. Gli esperimenti non producono una classifica generale degli humanizer.

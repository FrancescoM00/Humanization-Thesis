# Verifiche della consegna

La riorganizzazione è stata verificata il **12 novembre 2026** con Python 3.12/Linux e le dipendenze già disponibili nell'ambiente di lavoro. Non è stata eseguita un'installazione da zero su tutte le piattaforme. Non sono stati rilanciati riscrittori, detector o modelli NLI.

## Risultati dei controlli

| Controllo | Esito |
| --- | --- |
| Tre CSV ufficiali appena allegati | Identici byte per byte ai dati storici; 1.000 righe ciascuno |
| Ricostruzione pilot e studio dai CSV | JSONL identici agli input congelati: 300 umani di calibrazione +42/+126 fonti di test |
| Suite scientifica e launcher NLI | **192 test passati, 1 saltato perché richiede CUDA** |
| CLI scientifici controllati prima dell'importatore locale | 44 rispondono correttamente a `--help` |
| Celle Python dei due notebook | 20 celle valide sintatticamente; nessuna chiamata al giudice |
| Cinque input della calibrazione NLI | Copie identiche; archivio temporaneo formato dai soli input e dal nuovo codice |
| Analisi C3 con 100 candidate da cache | Riferimento C3 precedente riprodotto, `parent_C3_reproduced=true` |
| Sei esclusioni hedging e tre CSV | Percorsi migrati risolti e hash dei protocolli confermati |
| Tabelle dello studio su 126 fonti | **209/209 colonne numeriche controllate coincidenti** |
| Confronto appaiato dei pilot | **15/15 colonne numeriche controllate coincidenti** |

Nei confronti delle tabelle, la tolleranza è `1e-12` e la differenza massima osservata è circa **1,11e-16**. Sono controllati anche gli intervalli bootstrap. I CSV dei controlli sono [study_126_check.csv](validation/study_126_check.csv) e [pilot_42_check.csv](validation/pilot_42_check.csv), con i rispettivi manifest in `validation/`.

Il confronto legge i punteggi individuali, ricostruisce le metriche e usa i CSV finali soltanto come riferimento di verifica. Non copia le colonne finali come se fossero nuovi calcoli. `--reference` interrompe il comando con errore in presenza di differenze. I comandi completi sono nel [README](../README.md) e nelle guide [06](../experiments/06_pilot_42/README.md) e [07](../experiments/07_studio_126/README.md).

## Limiti

I test e i confronti verificano la migrazione e la riproducibilità delle analisi, non dimostrano l'affidabilità generale delle metriche. La validazione semantica con annotazioni umane indipendenti rimane distinta e ancora da completare. Le condizioni sintetiche e le letture dell'assistente non diventano etichette umane.

L'esecuzione GPU dei notebook nella nuova repository non è stata ripetuta: sono stati controllati codice, percorsi, configurazioni, coorti, recupero dell'ambiente e rianalisi dei risultati già disponibili. I modelli e le risorse necessarie a un nuovo run sono indicati nelle guide.

Il repository originale rimane a `d2868b323938a2a9371c2a872ad86b5596d0f097`, con checkout invariato. Questa nuova repository ha una cronologia propria.

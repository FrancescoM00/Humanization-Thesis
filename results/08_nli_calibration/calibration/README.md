# Calibrazione NLI completata: window2 + minimo

Il run ricevuto il 10 novembre 2026 completa 323.336 evidenze su 139 positivi presunti, 297 alterazioni e 137 identità, su 46 fonti. La policy congelata seleziona **window2 + minimo**: AUC di calibrazione 0,9735, media delle identità 0,9710 e q10 0,9509. Il test storico 141/265 resta escluso.

[Report](report.md), [interpretazione](interpretazione.md), [audit](integrity.md), [selezione](selection.json), [provenienza](provenance.json), [policy e protocollo](../../../docs/protocols/NLI_CALIBRATION.md). Il documento è escluso per copertura; la piccola differenza rispetto a window3 non dimostra superiorità statistica. Non si tratta di accuratezza 97,35% o validazione umana.

La configurazione è applicata ai [risultati dello study già calcolati](../../07_study_126/nli_primary_scores.csv), senza nuove inferenze. Il passo successivo è la verifica sulle [84 reference cieche](https://github.com/FrancescoM00/Humanization/blob/d2868b323938a2a9371c2a872ad86b5596d0f097/exports/kaggle_v2_study/reference/README.md), le cui nuove annotazioni non sono ancora ricevute. Nessuna nuova soglia di classificazione viene introdotta.

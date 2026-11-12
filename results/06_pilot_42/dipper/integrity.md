# Integrità del pilot DIPPER su 42 fonti

Audit del 9 novembre 2026 sui soli file restituiti da Kaggle, senza nuove generazioni o inferenze.

## Provenienza e confronto dei dati

- Archivio ricevuto: `humanization_pilotdipper.zip`, 1.947.790 byte; SHA-256 `21a66ed1cb1d3744be742a79f4fb7b924668d21200ed11033364ac48a18df31d`.
- Lo ZIP contiene 67 elementi, directory comprese, per 6.094.027 byte non compressi. I percorsi non sono assoluti e non contengono `..`.
- Il runtime registra commit `bde08c9447244debf4aea6be4d9491392940f889`, PyTorch `2.11.0+cu130`, due Tesla T4 disponibili e processi su `cuda:0`.
- `config.json`, manifest delle coorti e tutti e tre i pannelli smoke/pilot/study sono identici byte per byte al pilot Qwen e allo smoke DIPPER. Anche soglie detector e scale stilometriche sono identiche.
- La configurazione ha SHA-256 `86db7eb81f3d75fbfb306601395b70442659a3ea27d38b2e534991b571556d3b`; il pannello pilot ha SHA-256 `ed6b343dc972038515ff6dc4ebdfe7e1ce6ce50386cb49c5e104732da440574a`.
- Il pannello contiene 300 gruppi umani di calibrazione e 42 gruppi test disgiunti. Le 21 celle dominio/generatore hanno esattamente due fonti ciascuna. Gli originali delle riscritture coincidono con quelli Qwen.

Il confronto Qwen legge il run completo in `rewriting_cuda0/` dello ZIP storico, non presume un percorso `rewriting/`.

## Generazione e copertura delle metriche

- Il manifest di generazione è `completed`: 84 esiti attesi e presenti, 42 identità e 42 DIPPER, tutti `ok`, con ID unici.
- Le identità riproducono esattamente gli originali. Gli hash degli originali nei metadati coincidono con i testi.
- Tutte le 42 generazioni DIPPER terminano con EOS. Gli input registrati hanno 181–276 token e gli output 72–200 token, sotto i limiti configurati di 512. Non emergono arresti al limite d'uscita né esclusioni per input eccessivo.
- I sei output presenti nello smoke DIPPER sono identici nei testi al nuovo pilot. Lo smoke è incluso nel pilot e non si somma come campione indipendente.
- `rewriting/pairs.jsonl` e `semantic/pairs.jsonl` coincidono; SHA-256 `a7965a827d497a682ccbff9b348e0e581b9c9fec8935020e265d90fcd0e2a7f3`.
- Tutti e sei gli stadi semantici risultano `executed` su 84 coppie: lessicale, embedding, BERTScore, NLI, coseno allineato e BERTScore allineato. Ogni CSV copre gli stessi 84 ID senza duplicazioni; i manifest riportano lo stesso hash delle coppie.
- Gli hash del codice di generazione, metriche NLI e similarità allineata corrispondono ai sorgenti del repository. Non sono state introdotte nuove condizioni di scoring per il confronto.
- Rispetto alle identità del pilot Qwen, cinque stadi hanno gli stessi score numerici esatti, esclusi i contatori cumulativi di lavoro. BERTScore documentale presenta solo differenze dell'ordine dell'aritmetica floating point, con scarto massimo **7,17×10⁻⁹**. Non si afferma l'identità bit per bit di tutti i punteggi BERTScore.
- La cache NLI registra `inference_precision=float32` e `inference_device=cuda:0`. Il modello configurato resta DeBERTa-v3 base; la quantizzazione a 4 bit riguarda DIPPER. I pesi non sono inclusi nello ZIP e l'audit non ne verifica nuovamente i byte.
- Stile e guardie coprono tutte le 84 coppie; nessuna feature della RMS è mancante. Per le 42 identità RMS, POS-JS, variazione del centroide e cambiamenti delle guardie sono zero, con rapporto parole uno.
- L'esportazione continua comprende 84 coppie, senza soglia semantica e senza verdetto automatico di correttezza. I campi locali mancanti sono previsti per riscritture senza una singola frase bersaglio designata.

## Detector: statistiche ricalcolate dagli score

Sono presenti **1.404 score**, 468 per detector: 300 umani di calibrazione, 42 umani test, 42 originali macchina e 84 riscritture/identità. Il riepilogo è stato ricalcolato con `summarize_detection` sui soli score salvati: soglie identiche; tutte le 32 righe di riepilogo e le 8 righe macro coincidono entro 10⁻¹² nella lettura CSV.

Le **1.278 righe condivise** con Qwen — calibrazione, umani test, originali e identità — coincidono esattamente, compresi score e metadati. Il confronto dei sistemi usa quindi gli stessi denominatori e le stesse soglie.

| Detector | Soglia AI, superamento stretto | DIPPER successi / originali rilevati |
| --- | ---: | ---: |
| OpenAI RoBERTa-base | 0,9706076979637146 | 4/24 |
| Fakespot raw | 0,9611806273460388 | 7/37 |
| RADAR | 0,794653058052063 | 3/28 |
| Maggioranza | Almeno due voti AI su tre | 2/31 |

L'identità ha ASR zero. Le generazioni fallite sono zero; non si confondono score mancanti e classificazione umana. I FPR sugli umani test restano quelli Qwen perché i controlli sono gli stessi: 3/42 OpenAI, 0/42 Fakespot, 2/42 RADAR e maggioranza.

## Tempi e limiti

Le 42 generazioni DIPPER sommano **481,33 secondi** registrati, circa otto minuti, esclusi download e caricamento del modello. I sei stadi semantici/allineati sommano **63,99 secondi**, esclusi avvii, download e altri blocchi. Non sono tempi dell'intero notebook.

Non emerge un blocco tecnico nell'archivio. Le condizioni sono quelle dichiarate nello smoke: DIPPER XXL a 4 bit NF4, diversità lessicale 60, ordine 0, singola finestra sull'intero passaggio senza prefisso, greedy e un tentativo per fonte. Questa variante non rappresenta ogni configurazione DIPPER né isola l'effetto della quantizzazione.

Le schede di conservazione (12 coppie) e naturalezza (30 testi unici) sono vuote. Stato `ok` indica completamento tecnico, non fedeltà semantica o qualità. Il campione conserva 42 fonti, due per cella: le 84 righe e i tre detector non aumentano il numero di fonti indipendenti. Il confronto è appaiato e descrittivo; le annotazioni umane e l'estensione del campione rimangono necessarie.

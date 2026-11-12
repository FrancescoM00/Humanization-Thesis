# Audit del run NLI di calibrazione ricevuto

Verifica del 10 novembre 2026 sui soli file di `nli_calibration_v1.zip`, senza caricare modelli, eseguire nuove inferenze o chiamare API. Archivio di 6.183.514 byte, 31 elementi e 23.703.809 byte non compressi; SHA-256 `22669f3e6d67fc80b4b25fb98dcb975c4a70cce9b91695040e39dd08cf1f30d6`.

## Input congelati e separazione dei dati

Il `source_bundle.zip` incluso ha SHA-256 `bd294a67cb698943116ba44da2c4e01721369f4abd82dc055ed4dd04dd44a714`, uguale al bundle pubblicato prima del run. Coppie, allineamenti, etichette, policy e manifest coincidono byte per byte con gli input di `exports/nli_context_calibration/`. Anche i sorgenti dello scorer, dell'analizzatore e delle metriche coincidono con quelli del pacchetto preparato prima delle inferenze.

Sono presenti esattamente **573 coppie** su **46 gruppi di fonte**: 139 positivi presunti, 297 alterazioni intenzionali e 137 identità. Ogni etichetta appartiene allo split `calibration`. Gli ID non includono le 141 coppie positive e 265 alterate del confronto storico di test; non vi è sovrapposizione fra i rispettivi gruppi di fonte. I positivi non collegati ai negativi appaiati non sono stati aggiunti al pannello.

Le identità riproducono gli originali distinti dei positivi selezionati entro il gruppo di fonte. Originale e modificato sono uguali e usano gli stessi chunk in entrambe le direzioni. Non sono 137 fonti indipendenti e non entrano nell'AUC. Le classi delle altre coppie rimangono proxy operative, non giudizi umani gold.

## Esecuzione dichiarata e copertura

Il runtime registra DeBERTa-v3 base alla revisione `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`, `cuda:0`, float32, batch 4, due thread e budget di 384 token per coppia. Il manifest registra **323.336 evidenze nuove**, pari a quelle pianificate, in 3.222,22 secondi, circa **53,7 minuti** esclusa la preparazione del piano.

Sono presenti **4.584 righe uniche**, esattamente 573 ID × due direzioni × quattro modalità. ID, hash delle coppie e allineamenti coincidono con gli input congelati. Nessuna modalità o direzione è stata omessa.

| Modalità | Righe direzionali complete / attese | Coppie etichettate complete / 436 | Identità complete / 137 |
| --- | ---: | ---: | ---: |
| Singole esaustive | 1.146/1.146 | 436/436 | 137/137 |
| Fino a due unità | 1.146/1.146 | 436/436 | 137/137 |
| Fino a tre unità | 1.146/1.146 | 436/436 | 137/137 |
| Documento | 250/1.146 | 77/436 | 36/137 |

Il documento viene escluso per copertura, come previsto: non gli viene assegnata AUC zero e non si calcola un'AUC globale soltanto sui casi che entrano nel budget.

La cache delle probabilità e il JSONL delle evidenze estese sono esclusi dallo ZIP leggero, come dichiarato dal launcher; rimangono su Kaggle. L'audit può verificare la coerenza degli aggregati e della selezione, non ricalcolare indipendentemente i 323.336 output neurali o verificare i byte dei pesi assenti dall'archivio.

## Ricalcolo della selezione

L'analizzatore è stato rieseguito in una directory temporanea sui CSV ricevuti, con la stessa policy, 2.000 ricampionamenti dei gruppi di fonte e seed 42. Tabella dei 16 candidati, AUC per condizione, score combinati e selezione coincidono con gli output ricevuti entro 10⁻¹²; `selection.json` coincide integralmente.

La scelta risultante è **`window2` + `min`**:

- AUC di calibrazione **0,9734879**, IC bootstrap 95% **[0,9566194; 0,9873532]**;
- media degli score delle identità **0,9710096** e decimo percentile **0,9509247**, sopra i requisiti 0,95 e 0,90;
- mediana identità 0,9778432 e minimo individuale 0,8666299. Quest'ultimo non viola la policy, che impone soglie sulla media e sul percentile fra documenti, non su ogni singola identità;
- copertura completa su tutte le coppie etichettate e tutte le identità.

Sul medesimo pannello, `single_all` + `min` ha AUC 0,9469636 e `window3` + `min` 0,9712351. Non è corretto confrontare direttamente questi valori con lo storico 0,958 ottenuto sul diverso split di test e con retrieval storico. Anche le AUC inferiori degli aggregatori q10, circa 0,713–0,739, sono riprodotte dal ricalcolo: non è emersa un'inversione dello score o un errore del selettore.

La differenza window2−window3 con il minimo è soltanto **0,0022527**. Come controllo descrittivo aggiuntivo, un bootstrap appaiato sulle stesse 46 fonti dà IC 95% **[−0,0052441; +0,0094369]**. Non dimostra superiorità statistica di window2; non è stato usato per cambiare la regola di scelta fissata. Window2 resta il candidato scelto dalla massima AUC puntuale prevista dalla policy.

## Confini delle conclusioni

L'AUC è una prestazione di sviluppo sulle etichette presunte usate per scegliere il candidato. Gli intervalli non correggono l'ottimismo di questa selezione e non dimostrano accuratezza gold sui veri humanizer. Il passaggio alle annotazioni reference rimane necessario. Nessun giudizio umano nuovo, nuova soglia per gli humanizer o risultato del test storico è stato usato in questo audit.

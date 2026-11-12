# Interpretazione della calibrazione NLI

Il confronto ricevuto seleziona **`window2` con aggregatore `min`**, applicando la policy fissata prima delle nuove inferenze. L'AUC sul solo pannello di calibrazione è **0,9735**, con intervallo bootstrap per fonte **[0,9566; 0,9874]**. È una scelta di sviluppo su etichette operative presunte, non un'accuratezza del 97%, una percentuale di significato preservato o una validazione sui veri humanizer.

Le 436 coppie etichettate comprendono 139 conservazioni presunte OpAI e 297 alterazioni intenzionali su 46 fonti. Le 137 identità verificano la coerenza dello score e non entrano nell'AUC. Le 141 positive e 265 alterate dello storico test sono escluse; non è stato scelto il candidato dopo aver consultato nuove annotazioni umane.

La modalità primaria assegna a ciascuna frase/chunk ipotesi il massimo supporto fra tutte le singole unità e le finestre contigue fino a due unità della premessa. Prende poi il minimo fra le unità e fra le due direzioni. Rimangono quindi sia l'allineamento sia la sensibilità al caso peggiore, con una premessa più ampia della sola frase.

| Modalità | Media | q10 | Minimo | Quota non supportata, orientata alla conservazione |
| --- | ---: | ---: | ---: | ---: |
| Singole esaustive | 0,9143 | 0,7128 | 0,9470 | 0,9031 |
| Fino a due unità | 0,9495 | 0,7190 | **0,9735** | 0,9197 |
| Fino a tre unità | 0,9527 | 0,7392 | 0,9712 | 0,9221 |
| Documento | Mancante | Mancante | Mancante | Mancante |

Sono AUC sulla stessa calibrazione. **Non si presenta 0,9735 come miglioramento rispetto allo storico 0,958:** quest'ultimo era sul diverso test 141/265, mentre il nuovo valore è sullo sviluppo 139/297 usato per scegliere il candidato. Il confronto interno pertinente è fra 0,9470 delle singole esaustive e 0,9735 delle finestre fino a due, sempre sulla medesima calibrazione e senza trasformarlo in conferma indipendente.

Le dodici combinazioni di singole/finestre superano i requisiti di ammissibilità. Le quattro combinazioni documentali non hanno copertura completa e non vengono salvate calcolando l'AUC soltanto sul sottoinsieme facile: il documento copre entrambe le direzioni per **77/436 coppie etichettate** e **36/137 identità**, circa il 17,7% e il 26,3%. Questo ne esclude l'uso primario con il budget fissato a 384 token; non dimostra che ogni approccio documentale sia inferiore con altri modelli o contesti.

**Due unità non sono dimostrate statisticamente migliori di tre.** La differenza delle AUC dei minimi è soltanto 0,00225 e gli intervalli sono ampiamente sovrapposti. La scelta rimane `window2` perché la regola congelata massimizza l'AUC puntuale; non si cambia ora criterio per preferire il candidato desiderato. I tre contesti restano disponibili per sensibilità. Gli intervalli descrivono incertezza sullo sviluppo e non correggono l'ottimismo dell'aver selezionato fra sedici candidati.

Le finestre migliorano lo score delle identità senza renderlo perfetto:

| Premessa, aggregatore minimo | Media identità | Decimo percentile fra documenti | Minimo osservato | Identità sotto 0,90 |
| --- | ---: | ---: | ---: | ---: |
| Singole esaustive | 0,9668 | 0,9420 | 0,8486 | 10/137 |
| Fino a due unità | **0,9710** | **0,9509** | 0,8666 | 3/137 |
| Fino a tre unità | 0,9726 | 0,9597 | 0,8666 | 3/137 |

La policy richiede media almeno 0,95 e decimo percentile almeno 0,90, non che ogni identità raggiunga quei valori. La configurazione scelta passa, ma tre testi identici hanno ancora score inferiore a 0,90. Questo conteggio è descrittivo: 0,90 non è una soglia calibrata di allarme sui riscrittori. Il minimo mantiene la fragilità all'unità difficile e alla lunghezza, da verificare con le annotazioni. Le identità appartengono alle stesse 46 fonti e non sono 137 campioni indipendenti.

Le famiglie di alterazione non danno tutte lo stesso ordinamento. Le AUC seguenti usano per ogni famiglia i suoi positivi collegati, senza aggiungere positivi estranei:

| Alterazione | Positivi / negativi | Minimo singole | Minimo fino a due | Minimo fino a tre |
| --- | ---: | ---: | ---: | ---: |
| Entità | 129/129 | 0,9699 | 0,9884 | 0,9916 |
| Negazione | 95/95 | 0,9353 | 0,9583 | 0,9515 |
| Numeri | 26/26 | 0,9638 | 0,9571 | 0,9438 |
| Quantificatori | 28/28 | 0,9196 | 0,9675 | 0,9534 |
| Rimozione di cautela | 15/15 | 0,8200 | 0,8911 | 0,9311 |
| Dettaglio aggiunto | 4/4 | 0,9375 | 1,0000 | 1,0000 |

Il contesto più ampio aiuta qui entità e cautele, ma può ridurre la separazione di numeri, negazioni e quantificatori. L'aumento del massimo supporto non garantisce migliore discriminazione. La famiglia numerica mostra che le finestre non migliorano uniformemente ogni errore; un'AUC aggregata non deve nascondere queste differenze. Il risultato perfetto sui dettagli aggiunti riguarda soltanto quattro negativi e non sostiene una prestazione generale. Anche l'AUC delle cautele su quindici casi di calibrazione non chiude il limite osservato sui diversi casi del test storico.

Il minimo risulta favorito rispetto a q10 in questo disegno, dove le corruzioni sono prevalentemente locali: un singolo errore può non incidere sul decimo percentile di un documento con molte frasi. Il confronto non dimostra che il minimo sia l'aggregatore migliore per derive diffuse o trasformazioni strutturali più forti. Non si trasferiscono automaticamente né questa AUC né la soglia OpAI storica 0,04075 alle riscritture reali.

Il risultato operativo è ora una configurazione primaria riproducibile da verificare sulle **84 coppie reference**, non uno strumento già validato da persone. Le annotazioni indipendenti dovranno confrontare conservazioni corrette, errori e fusioni di frasi con gli score continui congelati. La review dell'assistente e i risultati Groq non diventano gold semantica; gli input e i risultati dello study già osservati rimangono dichiarati come tali.

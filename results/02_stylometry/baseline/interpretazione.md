# Risultati della prima fase sperimentale

L'esperimento è stato eseguito su **1.800 testi**: 300 umani e 300 macchina per Essay, Reuters e WritingPrompts. I 300 output macchina di ogni dominio sono distribuiti tra sette generatori (43 ciascuno, StableLM 42). Non sono 300 output per ciascun generatore. Sono state estratte 105 feature con spaCy piccolo, textstat e MATTR50 scritto a mano.

## Provenienza e pulizia

La fonte iniziale è un mirror MGTBench pubblico, fissato a un commit e con hash dei tre CSV nel manifest. I tre CSV successivamente forniti dall'autore come scaricati da Google Drive sono risultati **identici byte per byte**, anche rispetto agli hash già registrati nei manifest: [verifica successiva](../../../docs/provenance/mgtbench_drive_comparison.md). Nessun risultato cambia. TH-Bench non fornisce direttamente un campione clean bilanciato di queste dimensioni: i suoi file per gli attacchi non corrispondono al disegno richiesto.

La baseline è quindi stata preparata localmente: spacing del loader TH-Bench, rimozione simmetrica delle code senza una terminazione di frase plausibile, minimo 50 parole dopo la pulizia, deduplicazione e campionamento bilanciato per generatore. La regola di terminazione è euristica e non certifica la completezza grammaticale; non vengono rimossi selettivamente vocabolario Wikipedia o formule conversazionali.

Il pilot aveva evidenziato 430/900 testi macchina senza terminazione plausibile, contro 66/900 umani. Nel campione principale sono state rimosse code in 435 macchine e 66 umani. I campioni possono differire perché si pulisce prima di campionare: il confronto pilot/principale non stima causalmente il solo effetto del trimming. Il pilot completo resta in `../results_pilot/`.

## C'è segnale, ma il trasferimento di dominio è il problema principale

ROC-AUC della regressione logistica sulle sole feature stilometriche, senza conteggi globali, TTR/hapax e proxy Wikipedia:

| Dominio di test | CV nello stesso dominio | Addestramento sugli altri due domini | IC 95% del trasferimento |
| --- | ---: | ---: | --- |
| Essay | 0.922 | 0.779 | 0.741–0.811 |
| Reuters | 0.943 | 0.843 | 0.811–0.872 |
| WritingPrompts | 0.879 | 0.535 | 0.491–0.577 |

Gli split mantengono insieme le coppie di origine e gli eventuali prompt condivisi. Imputazione e standardizzazione sono apprese solo sul training. Non viene usato il testo grezzo come input del classificatore. Sono state controllate tutte le 21.600 predizioni esportate: nessuna sovrapposizione di gruppi train/test e una sola predizione OOF per documento entro ciascuna valutazione.

Dentro il dominio la separazione è buona; su WritingPrompts non visto l'AUC è compatibile con il caso. Inoltre AUC non equivale a buona classificazione alla soglia 0.5: per Essay non visto, AUC 0.779 accompagna macro-F1 0.454 e balanced accuracy 0.550. Il cambiamento di distribuzione altera anche la soglia decisionale.

Nel leave-one-generator-out il modello di stile ottiene AUC da **0.663 per StableLM** a **0.981 per ChatGLM**; Claude raggiunge 0.712. I generatori sono esclusi dall'addestramento, insieme all'origine dei testi di test. Questi test comprendono gli stessi tre domini, quindi non provano il trasferimento simultaneo a dominio e generatore entrambi nuovi.

## Lista corta candidata per il vettore stilometrico

Hedges g è la differenza standardizzata **macchina meno umano**: positivo = valore più alto nelle macchine. I valori seguenti sono aggregati; gli effetti per tutti i 21 incroci dominio×generatore sono nel CSV completo.

| Feature | g aggregato | IC 95% | Generatori con lo stesso segno |
| --- | ---: | --- | ---: |
| Frequenza di `and` | +0.589 | +0.508 / +0.678 | 7/7 |
| Punteggiatura totale per 100 parole | −0.518 | −0.595 / −0.442 | 7/7 |
| Proporzione POS numerali | −0.495 | −0.588 / −0.415 | 7/7 |
| Proporzione POS avverbi | −0.436 | −0.513 / −0.363 | 7/7 |
| Proporzione POS sostantivi | +0.357 | +0.297 / +0.427 | 6/7 |
| Flesch–Kincaid grade | +0.304 | +0.251 / +0.354 | 6/7 |
| MATTR50 | −0.612 | −0.697 / −0.528 | 5/7 |

Tutte superano |g|≥0.3 con lo stesso segno in ciascuno dei tre domini. Sono state rimosse ridondanze con correlazione Spearman assoluta ≥0.85. La coerenza per generatore è calcolata aggregando i domini: non significa che ogni feature sia coerente in tutti i 21 incroci.

Come nucleo più stabile terrei le prime quattro; conserverei MATTR, sostantivi e leggibilità come dimensioni aggiuntive, rendendo esplicite le eccezioni tra generatori. **La shortlist è esplorativa e selezionata sull'intero campione**: i risultati della regressione logistica sono quelli degli insiemi di feature fissati in anticipo, non una valutazione indipendente di queste sette feature selezionate dopo aver visto i dati.

Un esempio utile di feature esclusa è la variabilità delle frasi: il coefficiente di variazione ha un effetto aggregato grande (g≈−0.643), ma in Essay l'effetto è circa +0.02. La sola graduatoria aggregata avrebbe quindi suggerito una robustezza che non c'è.

## La lista Wikipedia funziona?

I sette proxy implementati coprono vocabolario AI, enfasi, conclusioni, formule di chatbot, parallelismi negativi, Markdown ed em dash. Non coprono tutta la guida, rule of three o affermazioni semantiche come analisi superficiale e attribuzioni infondate.

| Dominio non visto | Solo proxy Wikipedia | Stile + proxy Wikipedia | Solo lunghezza |
| --- | ---: | ---: | ---: |
| Essay | 0.580 | 0.803 | 0.849 |
| Reuters | 0.487 | 0.856 | 0.883 |
| WritingPrompts | 0.520 | 0.563 | 0.780 |

Questi proxy **non funzionano come criterio generale trasferibile** in questo campione. Il vocabolario AI ha un effetto positivo in tutti e tre i domini, ma relativamente contenuto (g 0.416, 0.306, 0.269); altri pattern cambiano verso. Per esempio il proxy dei parallelismi negativi è meno frequente nelle macchine Reuters (g≈−0.397), mentre ha effetti positivi piccoli negli altri due domini. Aggiungere i proxy allo stile migliora qui alcune AUC, ma non è stato eseguito un test indipendente che dimostri un miglioramento stabile.

## Limite decisivo e uso nella tesi

Anche dopo la pulizia, gli umani hanno in media circa **536 parole lessicali**, le macchine circa **253**. Il controllo di sola lunghezza raggiunge AUC elevate e spesso supera lo stile nel trasferimento. Quindi normalizzare le feature e rimuovere i conteggi dal modello **non dimostra che tutto il segnale residuo sia puramente stilistico**. Anche numerali e sostantivi possono catturare contenuto e specificità del dominio.

Userei le sette dimensioni come vettore candidato del meta-evaluatore, conservando differenze con segno e riferimenti umani per dominio. Non massimizzerei automaticamente MATTR o altri valori per definire un testo più umano: spostarsi in una direzione non garantisce naturalezza né conservazione del contenuto.

La successiva conferma dovrebbe usare un campione indipendente, controllo della lunghezza e generatori più recenti; soltanto dopo andrebbero fissati eventuali pesi di uno score aggregato. Questo esperimento non valuta ancora testi umanizzati.

Codice e riproduzione in `../README.md`; tabelle complete in `feature_effects.csv`, `feature_robustness.csv` e `classifier_metrics.csv`; dettagli della fonte in `data_manifest.json`; grafici esportabili in PNG e PDF. **23 test superati**.

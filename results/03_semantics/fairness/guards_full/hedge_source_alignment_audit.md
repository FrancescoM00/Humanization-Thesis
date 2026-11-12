# Diagnostica testuale dei nove hedge test

Revisione dell’assistente **dopo** aver osservato i risultati, non annotazione umana cieca. Nessuna etichetta o modello è modificato. Sono state cercate le frasi target nei vettori nativi dei Parquet, validando la lineage con il codice dello studio stilometrico. Tutti i nove abbinamenti sono univoci. Gli identificatori sNNNN sono gli ordinali ricostruiti della stessa posizione nativa, non una colonna sentence_id pubblicata.

Le quattro cancellazioni nette dei conteggi non provano che la corruzione abbia annullato una cautela aggiunta nella stessa proposizione. Nei casi sotto, omissioni del lessico, sostituzioni di espressioni e cambiamenti in altre frasi spiegano il totale. Due casi mantengono could dopo aver eliminato possibly: il negativo proxy non certifica una transizione semantica netta.

## 1. opai_matched:5abacc16529d471546cd0ce0

Posizione: s0005; operazione: opai_polish.

Il sorgente e il positivo mantengono could possibly; si elimina solo possibly e could rimane. La modifica non è una rimozione completa della possibilità.

## 2. opai_matched:5d8888e2cf8666da98893730

Posizione: s0021; operazione: opai_style.

May be able nel sorgente diventa might be able nel positivo e are able nel corrotto: viene rimossa una modalità presente nella stessa proposizione.

## 3. opai_matched:58f85670a21b0d9fcdbb01ab

Posizione: s0009; operazione: opai_style.

Can relate nel sorgente diventa may pertain nel positivo e do pertain nel corrotto. Il pareggio di conteggio originale/corrotto dipende anche dal fatto che il lessico v1 non conta can; non dimostra ripristino del significato.

## 4. opai_matched:978fbc0d2782ffed6df8340c

Posizione: s0040; operazione: opai_style.

Maybe...will be nel sorgente diventa may persist nel positivo e does persist nel corrotto. Il lessico v1 omette maybe; la cancellazione dei conteggi non dimostra conservazione della cautela.

## 5. opai_matched:1899ca0f3525e0538213bb22

Posizione: s0021; operazione: opai_paraphrase.

Could potentially nel sorgente diventa may be able nel positivo e are able nel corrotto: la possibilità originale viene rafforzata in asserzione.

## 6. opai_matched:6797f4465c4776cd56bab52d

Posizione: s0008; operazione: opai_polish.

Il sorgente e il positivo hanno could possibly; eliminato possibly ma mantenuto could. Caso di intensità/ridondanza, non rimozione completa del modale.

## 7. opai_matched:10d8af56f93943f995302779

Posizione: s0028; operazione: opai_polish.

La frase sorgente coincide col positivo e contiene may; il corrotto sostituisce may con does. Il conteggio documentale torna a quello iniziale per cambiamenti altrove, non per conservazione della modalità di questa proposizione.

## 8. opai_matched:27172f8c9313839d100c00ed

Posizione: s0028; operazione: opai_paraphrase.

La frase sorgente coincide col positivo e contiene may; la versione corrotta è assertiva con does. Il conteggio assoluto globale non localizza lo scope.

## 9. opai_matched:459130911b25ecaee015703e

Posizione: s0003; operazione: opai_style.

Il sorgente ha may...look like, il positivo may...appear, il corrotto do...appear. Viene rimosso may; appear, contato dal lessico v1, resta. Un totale uguale non distingue modalità e verbo di apparenza.

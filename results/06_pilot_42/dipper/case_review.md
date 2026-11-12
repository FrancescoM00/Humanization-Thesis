# Lettura qualitativa del pilot DIPPER e confronto con Qwen

Questa revisione è dell'assistente. Non è un'annotazione umana, un riferimento gold o una valutazione cieca: testi e punteggi sono stati consultati insieme. Non permette di trasformare il numero di casi commentati in una percentuale di errori del sistema, né di calcolare l'accuratezza delle metriche. Non misura la naturalezza.

Sono state lette integralmente **12 coppie DIPPER** e le corrispondenti riscritture Qwen. Sei fonti sono selezionate tramite una regola indipendente dagli score: per ciascun dominio, le prime due ordinate per SHA-256 del `source_id`. Altre sei sono approfondimenti mirati: la seconda consegna StableLM già nota, il forward medio NLI più basso, la maggiore compressione in Essay, il caso quantitativo già discusso per Qwen, un caso di importo/valuta fra le evasioni Fakespot e un caso con alto supporto reverse. Questa seconda selezione è esplorativa e sovrarappresenta fenomeni diagnostici.

Le citazioni provengono da `rewriting/rewrites.jsonl` del pilot DIPPER. Qwen è letto da `rewriting_cuda0/rewrites.jsonl` nello ZIP originale `exports/kaggle_pilot/humanization_pilot.zip`. Gli esempi NLI locali sono ricostruiti dall'allineamento e dalla cache salvati: nessuna nuova inferenza. Forward indica originale → riscrittura, reverse riscrittura → originale. I conteggi delle parole sono quelli alfabetici spaCy del run.

## Sei casi selezionati tramite identificatore

### 1. Essay/StableLM: la consegna rimane una consegna, ma perde i requisiti

- Fonte `Essay:245:198f5b62c8f94655`.
- DIPPER `humanizer_3f7bee92c7fcfa52c499ccdb`; Qwen `humanizer_9bf55003649b58186c865ce3`.
- Parole 133→64; NLI medio forward/reverse 0,803/0,351.

L'input chiede di discutere l'etica del marketing verso consumatori vulnerabili e specifica «APA format and MLA format is required», oltre al contenuto dei primi tre paragrafi. DIPPER conserva il nucleo della consegna: «Discuss the ethical implications of marketing to uninformed consumers», ma elimina formati e struttura numerata. Il basso reverse ha quindi una perdita testuale riconoscibile.

Qwen aveva invece svolto il compito, producendo un elaborato di 381 parole con esempi aggiunti. DIPPER resta nel compito di riscrittura della consegna, senza conservarne tutti i vincoli. L'esempio distingue due comportamenti su un input anomalo; non rende appropriata una classifica di qualità della prosa su quel solo caso.

### 2. Essay/ChatGLM: compressione e fusione di evidenze

- Fonte `Essay:198:ee27e99f1bc3dc32`.
- DIPPER `humanizer_2e39571e7c6c84c0fb41ce3c`; Qwen `humanizer_4feaa95c9c411db4a15d0b66`.
- Parole 160→122; NLI medio 0,794/0,713.

Il caso coincide con lo smoke. DIPPER mantiene medicina, assistenza infermieristica e PICOT, ma perde il riferimento esplicito ai colloqui clinici e alle popolazioni. La nuova lista «clinical history, physical examination, laboratory tests and diagnostic images» unisce elementi che l'originale distribuiva tra esami e frase successiva sulle tecniche di imaging. Il minimo forward di circa 0,0042 su questa lista non prova che le immagini siano nuove: nessuna singola premessa sostiene tutta la frase fusa.

Qwen conservava più dettagli diagnostici ma aggiungeva una spiegazione di PICOT e della cura olistica. Il confronto mostra perdita e aggiunta in forme differenti; il solo minimo NLI non stabilisce un ordinamento globale della fedeltà.

### 3. Reuters/StableLM: cambia il titolo citato e si rafforza una cautela

- Fonte `Reuters:373:f298982fe6a5acf4`.
- DIPPER `humanizer_b2a3acca34c2c79f4f39fe8d`; Qwen `humanizer_1adc1a1e5089587df2b72590`.
- Parole 140→112; NLI medio 0,702/0,599.

Il documentario «China's Rain of Torture» diventa «China's Tortured Rain». È un cambiamento dell'identificatore testuale di un'opera, non soltanto una variazione del lessico circostante. Non si verifica qui se il documentario o il resoconto originale siano veri nel mondo: il problema osservato è la fedeltà al testo fornito.

La formulazione «the Vice-President would have taken this step» diventa «the vice-president took this step», con perdita della modalità. Scompare anche «last month». Qwen conserva il titolo ma cambia a sua volta il riferimento temporale in «earlier that month» e presenta la visita in termini più assertivi. La riscrittura Qwen non è quindi assunta come gold per giudicare DIPPER.

### 4. Reuters/GPT4All: informazione normativa eliminata, supporto forward alto

- Fonte `Reuters:99:72f0d777c8384b92`.
- DIPPER `humanizer_b76aac64e96ccf0034ef7335`; Qwen `humanizer_7ae4eaa828dfba4332e87fb1`.
- Parole 169→108; NLI medio 0,987/0,619.

DIPPER mantiene l'azione della Building Societies Association e i rischi finanziari, ma elimina la disposizione che «enables UK companies to delay the payment of taxes» e l'ultima frase sulle acquisizioni transfrontaliere. La frase «The BSA has suggested that the legislation breaches» diventa «has also warned that the law violates»: la fonte dell'affermazione resta la BSA, mentre cambiano verbo e forza della formulazione.

Il forward molto alto è compatibile con una riscrittura in larga parte sostenuta dall'originale; non certifica la conservazione delle informazioni omesse. Qwen conserva sia il differimento fiscale sia il riferimento finale alle acquisizioni transfrontaliere. L'esempio rende utile leggere il reverse insieme al forward.

### 5. WritingPrompts/Claude: una lettera conserva l'orario ma cambia un oggetto

- Fonte `WritingPrompts:440:e85378f4f9e16583`.
- DIPPER `humanizer_c16c0a1a3bcb3bf286e803d1`; Qwen `humanizer_cedcd3fdc23788eadfad35b6`.
- Parole 152→136; NLI medio 0,612/0,601.

DIPPER conserva la lettera dal futuro, l'intervallo di un anno, le 9:32 e il doppio espresso. «a blueberry muffin» diventa però «a blueberry bun»: viene sostituito il tipo di prodotto richiesto. La frase sulla massima importanza delle istruzioni è sostituita da «I trust you to follow my instructions», e il saluto «Dear [My Name]» diventa «Dear Name». Nella citazione la chiusura delle virgolette non viene riprodotta.

Qwen conserva il muffin, ma aggiunge dopo la lettera reazioni del narratore e ipotesi come «Was it a prank, or something more sinister?». La similarità tematica di entrambi gli output non esclude differenze puntuali. La naturalezza della resa narrativa resta da annotare.

### 6. WritingPrompts/Dolly: cambia chi pronuncia una battuta

- Fonte `WritingPrompts:64:b0dc47cab2dbc7d6`.
- DIPPER `humanizer_0406323efb00a922f6f9386b`; Qwen `humanizer_a0660536de32b9c95478d812`.
- Parole 152→105; NLI medio 0,660/0,470.

Nell'originale Ronald si presenta con «Hi there, I'm Ronald», poi è il viaggiatore a dire «It's a pleasure to meet you, Ronald is such a lovely name». DIPPER scrive «It's a pleasure to meet you, I'm Ronald», attribuendo il piacere dell'incontro a Ronald e perdendo il commento del viaggiatore sul nome. Scompaiono anche «101 Burgers hill» e il passaggio lungo il sentiero di ghiaia; 9am diventa semplicemente «nine o'clock», con minore specificità.

Qwen mantiene l'identificatore della villa e le due battute separate. Alcune omissioni riguardano contesto, altre l'attribuzione del dialogo: un singolo valore di similarità non identifica quale aspetto sia stato modificato.

## Sei approfondimenti mirati

### 7. Reuters/StableLM: rimozione di ripetizioni e perdita di una seconda consegna

- Fonte `Reuters:903:3d3ffb6dca295be8`.
- DIPPER `humanizer_0bfa5e78ca7f7157f424b29c`; Qwen `humanizer_ff0e82d4bda03fd1550ea5f5`.
- Parole 164→63; NLI medio 0,741/0,721. Evasione di OpenAI e Fakespot.

L'input contiene cinque occorrenze della richiesta di scrivere 400 parole sul futuro del mercato azionario cinese, di cui quattro consecutive alla fine, e anche una consegna diversa: «Discuss the two best ways for US banks to increase customer satisfaction». DIPPER riassume la consegna cinese in un solo paragrafo, conservando 400 parole, cinque anni e due fonti, ma elimina quella sulle banche statunitensi e altri vincoli. La rimozione delle ripetizioni riduce molte parole senza altrettanta perdita di informazione unica; l'eliminazione della seconda consegna è invece sostanziale.

Qwen risponde a entrambe le tematiche con un testo di 274 parole e riferimenti bibliografici aggiunti, non verificati. I due sistemi escono dall'equivalenza in modi diversi. L'evasione DIPPER qui riguarda un input rumoroso: non prova un'umanizzazione fedele di prosa ordinaria.

### 8. WritingPrompts/ChatGPT: la fusione delle frasi fa crollare il forward

- Fonte `WritingPrompts:285:80b9ea16794068e3`.
- DIPPER `humanizer_f2e5b3f95dcf48c423b31a50`; Qwen `humanizer_e21288fa7287adb0dbb787ef`.
- Parole 148→126; NLI medio **0,00262/0,76640**. Evasione di OpenAI e RADAR.

La storia di Maria e Sofia è largamente riconoscibile: favoritismo dei genitori, vaso rotto, bugia, punizione, senso di colpa e confessione finale restano. Ci sono variazioni da annotare: «Sofia» diventa «Sophia» nella prima menzione e poi torna «Sofia»; «was never the same again» diventa «never recovered», più forte. Non si dichiara quindi conservazione perfetta.

Il dato diagnostico è che **15 frasi originali diventano cinque**, ciascuna delle quali combina più proposizioni. La prima ipotesi, «Maria was always the favorite of her parents, clever, beautiful and obedient», unisce le prime due originali; la quinta combina senso di colpa e confessione. Il confronto con singole premesse assegna a tutte e cinque supporti forward fra circa 0,00083 e 0,00428. Questo non dimostra cinque invenzioni: il documento contiene gran parte delle informazioni, distribuite in frasi diverse.

Anche il reverse ha limiti. Sulla frase originale «She approached Sofia and confessed everything», la nuova frase finale contenente «confessed everything to her sister» è fra i candidati, ma riceve supporto 0,00608 e neutralità 0,98994. L'avvicinamento fisico è omesso, mentre la confessione è esplicita. È un esempio di ipotesi congiunta/coreferenza e supporto graduato male come misura della gravità, non un errore di retrieval dimostrato. Qwen mantiene a sua volta la trama, aggiungendo intensificatori come «One fateful day» e «harshly punished». Il caso impedisce di equiparare il forward basso alla distruzione dell'intero significato.

### 9. Essay/ChatGPT: evasione dopo una forte sintesi

- Fonte `Essay:889:6dac9449b2023bbd`.
- DIPPER `humanizer_f3c34017cd2d9e5d1387c55c`; Qwen `humanizer_60313ab1c624837063d2b3c5`.
- Parole **152→59**; NLI medio 0,986/0,438. Evasione di Fakespot.

DIPPER conserva la distinzione generale fra ragionamento basato sulle evidenze e sulla frequenza, ma elimina PubMed, la valutazione di sintomi/storia clinica, i trattamenti di efficacia dimostrata e gli esiti dei pazienti. La chiusura menziona sintomi e malattie «observed in the patient», mentre l'originale si riferisce anche a un gruppo di pazienti.

Il forward alto coesiste con una notevole riduzione del contenuto: un riassunto può essere sostenuto dalla fonte senza essere equivalente. Qwen mantiene molti più dettagli. L'evasione del detector non va presentata senza segnalare il rapporto di lunghezza e la perdita di informazione.

### 10. Essay/Claude: il numero viene cambiato

- Fonte `Essay:308:a3392ef1e918715d`.
- DIPPER `humanizer_1262d303f86a672c96ff1467`; Qwen `humanizer_6927e6c090a0effb9fead649`.
- Parole 156→97; NLI medio 0,254/0,513. Evasione di Fakespot.

Il limite «up to 100 feet high» diventa «a height of thirty feet». Cambia il valore numerico e scompare il vincolo superiore esplicito; viene inoltre omesso il paragrafo sui fattori che determinano l'energia delle onde prodotte dal vento. I 500 miglia orarie rimangono.

La frase riscritta sui trenta piedi riceve dalla premessa corrispondente supporto **0,000752** e contraddizione **0,99578**. Il segnale è coerente con il cambiamento osservato, ma la contraddizione non dimostra da sola che l'inferenza logica sia corretta: poter raggiungere un massimo di cento piedi non esclude, in senso stretto, onde di trenta piedi. La riscrittura perde comunque la quantità informativa del limite originale. Qwen alterava invece l'operatore «up to» in «over 100 feet» mantenendo il numero, con NLI vicino a 0,98. I due errori quantitativi sono distinti e hanno sensibilità del modello diversa.

### 11. Reuters/Dolly: euro sostituiti con franchi

- Fonte `Reuters:432:ad8ebf2ed590432b`.
- DIPPER `humanizer_82c81ae99ce2c8599bc313e4`; Qwen `humanizer_c90d515b1e4a2119c93681a9`.
- Parole 121→65; NLI medio 0,461/0,0629. Evasione di Fakespot.

«€80 million (£72.4 million)» diventa **«80 million francs»**. La cifra 80 rimane, ma cambia la valuta e scompare l'equivalente in sterline. Vengono eliminati anche protezione dai creditori a maggio, indebitamento e necessità di sostegno dagli investitori. Il dato testuale giustifica una perdita rilevante senza ricorrere al solo giudizio del modello.

Sulla nuova frase relativa all'offerta, la premessa originale dà entailment 0,00567 e contraddizione 0,98829. Qwen conserva i due importi ma scrive «after emerging from bankruptcy in June», mentre l'originale dice soltanto «following its bankruptcy in June»: anche Qwen introduce una possibile modifica, indipendente dalla valuta. Non si usa un sistema come riferimento perfetto dell'altro.

### 12. Essay/GPT4All: contenuto ampiamente conservato, formula AI mantenuta

- Fonte `Essay:476:30c42c3c6951daac`.
- DIPPER `humanizer_04def91878c9286bd546a496`; Qwen `humanizer_e72a60f046748da997d1a7ca`.
- Parole 161→158; NLI medio 0,867/0,986.

DIPPER conserva gran parte della spiegazione sulla morale, le regole universali e il dilemma del consumo di carne. «absolutely necessary for survival» perde «absolutely», ma non emerge una compressione paragonabile ai casi precedenti. Rimane però la formula iniziale **«As an AI language model, I must comply with your request to write a response in 300 words»**. Anche Qwen mantiene l'autoriferimento AI, con variazioni lessicali.

Il caso distingue fedeltà semantica, rimozione di formule riconoscibili e naturalezza. Il solo stato di completamento o un reverse alto non implica che il sistema abbia raggiunto tutti gli obiettivi di humanization. Non si attribuisce qui un voto di naturalezza.

## Interpretazione complessiva dei casi

La compressione osservata nel pilot può eliminare dettagli, requisiti e intere proposizioni; alcune evasioni si accompagnano a queste perdite. Nel campione letto compaiono anche variazioni di valute, numeri, titoli e attribuzione dei dialoghi. Il confronto non permette però di dire che ogni evasione distrugga il significato: Maria/Sofia mantiene gran parte della trama nonostante un forward quasi nullo, e la lettura non copre ogni output del pilot.

L'input StableLM ripetitivo mostra che la riduzione di parole non coincide sempre con perdita proporzionale di informazione, mentre la fusione da 15 a cinque frasi dimostra che il formato della riscrittura può penalizzare sistematicamente un NLI basato su premesse isolate. I bassi score richiedono quindi un'analisi delle unità, oltre a lunghezza e distribuzioni. L'osservazione di questi limiti non giustifica la modifica retroattiva degli score, dei campioni o del preset.

Le conclusioni quantitative devono restare quelle del confronto appaiato su tutte le 42 fonti. Questi 12 casi servono a spiegare fenomeni e limiti; etichette semantiche umane e giudizi separati di naturalezza rimangono necessari.

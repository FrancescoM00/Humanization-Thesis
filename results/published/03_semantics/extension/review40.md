# Revisione qualitativa diretta di 40 perturbazioni

**Revisione qualitativa diretta dell’assistente, indipendente dai punteggi NLI; non annotazione umana.** Sono state lette le 40 coppie target con il contesto circostante; il documento intero è stato riletto per i casi ambigui. Non sono stati aperti i punteggi delle singole coppie. L’ipotesi generale dell’esperimento era nota: questa non è una valutazione cieca rispetto all’ipotesi scientifica.

Campionamento seed 2026: una sorgente per ciascuno dei 21 incroci dominio×generatore, omettendo casualmente una cella WritingPrompts; 20 sorgenti, esaminate sia in negazione sia in entity swap. Domini per condizione: 7 Essay, 7 Reuters, 6 WritingPrompts; sei generatori hanno 3 sorgenti e uno ne ha 2. I testi completi e il contesto restano in `data/semantic_extension/review40_blinded.jsonl`. I casi non sono stati sostituiti dopo la lettura.

Queste etichette sono giudizi linguistici esplorativi dell’assistente, non gold label né controllo indipendente umano. La relazione teorica locale riguarda ciò che la premessa supporta, non la verità nel mondo; una nuova entità può essere non supportata senza contraddire l’originale.

| Condizione | Applicate | Grammatica preservata / incerta / degradata | Significato cambiato / incerto | Test controllato valido / qualificato / invalido |
| --- | ---: | --- | --- | --- |
| negation | 20/20 | 18 / 2 / 0 | 19 / 1 | 15 / 4 / 1 |
| entity_substitution | 20/20 | 15 / 3 / 2 | 19 / 1 | 7 / 8 / 5 |

## Esito principale

Il campione conferma che gli edit sono stati materialmente applicati, ma **non** giustifica una gold label automatica per tutte le perturbazioni. Fra le negazioni, il caso `One of the main concerns is not X` non garantisce la negazione della proposizione originale: può riguardare un altro membro dell’insieme. Due posizionamenti `has not also…` hanno scope poco naturale. Un documento Reuters ripete la stessa asserzione cinque volte: negarne una lascia quattro evidenze affermative nel documento. Anche un testo Essay ripete due volte l’asserzione modificata, lasciandone una copia affermativa.

Fra le sostituzioni, 5/20 falliscono il criterio di entity swap omogeneo per errori NER o confini errati: malattia trattata da organizzazione, acronimo tecnico trattato da organizzazione, formato raster trattato da organizzazione, span che attraversa titolo e frase e verbo imperativo Format trattato da persona. Altri casi sono qualificati per collocazione temporale innaturale, granularità del referente, coreferenza del cognome irrisolta, nome comune Bank riferito alla banca NatWest o rinominazione coerente di un personaggio fittizio.

I casi invalidi restano nel campione e nell’analisi principale originale. Un’analisi che li esclude è soltanto una sensitivity analysis sul sottocampione revisionato, non una pulizia gold del corpus intero. Le percentuali sono descrittive su 20 casi per condizione, con la stessa sorgente ripetuta fra condizioni; non stimano l’errore su tutti i 200 testi senza ulteriore annotazione.

## Giudizi per ciascuna coppia

Legenda: `valid` = contrasto utilizzabile entro il criterio dichiarato; `qualified` = contrasto con artefatti o ambiguità che richiedono cautela; `invalid` = non valida il tipo di test controllato previsto.

### 01. negation — Essay / ChatGLM

ID: `ChatGLM:8a4ce2588b637e552f5cd6cc:negation:2c8c0073f4ceece9`

Estratti: «The adverse environmental effects of overpopulation are becoming more and more appare…» → «The adverse environmental effects of overpopulation are not becoming more and more ap…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

Nega direttamente la progressiva evidenza degli effetti ambientali, a soggetto e tempo invariati.

### 02. negation — Essay / ChatGPT

ID: `ChatGPT:d52bc089d6fb234fdebf5881:negation:129d376cf01fb046`

Estratti: «The pandemic has also had adverse effects on mental health, collective mental health,…» → «The pandemic has not also had adverse effects on mental health, collective mental hea…»

Edit applicato: **yes**; grammatica preservata: **uncertain**; significato cambiato: **yes**; relazione locale originale→modificato: **uncertain**; test controllato: **qualified**.

La posizione has not also had è poco naturale e lascia ambiguo lo scope su also; il contenuto è alterato ma non è un contrasto pulito P/non-P.

### 03. negation — Essay / ChatGPT-turbo

ID: `ChatGPT-turbo:dd1ceef0e6d5a4c1de9afa19:negation:c6573d300bc67094`

Estratti: «One of the main concerns is the impact on biodiversity.…» → «One of the main concerns is not the impact on biodiversity.…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **uncertain**; relazione locale originale→modificato: **uncertain**; test controllato: **invalid**.

One of the main concerns is not X può essere vero insieme a One of the main concerns is X: può trattarsi di un altro membro dell’insieme. Non è una contraddizione controllata affidabile.

### 04. negation — Essay / Claude

ID: `Claude:33ed8a7aaccf09e0b15a9501:negation:0cb525bc25cb2338`

Estratti: «Maps represent the landscape and its features in either a raster or vector format.…» → «Maps do not represent the landscape and its features in either a raster or vector for…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

Nega che le mappe rappresentino il paesaggio nei due formati appena affermati; il seguito descrive proprio quei formati.

### 05. negation — Essay / Dolly

ID: `Dolly:7df679cd2ed40f764c5efe6e:negation:d4bfbda105a71cff`

Estratti: «Corporate entrepreneurship is a key driver of economic growth.…» → «Corporate entrepreneurship is not a key driver of economic growth.…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

Inverte la proprietà attribuita allo stesso soggetto: essere un motore fondamentale della crescita.

### 06. negation — Essay / GPT4All

ID: `GPT4All:d1c85e7dc5bf1f12be45c2f8:negation:52cf6bb0f12aedb0`

Estratti: «The topic of the essay is the McDonald's lawsuit against their former CEO and the sha…» → «The topic of the essay is not the McDonald's lawsuit against their former CEO and the…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

Nega l’identità del tema del saggio, mantenendo invariati il saggio e il complemento predicativo.

### 07. negation — Essay / StableLM

ID: `StableLM:cef14c95fb0b5ea395a429a4:negation:60682cd91ff458f1`

Estratti: «The two main characters, Gorgias and Socrates, have very different perspectives on rh…» → «The two main characters, Gorgias and Socrates, do not have very different perspective…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **qualified**.

La frase nega chiaramente la differenza di prospettive, ma il documento contiene due volte l’asserzione originale: l’altra copia resta affermativa e può mascherare l’errore nel retrieval documentale.
### 08. negation — Reuters / ChatGLM

ID: `ChatGLM:e48c19245eaba0259323e297:negation:96d12583cb9b14f6`

Estratti: «IBM also reported a 35% increase in operating cash flow to $18.5 billion.  …» → «IBM also did not report a 35% increase in operating cash flow to $18.5 billion.  …»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

L’evento di comunicazione del medesimo incremento di cash flow viene negato; date e importi restano invariati.

### 09. negation — Reuters / ChatGPT

ID: `ChatGPT:f890ad5e0c28082a7298fb37:negation:2243c45af245c243`

Estratti: «The company reported a net profit of $2.4 billion for the quarter, up from $1.8 billi…» → «The company did not report a net profit of $2.4 billion for the quarter, up from $1.8…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

Nega la comunicazione del medesimo profitto riferito allo stesso trimestre, non cambia soltanto una valutazione.

### 10. negation — Reuters / ChatGPT-turbo

ID: `ChatGPT-turbo:ba9d6b58b8dda1d065a1da72:negation:642e95fde5437ebe`

Estratti: «Galhotra and Farley have both been mentioned as potential candidates, along with a nu…» → «Galhotra and Farley have not both been mentioned as potential candidates, along with …»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

Both became not both: l’originale afferma la candidatura di entrambi e la modifica nega che siano stati menzionati entrambi. È uno scope quantificato ma la contraddizione è interpretabile.

### 11. negation — Reuters / Claude

ID: `Claude:d270bdd31d50ac40126a58af:negation:520276d990f72e4a`

Estratti: «GM and the UAW have been down this road before, with strikes in 1970, 1972, and 1992,…» → «GM and the UAW have not been down this road before, with strikes in 1970, 1972, and 1…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

La negazione della precedente esperienza di scioperi contrasta con l’affermazione originale; l’elenco di scioperi conservato rende la frase alterata internamente incoerente.

### 12. negation — Reuters / Dolly

ID: `Dolly:7f13ade8c75e274104823bb7:negation:71bbe7401de69a67`

Estratti: «NatWest Markets appoints Peter Hall as COO amid questions over bank's strategy.  …» → «NatWest Markets does not appoint Peter Hall as COO amid questions over bank's strateg…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **qualified**.

La prima asserzione viene negata correttamente, ma il documento ripete altre quattro volte l’asserzione affermativa: il test documentale contiene evidenze identiche concorrenti.

### 13. negation — Reuters / GPT4All

ID: `GPT4All:1dd93da731652d68502d65ba:negation:7b105023afed10a0`

Estratti: «He has also accused the Secret Service of being involved in political interference in…» → «He has not also accused the Secret Service of being involved in political interferenc…»

Edit applicato: **yes**; grammatica preservata: **uncertain**; significato cambiato: **yes**; relazione locale originale→modificato: **uncertain**; test controllato: **qualified**.

Has not also accused mantiene una costruzione e uno scope poco naturali; l’accusa aggiuntiva viene alterata, ma non lo tratto come gold contradiction pulita.

### 14. negation — Reuters / StableLM

ID: `StableLM:ff88ee1abdd6fd11b7274444:negation:ac52801d061f9db5`

Estratti: «The History section contains information about the event in question.…» → «The History section does not contain information about the event in question.…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

Nega direttamente che la sezione History contenga l’informazione specificata; il carattere istruzionale del testo non invalida la differenza locale.

### 15. negation — WritingPrompts / ChatGPT

ID: `ChatGPT:1c6141882cb6454626b6801b:negation:4efa47acbcff1d09`

Estratti: «Finally, after what felt like an eternity of endless seas, the boat and its sailor wa…» → «Finally, after what felt like an eternity of endless seas, the boat and its sailor di…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

Nega l’approdo di barca e marinaio nello stesso episodio narrativo; il seguito mantiene l’arrivo sull’isola.

### 16. negation — WritingPrompts / ChatGPT-turbo

ID: `ChatGPT-turbo:18380584c647721668808d40:negation:5ec61295796d1a93`

Estratti: «The once-welcoming atmosphere in our home became eerily distant. …» → «The once-welcoming atmosphere in our home did not become eerily distant. …»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

Nega il passaggio a un’atmosfera distante nello stesso intervallo narrativo.

### 17. negation — WritingPrompts / Claude

ID: `Claude:79dac6234a1088def50b4de1:negation:42f0ac6732ee0822`

Estratti: «Tears of joy and wonder streamed down my face at the implications.…» → «Tears of joy and wonder did not stream down my face at the implications.…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

Nega il medesimo episodio di lacrime di gioia, senza cambiare partecipanti o tempo.

### 18. negation — WritingPrompts / Dolly

ID: `Dolly:3138e8a417b8db76d3451852:negation:99907fac3d91abbd`

Estratti: «Luckily, the man died in the water.…» → «Luckily, the man did not die in the water.…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

Inverte il decesso del medesimo uomo nell’episodio in acqua. Luckily resta pragmaticamente curioso, ma la grammatica della negazione è corretta.

### 19. negation — WritingPrompts / GPT4All

ID: `GPT4All:d4f3fae4b17cd720c15ce857:negation:fd4454f190fd8053`

Estratti: «The Super-Villain Alliance was made up of the most wicked and cunning individuals on …» → «The Super-Villain Alliance was not made up of the most wicked and cunning individuals…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

Nega la composizione del medesimo gruppo di supercriminali, mantenendo il complemento descrittivo.

### 20. negation — WritingPrompts / StableLM

ID: `StableLM:0c112a300e74d2ad8a6963e1:negation:f45548071a1dc675`

Estratti: «The floor was very high in my building.…» → «The floor was not very high in my building.…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

Nega la medesima valutazione graduata dell’altezza del piano, nello stesso contesto.

### 21. entity_substitution — Essay / ChatGLM

ID: `ChatGLM:8a4ce2588b637e552f5cd6cc:entity_substitution:06ea92ffed69e385`

Estratti: «The growth of populations has led to an increase in the demand for resources, which h…» → «The growth of populations has led to an increase in the demand for resources, which h…»

Edit applicato: **yes**; grammatica preservata: **uncertain**; significato cambiato: **yes**; relazione locale originale→modificato: **neutral**; test controllato: **qualified**.

Today diventa tomorrow in we are seeing: il riferimento temporale cambia ma la collocazione presente/futuro è innaturale. Il presente non esclude che il fenomeno continui domani.

### 22. entity_substitution — Essay / ChatGPT

ID: `ChatGPT:d52bc089d6fb234fdebf5881:entity_substitution:0e30fe435e33ef29`

Estratti: «The COVID-19 pandemic has affected lives and economies globally, and the rapid develo…» → «The United Nations pandemic has affected lives and economies globally, and the rapid …»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **uncertain**; test controllato: **invalid**.

COVID-19 è la malattia nel sintagma pandemic, non un’organizzazione; United Nations pandemic e United Nations vaccines non sono una sostituzione dello stesso tipo semantico.

### 23. entity_substitution — Essay / ChatGPT-turbo

ID: `ChatGPT-turbo:dd1ceef0e6d5a4c1de9afa19:entity_substitution:af9e73fb30860fd5`

Estratti: «Firstly, there should be increased research and development of alternative pest contr…» → «Firstly, there should be increased research and development of alternative pest contr…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **neutral**; test controllato: **invalid**.

IPM è l’acronimo del metodo integrated pest management, non un’organizzazione. Cambiarlo in UNESCO mantiene la sintassi ma corrompe una definizione/acronimo anziché scambiare due entità omogenee.

### 24. entity_substitution — Essay / Claude

ID: `Claude:33ed8a7aaccf09e0b15a9501:entity_substitution:15d096ef73fcc997`

Estratti: «Raster and vector digital maps each have strengths and weaknesses, and the approach s…» → «Toyota and vector digital maps each have strengths and weaknesses, and the approach s…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **uncertain**; test controllato: **invalid**.

Raster indica un formato cartografico, non un’organizzazione. Toyota maps produce un cambiamento di categoria semantica artificiale, ripetuto nel documento.

### 25. entity_substitution — Essay / Dolly

ID: `Dolly:7df679cd2ed40f764c5efe6e:entity_substitution:c19eaf85962e4ef7`

Estratti: «Samsung developed the Next Engine Technology, which is credited with increasing the l…» → «Samsung developed the Next Engine Technology, which is credited with increasing the l…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **neutral**; test controllato: **qualified**.

Samsung Galaxy è una linea di prodotti e Microsoft un’organizzazione: la proposizione sui telefoni cambia, ma il tipo/granularità del referente non è conservato. Il beneficio potrebbe riguardare entrambe le linee, quindi non è contraddizione logica.

### 26. entity_substitution — Essay / GPT4All

ID: `GPT4All:d1c85e7dc5bf1f12be45c2f8:entity_substitution:914cb5d36460038b`

Estratti: «McDonald's is a global fast food chain that has faced multiple lawsuits in recent yea…» → «Microsoft is a global fast food chain that has faced multiple lawsuits in recent year…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **neutral**; test controllato: **valid**.

Due nomi di organizzazione vengono scambiati nella stessa predicazione. Il testo non supporta l’asserzione sulla nuova organizzazione; la sua falsità fattuale non viene verificata qui.

### 27. entity_substitution — Essay / StableLM

ID: `StableLM:cef14c95fb0b5ea395a429a4:entity_substitution:c3d75060bb24d7fc`

Estratti: «The two main characters, Gorgias and Socrates, have very different perspectives on rh…» → «The fourteen main characters, Gorgias and Socrates, have very different perspectives …»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **qualified**.

Two diventa fourteen mantenendo l’apposizione che enumera soltanto Gorgias e Socrates: cambia il conteggio esatto, ma introduce anche un’incoerenza vistosa fra numero e lista.

### 28. entity_substitution — Reuters / ChatGLM

ID: `ChatGLM:e48c19245eaba0259323e297:entity_substitution:32f9a7e700cc863c`

Estratti: «IBM also reported a 35% increase in operating cash flow to $18.5 billion.  …» → «IBM also reported a 35% increase in operating cash flow to $83 million.  …»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

L’importo dello stesso cash flow nello stesso evento passa da 18,5 miliardi a 83 milioni; non è una semplice formattazione o conversione equivalente.

### 29. entity_substitution — Reuters / ChatGPT

ID: `ChatGPT:f890ad5e0c28082a7298fb37:entity_substitution:ddec3d4975a7d2fa`

Estratti: «The company's CEO, Bernard Looney, said that the results demonstrate the resilience a…» → «The company's CEO, Alex Morgan, said that the results demonstrate the resilience and …»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **neutral**; test controllato: **qualified**.

Il nome completo del CEO cambia ma Looney resta nel seguito come autore della citazione: il referente non è risolto coerentemente. L’originale non supporta l’attribuzione ad Alex Morgan.

### 30. entity_substitution — Reuters / ChatGPT-turbo

ID: `ChatGPT-turbo:ba9d6b58b8dda1d065a1da72:entity_substitution:3584ba70907955b8`

Estratti: «In a recent interview with Automotive News, Hackett emphasized that he is focused on …» → «In a recent interview with UNESCO, Hackett emphasized that he is focused on steering …»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **neutral**; test controllato: **valid**.

Automotive News e UNESCO sono organizzazioni; la modifica cambia l’interlocutore dell’intervista. L’originale non esclude un’altra intervista e non supporta questa nuova attribuzione.

### 31. entity_substitution — Reuters / Claude

ID: `Claude:d270bdd31d50ac40126a58af:entity_substitution:da6f126f33c1d0b0`

Estratti: «Analysts Predict Unlikely Protracted Strike by United Auto Workers Against General Mo…» → «Analysts Predict Unlikely Protracted Strike by Microsoft negotiations between the Uni…»

Edit applicato: **yes**; grammatica preservata: **no**; significato cambiato: **yes**; relazione locale originale→modificato: **uncertain**; test controllato: **invalid**.

Lo span United Auto Workers Against General Motors + a capo + Contract ingloba parti di titolo e frase successiva. La sostituzione distrugge il confine strutturale e lascia Microsoft negotiations; non è entity swap valido.

### 32. entity_substitution — Reuters / Dolly

ID: `Dolly:7f13ade8c75e274104823bb7:entity_substitution:75c99a1cbf086bcd`

Estratti: «NatWest Markets Appoints Peter Hall as COO Amid Questions Over Bank's Strategy NatWes…» → «NatWest Markets Appoints Peter Hall as COO Amid Questions Over UNESCO's Strategy NatW…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **neutral**; test controllato: **qualified**.

Bank è un nome comune capitalizzato nel titolo e riferito alla banca NatWest, non un nome proprio distinto. Sostituirlo con UNESCO cambia comunque un referente organizzativo, ma la selezione NER e la coreferenza sono deboli: caso qualificato, non errore certo del tipo di test.
### 33. entity_substitution — Reuters / GPT4All

ID: `GPT4All:1dd93da731652d68502d65ba:entity_substitution:64d8d843f8c6ddf4`

Estratti: «Babis was one of the most vocal critics of Klaus’ tenure, which lasted from 2006 to 2…» → «Babis was one of the most vocal critics of Klaus’ tenure, which lasted from 1998 to 2…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

Per la stessa tenure il limite iniziale 2006 diventa 1998, mentre il finale resta 2012: è una modifica del periodo, non riformulazione.

### 34. entity_substitution — Reuters / StableLM

ID: `StableLM:ff88ee1abdd6fd11b7274444:entity_substitution:88d732e53f61c94b`

Estratti: «Format the text of your article in a Microsoft Word document.…» → «Alex Morgan the text of your article in a Microsoft Word document.…»

Edit applicato: **yes**; grammatica preservata: **no**; significato cambiato: **yes**; relazione locale originale→modificato: **uncertain**; test controllato: **invalid**.

Format è il verbo imperativo che regge the text, non una persona. Alex Morgan the text elimina il predicato e rende la frase agrammaticale.

### 35. entity_substitution — WritingPrompts / ChatGPT

ID: `ChatGPT:1c6141882cb6454626b6801b:entity_substitution:72e0218287015f22`

Estratti: «The boat's little sailor, Timmy, was heartbroken when his beloved boat sailed away, b…» → «The boat's little sailor, Jordan Chen, was heartbroken when his beloved boat sailed a…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **uncertain**; relazione locale originale→modificato: **uncertain**; test controllato: **qualified**.

Timmy viene rinominato coerentemente Jordan Chen nel racconto. Cambia la stringa/referenza nominale, ma la struttura degli eventi è conservata: la perdita di significato dipende dalla policy sulle rinominazioni fittizie, non è gold contradiction.

### 36. entity_substitution — WritingPrompts / ChatGPT-turbo

ID: `ChatGPT-turbo:18380584c647721668808d40:entity_substitution:01a0ed89bcde97a1`

Estratti: «As the morning sun rose, a sense of serenity settled over our home.…» → «As the 3:45 p.m. sun rose, a sense of serenity settled over our home.…»

Edit applicato: **yes**; grammatica preservata: **uncertain**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **qualified**.

Morning diventa 3:45 p.m. davanti a sun: cambia mattina/pomeriggio, ma la collocazione the 3:45 p.m. sun rose è artificiale. Considero il contrasto temporale, non una sostituzione stilisticamente pulita.

### 37. entity_substitution — WritingPrompts / Claude

ID: `Claude:79dac6234a1088def50b4de1:entity_substitution:502226618554668f`

Estratti: «One night, I was feeling lonely and sorry for myself.…» → «3:45 p.m., I was feeling lonely and sorry for myself.…»

Edit applicato: **yes**; grammatica preservata: **uncertain**; significato cambiato: **yes**; relazione locale originale→modificato: **neutral**; test controllato: **qualified**.

One night diventa il nudo 3:45 p.m. senza At: il riferimento temporale cambia e la resa è telegrafica. Essere soli di notte non esclude esserlo anche di pomeriggio; non è contraddizione dimostrata.

### 38. entity_substitution — WritingPrompts / Dolly

ID: `Dolly:3138e8a417b8db76d3451852:entity_substitution:fb9c4bc1ca27b086`

Estratti: «I witnessed something absolutely awful today, but I am struggling to figure out if it…» → «I witnessed something absolutely awful yesterday, but I am struggling to figure out i…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **neutral**; test controllato: **valid**.

Today diventa yesterday per l’episodio narrato. Cambia la collocazione temporale del racconto, ma una frase isolata su un’esperienza oggi non nega altre esperienze ieri.

### 39. entity_substitution — WritingPrompts / GPT4All

ID: `GPT4All:d4f3fae4b17cd720c15ce857:entity_substitution:5488ed0c12a0e00f`

Estratti: «The Super-Villain Alliance was made up of the most wicked and cunning individuals on …» → «The Super-Villain Alliance was made up of the most wicked and cunning individuals on …»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **neutral**; test controllato: **valid**.

Earth e Lake Tanganyika sono riferimenti geografici, ma di granularità diversa. Cambia l’area del superlativo; il testo non supporta la localizzazione sul lago e non prova da solo una contraddizione.

### 40. entity_substitution — WritingPrompts / StableLM

ID: `StableLM:0c112a300e74d2ad8a6963e1:entity_substitution:699c4791f7c3b64b`

Estratti: «I came to know that his office was situated at the 4th floor.…» → «I came to know that his office was situated at the eleventh floor.…»

Edit applicato: **yes**; grammatica preservata: **yes**; significato cambiato: **yes**; relazione locale originale→modificato: **contradiction**; test controllato: **valid**.

La collocazione dello stesso ufficio passa dal quarto all’undicesimo piano. Nel contesto è un cambiamento della localizzazione, non un cambio di notazione dell’ordinale.

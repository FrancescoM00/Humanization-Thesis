# Audit qualitativo di nove controlli positivi OpAI

**Revisione testuale diretta dell’assistente, non annotazione umana.** Sono stati letti entrambi i documenti interi per ciascuna coppia, senza consultare i punteggi individuali delle metriche. L’ipotesi generale era nota. Non sono state modificate le classi di valutazione dopo la lettura.

Campionamento seed 2027: tre coppie non identiche per ciascuna operazione Polish, Paraphrase e Style; tre domini distinti scelti casualmente per operazione, poi una coppia casuale entro dominio. Le nove coppie provengono da sette sorgenti: due sorgenti ricompaiono in operazioni adiacenti. I testi integrali restano in `data/semantic_extension/positive_review9_blinded.jsonl`; il campione non è rappresentativo di tutti gli editor o di tutte le riscritture.

**Esito: 5 coppie con conservazione chiara del contenuto e 4 qualificate; nessuna con distruzione evidente del significato.** Qualified segnala dettagli procedurali più specifici, modifiche di intensità/cautela o riscrittura di una citazione, non una gold label di significato distrutto. Non va interpretato come stima del tasso di errore del corpus.

Il controllo supporta l’uso delle operazioni come esempi *presunti* di conservazione, ma non autorizza a identificare automaticamente ogni basso NLI con un falso allarme: parte dei cambi può essere semantica e il criterio stretto di entailment bidirezionale può essere più severo dell’istruzione editoriale. I giudizi non dipendono dall’NLI e non verificano la verità esterna degli articoli.

| ID revisione | Operazione | Dominio | Generatore | Conservazione | Aggiunte o cambi di forza da verificare |
| --- | --- | --- | --- | --- | --- |
| 1 | opai_polish | abstracts | qwen3-8b | yes | none_material / none_material |
| 2 | opai_polish | essays | qwen3-8b | yes | none_material / none_material |
| 3 | opai_polish | reports | gpt-5.4 | yes | none_material / minor_aspect_nuance_resolved_by_context |
| 4 | opai_paraphrase | abstracts | qwen3-8b | yes | none_material / minor_evaluative_nuance |
| 5 | opai_paraphrase | news | gpt-5.4-nano | qualified | agreement_signature_not_explicit_in_source / reached_to_signed |
| 6 | opai_paraphrase | reports | gemini-2.5-flash | qualified | possible_scalar_strengthening / limited_data_to_insufficient_information |
| 7 | opai_style | essays | gpt-5.4 | qualified | unquantified_magnitude_adverb / pollution_magnitude_strengthened_evidence_hedged |
| 8 | opai_style | news | gpt-5.4 | qualified | selection_agent_made_more_specific / collective_selection_to_named_coach_selection |
| 9 | opai_style | reports | gemini-2.5-flash | yes | none_material / none_material |

## Giudizi per ciascuna coppia

### 01. opai_polish — abstracts / qwen3-8b

ID: `opai:10351476784bc0c347828e8b`

Passaggio indicativo: «parallel processing of sequence» → «parallel processing of sequences»

Conservazione: **yes**; aggiunte: `none_material`; omissioni: `none_material`.

Il testo modifica sequences e la punteggiatura introduttiva di e.g./i.e.; modalità, elenco completo dei compiti, benefici, limiti e programma della survey restano invariati. Non emergono nuove proposizioni fattuali.

### 02. opai_polish — essays / qwen3-8b

ID: `opai:c3b28ddc22b59936f7ff1f01`

Passaggio indicativo: «The taxes we pay and all the bills we have to pay» → «The taxes we pay, along with all the bills we have to pay»

Conservazione: **yes**; aggiunte: `none_material`; omissioni: `none_material`.

Viene regolarizzata la frase sulle tasse e sulle bollette spese per invenzioni inutili. La critica, i dubbi sulla domanda, i riferimenti alla privacy e la conclusione sono conservati. La tesi sulle spese era già presente, anche se poco chiara, nell’originale.

### 03. opai_polish — reports / gpt-5.4

ID: `opai:852e42217b5593342f224442`

Passaggio indicativo: «having drafted a strategic plan» → «by drafting a strategic plan»

Conservazione: **yes**; aggiunte: `none_material`; omissioni: `none_material`.

Le riformulazioni su ingressi/uscite, contesto strategico e program execution conservano i contenuti. Having drafted diventa by drafting, ma entrambe le versioni precisano subito che il piano attende approvazione: nel documento non si perde lo stato del piano. Restano tutti i numeri delle sedi e le cautele sui costi e sui rischi.

### 04. opai_paraphrase — abstracts / qwen3-8b

ID: `opai:99d33c433b1263bb76027964`

Passaggio indicativo: «Among their salient benefits» → «Transformers offer several notable advantages»

Conservazione: **yes**; aggiunte: `none_material`; omissioni: `none_material`.

Si riscrivono tre passaggi della survey lasciando invariati meccanismi, modalità, lista dei task e obiettivi di confronto. Key improvement è una sfumatura valutativa compatibile con salient benefits del contesto, non una nuova misura empirica. Experimental value diventa experimental outcomes senza aggiungere risultati specifici.

### 05. opai_paraphrase — news / gpt-5.4-nano

ID: `opai:2e31cbc1e03f0f294b9cb173`

Passaggio indicativo: «a June 2011 agreement reached» → «a June 2011 agreement signed»

Conservazione: **qualified**; aggiunte: `agreement_signature_not_explicit_in_source`; omissioni: `none_material`.

Importi, numero di navi e marinai, date, attribuzioni e incertezza sull’accordo sono conservati. Tuttavia agreement reached diventa agreement signed: la firma è un dettaglio procedurale più specifico non esplicitamente affermato dal testo sorgente. Può essere corretto nel contesto storico, ma qui non viene verificato né garantito dal solo originale.

### 06. opai_paraphrase — reports / gemini-2.5-flash

ID: `opai:5be3b55b0844147a1b0c8d6f`

Passaggio indicativo: «Congress may have limited data» → «Congress might have insufficient information»

Conservazione: **qualified**; aggiunte: `possible_scalar_strengthening`; omissioni: `none_material`.

Programma, scadenze e riferimenti numerici sono conservati. La conclusione may have limited data diventa might have insufficient information: scarsità e insufficienza non sono logicamente identiche, anche se il senso critico generale è simile. Contributed to lateness diventa delay was due to, con una lieve formulazione causale più forte. Non rilevo una nuova data o un fatto palesemente contraddittorio.

### 07. opai_style — essays / gpt-5.4

ID: `opai:d1fd09a82159dd8ccbbba992`

Passaggio indicativo: «Eliminating cars could cut pollution» → «Removing cars could significantly reduce pollution»

Conservazione: **qualified**; aggiunte: `unquantified_magnitude_adverb`; omissioni: `none_material`.

Nomi, percentuali, multa, città e argomento generale sono preservati. Could cut pollution diventa could significantly reduce pollution, aggiungendo una qualificazione della grandezza. La conclusione passa inoltre da studies indicate a studies suggest, indebolendo il grado di supporto espresso. Sono sfumature, non una distruzione evidente del significato, ma impediscono di usare l’etichetta come gold entailment bidirezionale rigoroso.

### 08. opai_style — news / gpt-5.4

ID: `opai:c499c6bc4856797fb2280f0e`

Passaggio indicativo: «we’ll be picking what Warren Gatland feels» → «Warren Gatland will select what he believes»

Conservazione: **qualified**; aggiunte: `selection_agent_made_more_specific`; omissioni: `verbatim_quote_not_preserved`.

Giocatori, infortuni, date e quasi tutte le cautele restano uguali. Nella citazione diretta, we will be picking what Warren Gatland feels is our best side diventa Warren Gatland will select what he believes is our strongest side: passa da selezione collettiva secondo il giudizio del capo allenatore a selezione attribuita direttamente a lui. È conservata la sostanza sportiva, ma la citazione non è più letterale e l’agente della scelta è più specifico.

### 09. opai_style — reports / gemini-2.5-flash

ID: `opai:0ddbdde48ef1fdb1e150a71f`

Passaggio indicativo: «Congress might have insufficient information» → «Congress may lack sufficient information»

Conservazione: **yes**; aggiunte: `none_material`; omissioni: `none_material`.

Il registro diventa più formale, ma rimangono obiettivi DOD, eccezione Corea del Sud, scadenze 2006/2008/2010/2014, vincoli del piano Guam, incertezze e avvertenza finale. Le coppie may lack sufficient information e might have insufficient information mantengono la stessa cautela; non emergono nuovi dettagli materiali.

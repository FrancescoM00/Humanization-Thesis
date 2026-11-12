# Audit testuale e meccanico della politica entity clean

**Revisione dell’assistente, non annotazione gold dell’utente.** La patch ricevuta non contiene le nuove etichette umane del review40. Non sono stati consultati score NLI, embedding o BERTScore per giudicare i testi. Le etichette storiche del CSV review40 erano già una revisione dell’assistente e vengono riportate soltanto per rintracciare i casi.

## Campione e controllo meccanico

La politica finale `clean_v2`, `guard_revision=review_fixes_3`, è stata rigenerata in memoria sugli stessi 200 originali del run storico: **145 sostituzioni applicate, 55 saltate**. La preparazione CLI definitiva conferma questi conteggi; originali, modificati, target locali e metadati dei 145 coincidono con lo screening indipendente in memoria. Il primo tentativo con 149 sostituzioni è stato fermato prima di completare le inferenze e non costituisce il risultato definitivo.

Sui 145 applicati sono stati controllati tutti gli offset, la corrispondenza degli span con l’originale, il replay degli edit nel documento, lo span della frase locale e il replay locale. Risultato: **zero discrepanze**. Sono state controllate anche espansioni parziali di PERSON dentro altre entità, possessivi inglobati e nuove duplicazioni di simboli valutari/percentuali/unità: **zero artefatti meccanici di questi tipi nel campione finale**. Due sostituzioni dentro span possessivi più lunghi sono lecite perché preservano il suffisso esterno: `Lena` → `Alex Morgan` in `Lena’s`, e `Saint Peter` → `Sam Rivera` in `Saint Peter's`.

Distribuzione dei 145: PERSON 48; GPE 36; CARDINAL 23; DATE 18; PERCENT 10; MONEY 6; ORDINAL 4. Il controllo di unità esterne comprende tutti i 6 MONEY e 10 PERCENT finali: non produce `$$`, `% percent` o unità million/billion duplicate.

## Difetti concreti corretti prima del run finale

- Mesi e omografi: `May` non può trasformare l’ausiliare `may`; `March` usato come verbo non viene riscritto come mese senza evidenza DATE.
- Cognomi di persone distinte: rinominare Bernard Looney non modifica Jane Looney. Il cognome di destinazione non deve già comparire nel documento.
- Possessivi inglobati dal NER: sei span nel primo tentativo avrebbero eliminato `'s`/`’s`, per esempio `Phineas Taylor Barnum's` → `Sam Rivera`, `Hong Kong's` → `Canada` e `Lena’s` → `Alex Morgan`. Questi span interi vengono rifiutati; altre entità sicure possono essere selezionate nello stesso documento.
- Valuta fuori dallo span MONEY: due span del primo tentativo, `69.99` e `800`, lasciavano il `$` esterno e producevano un doppio simbolo. La clean rifiuta questi candidati.
- Nomi parziali: scegliere `Cooper`, `Ma` o `Looney` e inserirvi un nome completo produceva `Alison Alex Morgan`, `Jack Sam Rivera`, `Bernard Alex Morgan`. La clean ora rifiuta candidati che richiederebbero tali espansioni dentro PERSON/ORG più lunghe, salvo il possessivo esterno preservato.
- Il parametro `conditions` è materializzato per poter riusare un iteratore su più sorgenti; valori sconosciuti generano errore invece di diventare cancellazioni casuali.

Il file di test originale della patch conserva **15 test passati**. I regressivi separati aggiungono 19 casi: **34 test passati** complessivi. Le funzioni v1 delle perturbazioni restano identiche; `generate_perturbation_pairs` aggiunge soltanto la selezione esplicita delle condizioni e il relativo controllo.

## Residui semantici: clean non significa gold

La lettura dei target mostra ancora errori del modello NER e incompatibilità di sottotipo. Le guardie meccaniche non possono trasformare il NER piccolo in un validatore semantico. Esempi presenti nel campione finale:

| Sorgente | Scambio | Problema osservato |
| --- | --- | --- |
| `StableLM:74d27b269e0f4736e384b1fe` | Twitter → Sam Rivera | Il servizio social è etichettato PERSON: «a link to your Sam Rivera, Instagram or Facebook account». |
| `ChatGPT-turbo:609143f65044f98a85cdb2f0` | Centre → Canada | Un frammento di nome organizzativo è etichettato GPE: «Canada for the Study of Terrorism and Political Violence». |
| `ChatGPT:10b881654e5b819eb09d5bf5` | Tituba → Japan | Una persona/personaggio è etichettata GPE e diventa un paese. |
| `Claude:9b94e71ce2eff4112522c465` | Bon → Jordan Chen | «Bon voyage» è spezzato da una falsa PERSON. |
| `Claude:9b7e360301478d0147e06ac2` | Hush → Sam Rivera | L’imperativo nella citazione viene etichettato PERSON dal modello. |
| `GPT4All:12d0f8864146c5b3b394e804` | Ukrainian → Brazil | Un aggettivo nazionale GPE diventa sostantivo: «Russian and Brazil traditions». |
| `ChatGPT-turbo:3af689cb8c8fbd5fff6d1b99` | Verizon → Canada | Un’azienda è etichettata GPE. |
| `ChatGLM:5670f2380ef67a632df5847a` | Bitcoin → Alex Morgan | Una criptovaluta è etichettata PERSON. |
| `StableLM:010b6e9f844878e1593ee14c` | Epson → Sam Rivera | Un’azienda è etichettata PERSON. |
| `ChatGPT-turbo:ba9d6b58b8dda1d065a1da72` | Cox Automotive → Sam Rivera | La clean sceglie un’azienda etichettata PERSON: il difetto non è risolto escludendo solo ORG. |
| `ChatGPT:a3ad8e25a89072981f7e5241` | New York → Japan | GPE include città e paesi: il dateline diventa «Japan, NY». |
| `ChatGLM:a3e54ca3ef9b5dcdc3e5c2b8` | Collingswood → Japan | GPE cambia granularità: «The small town of Japan». |

Questi sono esempi diagnostici selezionati, non una stima annotata del tasso di errore. Non sono state aggiunte whitelist ontologiche costruite ad hoc per cancellare gli esempi del campione. Rimangono anche rinominazioni di personaggi fittizi, coreferenza incompleta, conteggi dentro enumerazioni o marcatori di elenco, e span monetari/percentuali che comprendono qualificatori come over, only o approximately. Cambiare un nome non dimostra una contraddizione; mantenere uno span grammaticalmente valido non certifica una sostituzione semanticamente omogenea.

## Rintracciare le entity del review40 storico

Nel CSV disponibile ci sono **20 entity, review_id 21–40**: 14 sorgenti ammettono una clean e 6 sono saltate. La clean può scegliere un’entità diversa, non soltanto una nuova sostituzione della stessa entità v1; inoltre usa un RNG distinto per condizione. La tabella non trasferisce automaticamente il giudizio storico al nuovo target.

| Review | Giudizio storico dell’assistente | Entità v1 | Entità clean finale | Stato clean |
| --- | --- | --- | --- | --- |
| 21 | qualified | today → tomorrow | — | skipped |
| 22 | invalid | COVID-19 → United Nations | — | skipped |
| 23 | invalid | IPM → UNESCO | — | skipped |
| 24 | invalid | Raster → Toyota | — | skipped |
| 25 | qualified | Samsung Galaxy → Microsoft | Rhys Oler → Jordan Chen | applied |
| 26 | valid | McDonald's → Microsoft | Ronald McDonald → Jordan Chen | applied |
| 27 | qualified | two → fourteen | Nick Naylor → Jordan Chen | applied |
| 28 | valid | $18.5 billion → $83 million | $49.3 billion → $47 million | applied |
| 29 | qualified | Bernard Looney → Alex Morgan | $1.8 billion → $47 million | applied |
| 30 | valid | Automotive News → UNESCO | Cox Automotive → Sam Rivera | applied |
| 31 | invalid | United Auto Workers Against General Motors /  / Contract → Microsoft | Kristin Dziczek → Sam Rivera | applied |
| 32 | qualified | Bank → UNESCO | Peter Hall → Alex Morgan | applied |
| 33 | valid | 2006 → 1998 | 2012 → 2032 | applied |
| 34 | invalid | Format → Alex Morgan | two → fourteen | applied |
| 35 | qualified | Timmy → Jordan Chen | Timmy → Jordan Chen | applied |
| 36 | qualified | morning → 3:45 p.m. | Greg → Alex Morgan | applied |
| 37 | qualified | One night → 3:45 p.m. | — | skipped |
| 38 | valid | today → yesterday | — | skipped |
| 39 | valid | Earth → Lake Tanganyika | Elon Musk → Jordan Chen | applied |
| 40 | valid | 4th → eleventh | 4th → third | applied |

Fra i casi storici evidenti, COVID-19/United Nations, IPM/UNESCO e Raster/Toyota vengono ora saltati; il confine NER che attraversava titolo e frase viene evitato selezionando Kristin Dziczek. L’errore Format/PERSON non è più scelto. Il caso Timmy rimane una rinominazione narrativa, mentre la nuova selezione Cox Automotive resta un errore di tipo NER.

## Come leggere il confronto delle metriche

Confrontare solo entity_substitution, distinguendo tutti i 200 v1 dai 145 clean applicati e dal sottoinsieme appaiato sulle stesse sorgenti. La copertura più bassa e la distribuzione diversa di tipi sono parte del risultato. Non chiamare un eventuale aumento della contraddizione NLI una validazione della qualità delle perturbazioni: sarebbe circolare. Per precisione semantica e relazione logica servono le annotazioni umane effettive, separate dai punteggi del modello.

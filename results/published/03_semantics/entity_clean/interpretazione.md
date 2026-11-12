# Entity clean v2: patch corretta e confronto eseguito

La patch è stata applicata e collegata alla pipeline con `--entity-policy clean`. La v1 rimane la scelta predefinita. I **15 test originali passano**; sono passate anche **19 regressioni** sui difetti trovati durante la revisione e cinque test del comparatore. I risultati storici non sono stati sovrascritti.

Il run definitivo usa `entity_policy=clean_v2`, `guard_revision=review_fixes_3`, nelle cartelle `data/semantic_clean/` e `results_semantic_clean/`.

## Campione e invarianza

Sono state mantenute le stesse **200 fonti** e le stesse **800 coppie OpAI**. La clean trova un candidato applicabile in **145/200 testi (72,5%)**; gli altri **55** sono registrati in `skipped.jsonl`, senza campionare sostituti. In totale il nuovo file contiene **2.297 coppie**, inclusi controlli locali e fixture.

Le **2.007 coppie estranee alla modifica sono identiche** per ID, testi, target e metadati. Tutte le **61 colonne di metriche** del confronto rientrano nella tolleranza numerica `atol=rtol=1e-6`: negazione, shuffle, troncamento, cancellazione, identità, controlli non-entity, fixture e OpAI conservano i risultati. I contatori cumulativi di lavoro sono esclusi dal confronto scientifico.

Embedding e BERTScore sono stati ricalcolati sull'intero campione. Per l'NLI, sulle coppie immutate sono mantenute le evidenze storiche; sulle nuove entity e sui relativi controlli sono usati gli allineamenti nuovi. Sono state riusate soltanto probabilità delle stesse coppie di stringhe sullo stesso checkpoint. **DeBERTa-v3 base float32** è rimasto invariato: 87.825 confronti unici, 86.413 già in cache e **1.412 inferenze nuove**. Il riuso delle evidenze è stato stabilito prima dell'inferenza nuova, senza selezionarle in base agli score NLI.

## Confronto delle sole entity sulle stesse 145 fonti

Le medie della v1 qui sotto riguardano le stesse 145 fonti applicabili alla clean, non tutte le 200 entity del run precedente.

| Misura | V1 | Clean |
| --- | ---: | ---: |
| Entailment locale, originale → modificato | 0,017020 | **0,003059** |
| Supporto minimo sul documento, forward | 0,019723 | **0,009033** |
| Supporto minimo sul documento, bidirezionale | 0,016414 | **0,007270** |
| Coseno sul documento | 0,977648 | **0,983924** |
| Coseno sulla frase modificata | 0,864937 | **0,882543** |
| BERTScore F1 sul documento | 0,997241 | **0,998089** |
| BERTScore F1 sulla frase modificata | 0,984824 | **0,989241** |

Il minimo bidirezionale è il minore fra minimo forward e minimo reverse, calcolato per ciascun documento e poi mediato. BERTScore usa il medesimo protocollo non riscalato del run storico: questi valori servono al confronto omogeneo fra politiche, non sono probabilità di correttezza semantica.

Le similarità rimangono alte, soprattutto sul documento; l'NLI locale assegna invece pochissimo supporto alla frase con il referente cambiato. Anche il minimo documentale rimane basso. Questo andamento è coerente con il risultato precedente sulle sostituzioni, pur usando una politica più conservativa.

La diminuzione media dell'entailment locale è −0,01396, IC bootstrap appaiato 95% [−0,03244; −0,00166]. Per il minimo documentale bidirezionale il delta è −0,00914, IC [−0,02736; +0,00313]: **non emerge una differenza netta del minimo fra politiche**. I cambiamenti medi del coseno hanno anch'essi IC che includono zero. L'aumento medio di BERTScore è piccolo; tutti gli IC e le direzioni inverse sono nel [confronto completo](entity_policy_comparison.md) e nei CSV.

Un entailment più basso non dimostra che le nuove perturbazioni siano più valide. Anche sulla stessa fonte possono cambiare entità, frase bersaglio, tipo e numero di menzioni sostituite; il confronto valuta l'intera politica, non isola causalmente una sola guardia.

## Controllo sulla famiglia numerica più omogenea

Il [controllo delle sole date, somme e percentuali](numeric34/report.md) comprende 34 coppie (18/6/10), senza nuove inferenze. L'entailment locale medio resta 0,002572 contro 0,991547 sulle stesse frasi identiche; il supporto minimo documentale bidirezionale è 0,003436. Coseno documento/frase 0,998363/0,954655 e BERTScore 0,999082/0,992778 rimangono alti. Il risultato persiste nel sottoinsieme, con limiti sulle perturbazioni e sulla generalizzazione esplicitati nel report. Le identità sono controlli facili e non bastano per confrontare la discriminazione di parafrasi valide.

## Valutazione della patch

La separazione dalla v1 è corretta e utile. La patch originale superava i suoi 15 test, ma lasciava errori meccanici non coperti:

- un mese come `May` poteva cambiare anche il modale `may`;
- la sostituzione di un cognome poteva modificare un'altra persona o espandere un nome parziale dentro un nome completo;
- un cognome di destinazione già presente poteva creare collisioni;
- possessivi inglobati nello span NER perdevano `'s`/`’s`;
- un simbolo monetario esterno allo span poteva produrre `$$83 million`;
- un iteratore `conditions` si esauriva dopo il primo documento; condizioni sconosciute finivano nella cancellazione casuale.

Questi casi sono stati corretti con guardie conservative e regressioni separate. Nella preparazione definitiva non sono emersi errori di offset, replay delle modifiche, target locale, possessivi inglobati, nomi parziali espansi o duplicazioni di simboli/unità dei tipi controllati.

**Restano errori NER e di sottotipo:** per esempio Twitter o Cox Automotive possono essere etichettati PERSON; una città può essere sostituita con un paese. Sono documentati nell'[audit testuale](clean_entity_text_audit.md), svolto dall'assistente senza usare gli score per giudicare i testi. Non è annotazione gold dell'utente: le nuove etichette manuali del review40 non sono contenute nella patch caricata. Il nome clean descrive quindi una politica più conservativa, non un insieme certificato di negativi corretti.

## File

- Copie scaricabili direttamente dal repository: exports/semantic_clean/, con JSONL e ZIP.
- Dati completi: `data/semantic_clean/pairs.jsonl`.
- Casi non applicabili: `data/semantic_clean/skipped.jsonl`.
- Archivio leggero con entrambi: `data/semantic_clean/semantic_clean_pairs.zip`.
- Patch corretta con collegamento CLI e test: `entity_clean_v2_corrected.patch`.
- Protocollo e comandi: [ENTITY_CLEAN_EXPERIMENT.md](../../../docs/protocols/ENTITY_CLEAN_EXPERIMENT.md).
- Numeri, IC, copertura e invarianza: [entity_policy_comparison.md](entity_policy_comparison.md).

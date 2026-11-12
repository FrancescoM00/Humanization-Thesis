# Controlli positivi e perturbazioni sottili: interpretazione

**La calibrazione sceglie il minimo bidirezionale NLI.** Sul test OpAI appaiato ottiene AUC **0,958** (IC 95% 0,931–0,979); sul confronto separato dei soli errori sottili, **0,900** (0,820–0,964). È un candidato operativo per questo protocollo, non un verificatore universale del significato. Le cautele rimosse restano un punto debole.

## Campione e confronto

Sono state elaborate 2.621 coppie: 1.607 controlli precedenti, 300 positivi OpAI (100 Polish, 100 Paraphrase, 100 Style Rewrite), 562 versioni alterate degli stessi output e 152 nuove perturbazioni sottili MGTBench. DeBERTa-v3 **base** è stato mantenuto.

I positivi OpAI usano transizioni adiacenti v0→v1, v1→v2, v2→v3. Il negativo principale conserva lo stesso originale e lo stesso output positivo, al quale viene aggiunta una perturbazione. Si evita così di misurare soltanto la differenza tra positivi OpAI e negativi MGTBench. Le etichette sono **conservazione presunta e alterazione intenzionale**, non gold label umane.

Il confronto principale comprende 280 positivi collegati e 562 negativi, su 94 sorgenti distinte. Calibrazione: 139 positivi e 297 negativi, 46 sorgenti. Test: 141 positivi e 265 negativi, 48 sorgenti. Tutti gli editor e le trasformazioni della stessa sorgente rimangono nello stesso split. I 20 positivi senza corruzione applicabile restano disponibili per l'analisi degli allarmi sul campione completo.

La scelta fra quattro aggregatori usa soltanto la calibrazione. AUC e IC bootstrap percentile sono calcolati per fonte, con uguale peso alle fonti entro ciascuna classe. Gli IC hanno limiti di bias finito, particolarmente sulle metriche deboli nei confronti appaiati; non dimostrano piccole differenze fra aggregatori. Le analisi con negativi MGTBench rimangono secondarie e conservano il confondente del corpus.

## Quale aggregatore usare

Per media, q10 e minimo si prende il supporto minore fra le due direzioni. Per la quota non supportata si prende la maggiore; una frase è non supportata se il suo supporto è sotto 0,5. Le soglie di allarme successive sono invece calibrate sui positivi, mirando a un tasso nominale ≤10%.

| Aggregatore bidirezionale | AUC test completo | AUC test sottili | Sensibilità test completo alla propria soglia calibrata |
| --- | ---: | ---: | ---: |
| Media | 0,929 | 0,885 | 63,8% |
| q10 | 0,728 | 0,657 | 39,4% |
| **Minimo** | **0,958** | **0,900** | **83,1%** |
| Quota non supportata | 0,908 | 0,833 | 51,5% |

La calibrazione dei soli sottili sceglie anch'essa il minimo. Questo confronto secondario ha 42 positivi e 49 negativi nel test, su 25 fonti: il risultato resta esplorativo. La prevalenza di negazioni ed entità nel confronto completo (440/562 negativi) rende necessario affiancare questi risultati, anziché interpretare soltanto l'AUC complessiva.

Con il minimo principale, la regola è **segnalare se il supporto minimo fra entrambe le direzioni è <0,04075**. Questa soglia è appresa dalla calibrazione, non scelta dopo aver visto il test. Sul confronto principale collegato, il tasso di allarme sui positivi presunti è 7,6%, la sensibilità alle alterazioni 83,1%. Trasferendo la stessa soglia ai sottili appaiati, senza ricalibrarla: 8,0% di allarmi sui positivi e 70,7% di sensibilità.

Su **tutti i 153 positivi presunti del test**, inclusi quelli senza corruzione applicabile, gli allarmi sono **11/153 = 7,2%**. Per operazione: Polish 1/51, Paraphrase 4/51, Style Rewrite 6/51. Una soglia ingenua di 0,5 sul minimo ne segnalerebbe 20/153 (13,1%); sull'intero campione positivo, 53/300 (17,7%). Il minimo è fragile se interpretato con una soglia universale: la calibrazione ne cambia sensibilmente l'uso, senza eliminare gli errori.

Questi sono allarmi rispetto a etichette presunte. Un giudizio NLI basso può anche cogliere una sfumatura realmente modificata da una riscrittura nominalmente conservativa. La soglia non è validata su un diverso humanizer o su un corpus esterno.

## Dove funziona e dove rimane un problema

Ogni AUC confronta il sottotipo con gli esatti positivi collegati. Sensibilità alla soglia principale, con pesi per fonte: non sempre coincide con il semplice conteggio delle coppie.

| Corruzione OpAI appaiata | Negativi nel test | AUC del minimo | Sensibilità alla soglia principale |
| --- | ---: | ---: | ---: |
| Negazione | 94 | 0,974 | 89,0% |
| Entità | 122 | 0,958 | 87,8% |
| Numero | 24 | 0,970 | 88,9% |
| Quantificatore | 15 | 0,931 | 50,0% |
| **Cautela rimossa** | **9** | **0,571** | **13,3%** |
| Dettaglio aggiunto | 1 | Non conclusiva | Non conclusiva |

L'esperimento rileva bene i cambi numerici espliciti generati da queste regole; non è un test di calcolo, arrotondamenti o ragionamento numerico complesso. La rimozione delle cautele è il limite più evidente: l'IC AUC è ampio (circa 0,344–0,702) e la numerosità ridotta. L'AUC aggregata sui sottili non autorizza a dire che tutte le forme sottili siano coperte. Per quantificatori, il ranking è buono ma la soglia conservativa perde circa metà degli esempi. Il singolo dettaglio nel test non consente una conclusione per quel tipo.

## Similarità, baseline e BLEU

Sul test principale: coseno AUC 0,599; BERTScore chunk 0,544; ROUGE-L 0,537; BLEU 0,536. Sui sottili appaiati, queste metriche rimangono circa fra 0,52 e 0,53. In questo esperimento discriminano poco le alterazioni rispetto alle riscritture conservative; un valore documentale alto non basta come garanzia semantica.

La baseline ufficiale inglese RoBERTa-base, layer 10, ha F=0,814463. È stata applicata la formula esatta di `rescale_with_baseline=True`, equivalente a ripetere lo scoring con quel flag. Si conservano anche le distanze raw−baseline.

| Esempio/condizione | BERTScore raw | Distanza dalla baseline | Riscalato |
| --- | ---: | ---: | ---: |
| Negazione, documento | 0,9989 | 0,1844 | 0,9939 |
| Negazione, frase modificata | 0,9843 | 0,1698 | 0,9153 |
| Fixture non correlata, frase | 0,8653 | 0,0509 | 0,2743 |

Il rescaling chiarisce la scala, **non cambia ranking o AUC**. Per i documenti rimane un'approssimazione con matching di chunk. La singola fixture non correlata non stima una distribuzione del rumore. Anche il coseno locale cambia con la negazione (circa 0,869 nel primo esperimento): il limite documentale non va trasformato nella frase assoluta «il coseno è cieco alla negazione».

BLEU e ROUGE-L hanno correlazione Spearman 0,936 nell'insieme valutato e **0,986 nel confronto OpAI appaiato**. Con AUC principali quasi identiche e basse, BLEU può restare una misura secondaria per confrontabilità. La correlazione è descrittiva e dipende dal mix delle trasformazioni.

## Revisione testuale indipendente dai punteggi

Sono state lette direttamente 40 perturbazioni (20 sorgenti, entrambe le condizioni), senza usare i punteggi individuali NLI. **È una revisione dell'assistente, non annotazione umana né gold indipendente.**

| Condizione | Contrasti puliti | Con riserve | Invalidi per il test previsto |
| --- | ---: | ---: | ---: |
| Negazione | 15 | 4 | 1 |
| Entità | 7 | 8 | 5 |

Tutti gli edit sono materialmente applicati; il cambiamento di significato è chiaro in 38 casi e incerto in 2. Le riserve comprendono scope e frasi duplicate rimaste affermative. Cinque entity swap hanno errori NER certi, fra cui COVID-19 trattato da organizzazione, Format imperativo trattato da persona e uno span che attraversa titolo e frase. Questi problemi impediscono di usare l'alta contraddizione NLI come validazione della correttezza della perturbazione. I dati originali non sono stati ripuliti a posteriori: sono disponibili analisi secondarie sui casi revisionati validi e validi+qualificati. Nei nuovi negativi OpAI lo screening NER è più conservativo, senza usare NLI per scegliere i casi, ma le etichette restano proxy.

Sono stati letti integralmente anche nove positivi OpAI: cinque conservazioni chiare e quattro con riserve, nessuna distruzione evidente del significato. Le riserve riguardano specificità, intensità, cautela e agency dentro una citazione. Questo piccolo audit non stima un tasso d'errore del benchmark, ma mostra perché «operazione conservativa» non equivale a gold entailment bidirezionale.

## Stile nelle cinque operazioni OpAI

Analizzate 200 traiettorie comuni, 188 sorgenti, 1.000 coppie e tutte le 105 feature. Esempi sui documenti interi:

| Operazione | Feature | Variazione dopo−prima | IC 95% |
| --- | --- | ---: | --- |
| Polish | Lunghezza media parole | +0,0132 caratteri | [0,0095; 0,0169] |
| Paraphrase | MATTR50 | +0,00795 | [0,00670; 0,00926] |
| Style Rewrite | Lunghezza media parole | +0,1088 caratteri | [0,0892; 0,1301] |
| Compress | Quota nomi propri | +0,00438 | [0,0037; 0,0052] |
| Expand | Lunghezza media frasi | +3,77 parole | [3,37; 4,17] |

È disponibile un controllo con prefissi a fine frase e conteggio lessicale identico su 336/1.000 coppie. Fra le quattro feature della shortlist, soltanto Compress/pos_cconj mantiene q BH<0,05 nel controllo. Cambiano numerosità, selezione e passaggio osservato: questo non attribuisce causalmente i risultati interi alla sola lunghezza. Style Rewrite continua ad aumentare la lunghezza media delle parole anche nel controllo. Una variazione delle feature non prova maggiore naturalezza.

Compress ed Expand restano controlli direzionali, non gold esempi di errore. Le compressioni moderate possono preservare l'informazione essenziale: un reverse alto non è di per sé un difetto della metrica. Forward basso è compatibile con aggiunte o modifiche non supportate; reverse basso con omissioni o modifiche. Le direzioni, da sole, non identificano la causa.

## File e uso dei risultati

[AUC e falsi allarmi completi](aggregator_report.md), [AUC per sottotipo/dominio/editor](auc_by_stratum.csv), [scelta dell'aggregatore](aggregator_selection.json), [allarmi su tutti i positivi](positive_controls_all_fpr.csv), [revisione dei 40 casi](review40.md), [audit dei nove positivi](positive_review9.md), [report stilometrico](../../04_opai_style/document/report.md).

Comandi in [SEMANTIC_CONTROLS.md](../../../docs/protocols/SEMANTIC_CONTROLS.md). Il risultato operativo è un **minimo bidirezionale calibrato da usare come indicatore**, affiancando media e controlli specifici sulle cautele. Gli errori sugli humanizer reali richiedono una successiva valutazione con giudizi umani; questi numeri misurano il protocollo controllato e le sue etichette presunte.

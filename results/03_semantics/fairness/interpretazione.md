# Confronto equo, errori distribuiti e stile localizzato

Il confronto con aggregazione uguale conferma un vantaggio dell'NLI **su queste corruzioni controllate**. Cambiano però due conclusioni operative: il minimo non è l'aggregatore da preferire in ogni scenario e un suo valore grezzo non va confrontato fra testi di lunghezza diversa come se fosse una probabilità di correttezza. La suite ora esporta punteggi continui e distribuzioni; la precedente soglia 0,04075 resta soltanto una diagnostica storica di OpAI.

## 1. Stesse unità, allineamento e aggregatori

Per ogni frase/chunk riscritto si prende la similarità massima con le unità dell'originale, poi si calcolano media, q10 e minimo. Si ripete nella direzione inversa; per il risultato bidirezionale si conserva il supporto minore. Le unità sono **esattamente quelle già usate dall'NLI**: nel campione precedente tutti i conteggi coincidono; nessun chunk condiviso supera il budget di RoBERTa e non sono stati troncati documenti.

Test OpAI appaiato: 141 positivi presunti, 265 alterazioni, 48 fonti. Stesse fonti e stesse coppie del confronto precedente, separate dalla calibrazione.

| Metrica | Media | q10 | Minimo |
| --- | ---: | ---: | ---: |
| NLI | 0,929 | 0,728 | **0,958** |
| Coseno allineato | 0,598 | 0,568 | **0,607** |
| BERTScore allineato | 0,544 | 0,532 | **0,534** |

Sono AUC con orientamento fissato. IC 95% del minimo: NLI [0,935; 0,980], coseno [0,570; 0,649], BERTScore [0,518; 0,552]. Sui soli sottili del campione precedente le AUC del minimo sono rispettivamente 0,900, 0,519 e 0,510. Gli IC percentile appaiati possono avere bias finito, soprattutto vicino a 0,5; non vengono usati per sostenere piccole superiorità delle similarità sul caso.

Coseno e BERTScore cercano il massimo su **tutte** le unità dell'altro testo. L'NLI conserva il massimo fra tre candidati recuperati con MiniLM. Limitare anche BERTScore agli stessi tre candidati produce AUC del minimo circa 0,534, quasi identica: il diverso numero di candidati non spiega il divario osservato. Il rescaling di BERTScore è affine e non cambia le AUC.

Questo risponde al confondente fra media documentale e minimo locale. Non dimostra che le similarità siano universalmente inutili o che l'NLI copra tutte le derive degli humanizer.

## 2. Errori su 1, 2, 3 e 5 frasi

Il primo pannello di 392 corruzioni aveva pochissime fonti indipendenti idonee a cinque frasi. È conservato come esplorativo. Per il confronto principale sono state esaminate altre 800 fonti OpAI già scaricate, escludendo le 99 fonti dei controlli precedenti e scegliendo un editor per fonte prima degli score.

Il pannello ampliato contiene **50 fonti per famiglia**, 25 in calibrazione e 25 nel test, tutte con almeno cinque frasi applicabili. Sono sempre transizioni Style v2→v3. Le due famiglie condividono alcune fonti: 87 positivi aggiuntivi distinti e 400 negativi. Gli edit sono annidati: K=2 mantiene quello di K=1, K=3 mantiene i primi due, K=5 i primi tre. Esattamente K frasi del sentencizer usato dalle metriche cambiano, senza modificare il numero di frasi.

AUC sul test, sempre gli stessi 25 positivi e 25 negativi per famiglia e K:

| Famiglia | Frasi corrotte | Media NLI | q10 NLI | Minimo NLI | Quota non supportata |
| --- | ---: | ---: | ---: | ---: | ---: |
| Negazioni | 1 | 0,906 | 0,688 | 0,922 | 0,889 |
| Negazioni | 2 | **0,989** | 0,824 | 0,965 | 0,994 |
| Negazioni | 3 | **1,000** | 0,920 | 0,978 | 1,000 |
| Negazioni | 5 | **1,000** | 0,998 | 0,997 | 1,000 |
| Sottili | 1 | 0,938 | 0,662 | 0,935 | 0,930 |
| Sottili | 2 | **0,992** | 0,907 | 0,986 | 0,990 |
| Sottili | 3 | 1,000 | 0,987 | 1,000 | 1,000 |
| Sottili | 5 | 1,000 | 1,000 | 1,000 | 1,000 |

Il minimo non domina questo confronto: con più errori la media raggiunge o supera il suo ranking, e q10 recupera quando il difetto non riguarda più una sola frase. Un'AUC empirica di 1,000 su 25 fonti indica separazione completa di questo campione, non perfezione sulla popolazione.

La famiglia sottile rimane soprattutto numerica: a K=5, 227/250 edit sono numeri, 13 quantificatori, 6 cautele, 4 dettagli. È uno **stress test della diffusione**, non una riproduzione della frequenza e della difficoltà degli errori reali. Coseno e BERTScore con minimo rimangono deboli: nelle negazioni passano da 0,533/0,510 a 0,632/0,542 fra K=1 e K=5; nei sottili restano circa 0,50–0,52. Tutte le tabelle e gli IC sono in [distributed_auc.csv](distributed_auc.csv).

## 3. Lunghezza e numero di opportunità

Sui 300 positivi presunti, dopo aver mediato editor e operazioni per le 99 fonti, il minimo NLI è correlato negativamente con il numero massimo di frasi: **Spearman −0,316**, IC [−0,502; −0,102]. Per il coseno il valore è −0,481; per BERTScore −0,206 con IC che include zero. La correlazione fra documenti può dipendere anche da contenuto e difficoltà: non viene presentata come effetto causale.

È stato aggiunto un controllo entro la stessa coppia: 137 positivi, 45 fonti, con almeno 40 unità ipotesi complessive nelle due direzioni. Tenendo fissi evidenze, contenuto e score di tutte le unità, si calcola esattamente il minimo atteso controllando un sottoinsieme uniforme di m unità. Le unità provengono da entrambe le direzioni: non sono m frasi uniche di un documento troncato.

| Metrica | Minimo atteso con 5 unità | Con 40 unità | Differenza 40−5 |
| --- | ---: | ---: | ---: |
| NLI | 0,952 | 0,809 | −0,143 |
| Coseno | 0,911 | 0,802 | −0,109 |
| BERTScore riscalato | 0,863 | 0,740 | −0,123 |

La monotonia è matematica; questo controllo ne quantifica la dimensione sui punteggi osservati a coppia fissa, senza nuove inferenze. La media delle unità ha invece aspettativa costante. [Metodo e IC appaiati](minimum_multiplicity/report.md).

Per il controllo empirico della lunghezza sono stati appresi **soltanto sui positivi di calibrazione** tre intervalli: ≤14, 15–23 e >23 frasi. Il supporto diventa il suo percentile di riferimento entro fascia, con pesi uguali per fonte. La trasformazione continua conserva nel test un'AUC NLI del minimo di **0,951** (IC [0,925; 0,975]); coseno 0,607 e BERTScore 0,538. Il vantaggio di ranking dell'NLI nel confronto principale non scompare, ma i valori grezzi e le soglie rimangono dipendenti dalla distribuzione.

Le fasce sono un riferimento OpAI, non una normalizzazione già validata su DIPPER. Il file di esportazione generico non le applica implicitamente a un nuovo humanizer.

## 4. Guardie lessicali: implementate, con esito non risolutivo

Le guardie conservano conteggi e delta firmati, perdite e guadagni di cautele, negazioni e quantificatori. Per i numeri confrontano **multiset di valori normalizzati**, non soltanto quante cifre compaiono: 20→30 è visibile anche a conteggio invariato; formati come 1,000/1000, 3/three e 20 percent/20% sono normalizzati nel sottoinsieme supportato.

Sono segnali lessicali, non prove di errore. Nel confronto congelato, la regressione L2 sulle quattro guardie ottiene AUC 0,719; NLI più guardie **0,938**, contro 0,958 dell'NLI solo. Sui sottili 0,869 contro 0,900. Non c'è quindi una complementarità operativa dimostrata da questo modello sul test.

Le rimozioni di cautela restano problematiche: la guardia globale di conteggio ha AUC 0,444 sui nove casi, contro 0,571 dell'NLI. L'audit per frase mostra che alcuni cambiamenti si compensano nei totali o sono espressi con parole diverse; il lessico v1 omette anche marker comuni come maybe e can. Non è corretto inferire dal ritorno al conteggio originale che sia stata ripristinata la modalità della stessa proposizione. Due negativi eliminano soltanto possibly da could possibly lasciando could: non sono esempi gold di completa eliminazione della possibilità.

Come diagnostica secondaria, la perdita di hedge segnala 39/39 rimozioni dirette MGTBench e 14/153 positivi presunti OpAI. La prima è copertura della regola applicata, non accuratezza semantica; il confronto usa corpora diversi. Le guardie vengono mantenute come avvisi interpretabili, ma la chiusura del problema richiede espressioni e scope meglio allineati e casi annotati. [Report](guards_full/report.md), [audit delle nove frasi](guards_full/hedge_source_alignment_audit.md).

## 5. Stile sulle frasi toccate

Sono state recuperate le array native `sentences` e `sentence_labels`. Il Parquet non pubblica la colonna sentence_ids, ma la lineage ordinale del costruttore è verificata su tutte le 200 traiettorie: cardinalità costante fra versioni, ricostruzione del testo e ancore non modificate coerenti. Non si tratta di un matching approssimato.

Le etichette sono cumulative: Polish, Paraphrase e Style riscrivono circa il 15%, 25% e 40% delle frasi, comprese quelle già selezionate. Si distinguono frasi marcate e frasi effettivamente cambiate. Sono disponibili tutte le 105 feature sui passaggi toccati; MATTR resta mancante quando il passaggio ha meno di 50 parole.

La diluizione emerge anche confrontando le **stesse** coppie del test, con MATTR definita in entrambi i passaggi: per Paraphrase, ΔMATTR è **+0,02198 sui passaggi cambiati contro +0,00747 sul documento** (92 coppie, 88 fonti). Il contrasto appaiato è +0,01451, IC [0,00881; 0,02067]. Per `pos_cconj`, −0,00308 contro −0,000835 sulle stesse 99 coppie. Non si conclude che la shortlist sia inutile.

La distanza principale usa RMS dei delta standardizzati per frase marcata, con 91 dimensioni fisse e scale apprese dalle sole frasi v0 di calibrazione. Le alternative sono norma sul passaggio concatenato, Jensen–Shannon POS e RMS pesata per la copertura. Sul test (103 traiettorie, 99 fonti):

| Indicatore | Polish | Paraphrase | Style |
| --- | ---: | ---: | ---: |
| RMS per frase marcata | 0,366 | 0,717 | 0,656 |
| RMS sul passaggio | 0,210 | 0,320 | 0,250 |
| Jensen–Shannon POS | 0,056 | 0,100 | 0,072 |
| RMS pesata per copertura | 0,152 | 0,369 | 0,425 |

L'ordine completo non emerge dalla metrica principale: Style−Paraphrase = −0,0615, IC [−0,1257; 0,0157]. L'ordine entro la stessa traiettoria vale per il 33,8% delle fonti. La diagnostica per copertura moltiplica la RMS per sqrt(frasi toccate valide/frasi totali); recupera l'ordine delle medie e lo rispetta nel 61,1% delle fonti. Il budget cresce per costruzione: questo non dimostra maggiore intensità linguistica di Style per frase, né autorizza a scegliere il punteggio soltanto perché restituisce l'ordine desiderato.

I casi letti mostrano parafrasi che cambiano struttura sintattica e Style che cambia pochi sinonimi, output Style identici ai precedenti e frammenti di elenco nelle unità native. La conclusione utile è separare **intensità locale** e **copertura dell'intervento**, non imporre una scala gold a nomi di operazioni con obiettivi diversi. [Report e casi](../../04_opai_style/touched/report.md).

## Uso nella suite

L'output principale è un profilo continuo: NLI media/q10/minimo/quota, similarità con aggregazione uguale, conteggi di guardia e lunghezze. [continuous_suite/](continuous_suite/manifest.json) contiene score per coppia e quantili per sistema, con uguale peso alle fonti. La colonna sistema usa `humanizer` se presente e altrimenti il generatore/editor: qui non costituisce una valutazione di DIPPER o di servizi commerciali.

La precedente soglia 0,04075 non viene applicata dall'esportatore. Qualunque indicatore binario futuro richiede una nuova calibrazione sulla popolazione bersaglio. Questi risultati rafforzano l'uso dell'entailment come complemento alle similarità sui fatti alterati testati, ma ridimensionano l'idea di un unico aggregatore o indice stilometrico universalmente corretto. Rimane necessaria la valutazione degli humanizer reali con annotazioni umane.

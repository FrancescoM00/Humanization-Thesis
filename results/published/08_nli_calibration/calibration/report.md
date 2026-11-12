# Calibrazione NLI completata: scelta della configurazione primaria

Archivio ricevuto e analizzato il **10 novembre 2026**. [Provenienza](provenance.json), [audit](integrity.md), [interpretazione dei limiti](interpretazione.md).

Scelta operativa: **window2 + min**, AUC di calibrazione 0.9735.

Sono incluse 139 coppie di conservazione presunta, 297 alterazioni intenzionali e 137 identità separate, su 46 gruppi di fonte. I positivi OpAI e le perturbazioni sono proxy, non etichette umane gold. Le 141 coppie positive e 265 alterate del test già esaminato non sono usate per selezionare questa configurazione.

La policy viene salvata prima delle nuove inferenze. Si richiedono copertura completa in entrambe le direzioni per tutte le coppie e tutte le identità, media degli score identità almeno 0,95 e decimo percentile almeno 0,90. Sono criteri operativi di coerenza, non soglie universali di validità NLI. Lo score è il minimo delle due direzioni dell'aggregatore; per la quota non supportata è uno meno la quota massima. Si massimizza l'AUC di calibrazione; parità entro 10^-12 preferisce la premessa più corta, poi mean/q10/min/quota. Se nessuno passa, non si sceglie un vincitore.

| Modalità | Aggregatore | AUC calibrazione | Media identità | q10 identità | Primaria ammissibile |
| --- | --- | ---: | ---: | ---: | --- |
| single_all | mean | 0.9143 | 0.9885 | 0.9872 | sì |
| single_all | q10 | 0.7128 | 0.9814 | 0.9779 | sì |
| single_all | min | 0.9470 | 0.9668 | 0.9420 | sì |
| single_all | low_support_fraction | 0.9031 | 1.0000 | 1.0000 | sì |
| window2 | mean | 0.9495 | 0.9897 | 0.9881 | sì |
| window2 | q10 | 0.7190 | 0.9833 | 0.9795 | sì |
| window2 | min | 0.9735 | 0.9710 | 0.9509 | sì |
| window2 | low_support_fraction | 0.9197 | 1.0000 | 1.0000 | sì |
| window3 | mean | 0.9527 | 0.9906 | 0.9890 | sì |
| window3 | q10 | 0.7392 | 0.9846 | 0.9803 | sì |
| window3 | min | 0.9712 | 0.9726 | 0.9597 | sì |
| window3 | low_support_fraction | 0.9221 | 1.0000 | 1.0000 | sì |
| document | mean | mancante | mancante | mancante | no |
| document | q10 | mancante | mancante | mancante | no |
| document | min | mancante | mancante | mancante | no |
| document | low_support_fraction | mancante | mancante | mancante | no |

L'AUC non viene calcolata eliminando le coppie prive di copertura. Gli intervalli nel [CSV](candidate_summary.csv) ricampionano i gruppi di fonte (2000 repliche, seed 42); descrivono l'incertezza di sviluppo e non correggono l'ottimismo della selezione. Le [AUC per alterazione](auc_by_condition.csv) usano soltanto i positivi collegati a quella condizione. Le identità verificano coerenza, non dimostrano accuratezza sulle parafrasi o sulle fusioni di frasi.

Questa è una scelta di sviluppo successiva allo studio sulle 126 fonti, non una preregistrazione retroattiva. Le modalità secondarie restano disponibili; non si trasferiscono automaticamente AUC o soglie ai veri riscrittori. La verifica con le nuove annotazioni umane reference, non ancora ricevute, resta necessaria e non entra nella scelta qui effettuata.

## Esecuzione e portata del risultato

Sono completate tutte le 323.336 evidenze pianificate con DeBERTa-v3 base float32 su GPU, in circa 53,7 minuti di inferenza, esclusi setup e pianificazione. Coppie, allineamenti e policy coincidono con quelli pubblicati prima del run. Le cache delle probabilità e le evidenze estese sono escluse dallo ZIP leggero e restano su Kaggle; l'audit ricalcola aggregazioni, AUC e scelta dagli score documentali ricevuti, non ripete l'inferenza.

Per il minimo, AUC single_all=0,9470, window2=0,9735 e window3=0,9712: la differenza fra due e tre unità è circa 0,0023 e non dimostra superiorità fuori campione. Lo storico 0,958 era invece sul test 141/265: non è un confronto diretto di miglioramento con questo split di calibrazione. Sulle identità il minimo individuale è 0,8666 e tre casi sono sotto 0,90; i criteri fissati riguardavano media e percentile della distribuzione, non un limite per ogni testo.

Il documento ha copertura bidirezionale completa solo per 77/436 coppie etichettate e 36/137 identità: è escluso per copertura, senza attribuirgli un'AUC globale ottenuta eliminando i testi difficili. Fra i 12 candidati ammissibili il massimo puntuale sceglie window2+min; non è un'accuratezza del 97,35% o una validazione umana.

## Applicazione ai dati dello study già calcolati

La configurazione primaria è fissata in [selection.json](selection.json). Per ogni unità ipotesi si prende il massimo entailment fra una o due unità contigue ammissibili; si prende poi il minimo fra le ipotesi e fra forward/reverse. Lo score rimane continuo: non è stata scelta una nuova soglia e non si riusa 0,04075.

Sono state estratte, senza nuove inferenze, le [misure primarie delle 630 righe dello study](../../07_study_126/nli_primary_scores.csv), con 626 score validi e quattro fallimenti mantenuti espliciti. Tutte le 84 reference hanno uno score disponibile. È una rielaborazione descrittiva successiva allo study secondo una scelta effettuata sul pannello di calibrazione, non una modifica retroattiva del protocollo originario.

Il passo successivo è compilare le 84 coppie reference, senza guardare chiavi e score individuali. Le nuove annotazioni non sono ancora ricevute; serviranno a verificare la scelta congelata sulle riscritture reali, senza ritoccarla per massimizzare l'accordo sullo stesso campione. Non occorre un altro run Kaggle per iniziare questa verifica.

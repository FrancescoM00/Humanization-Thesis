# Semantica e stile sul pannello da 126 fonti

Analisi del 10 novembre 2026 sui soli output ricevuti. Non sono state eseguite nuove generazioni, inferenze neurali, chiamate API o annotazioni. Le tabelle complete sono in [per_pair_summary.csv](per_pair_summary.csv), [riepilogo](semantic_style_summary.csv), [confronti a successo comune](semantic_common_success_summary.csv) e [differenze appaiate](semantic_paired_differences.csv).

## Copertura e denominatori

Il file di riscrittura contiene 630 esiti: 126 identità e 126 tentativi per ciascuno dei quattro preset. Quattro esiti sono `truncated`: due della baseline 3B e uno per ciascun preset DIPPER, su tre fonti complessive. Le sei tabelle semantiche, stile e guardie contengono gli stessi **626 ID validi**, compresi i 126 controlli identità. I fallimenti non vengono sostituiti o trasformati in punteggi di conservazione.

| Preset | Tentativi | Coppie con score | Coppie con score nelle 84 fonti nuove |
| --- | ---: | ---: | ---: |
| Baseline Qwen2.5-3B, prompt fissato | 126 | 124 | 82/84 |
| Qwen2.5-14B NF4, stesso prompt | 126 | 126 | 84/84 |
| DIPPER XXL NF4, passaggio intero | 126 | 125 | 83/84 |
| DIPPER XXL NF4, tre frasi e prefisso | 126 | 125 | 83/84 |
| Identità | 126 | 126 | 84/84 |

Le medie per preset sono **condizionate al successo tecnico** e quindi non hanno sempre gli stessi input. Il confronto supplementare sui successi comuni comprende **123 fonti**, di cui **81 nuove**, con tutti e quattro gli output validi. I due DIPPER condividono esattamente le stesse 125 fonti valide; il confronto appaiato fra 3B e 14B usa 124 fonti. I risultati metrici non assorbono la gravità dei tentativi troncati: frequenze di fallimento e qualità dei successi vanno presentate insieme.

Le 42 fonti dei vecchi pilot sono incluse nelle 126; le 84 nuove non erano state usate per le ipotesi nate osservando i pilot. Tutti i fallimenti di questo run sono sulle 84 nuove fonti. Nei CSV `source_subset` separa i due gruppi, senza ricampionare calibrazione o fonti.

## Lunghezza e distanza stilometrica

| Preset | Rapporto parole medio / mediano | Più corti | Più lunghi | RMS medio | Δ MATTR medio | Più vicini al centroide umano |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline 3B | 0,962 / 0,926 | 95/124 | 25/124 | 0,831 | +0,0616 | 60/124 |
| 14B NF4 | 0,871 / 0,858 | 107/126 | 19/126 | 1,047 | +0,0813 | 61/126 |
| DIPPER passaggio | 0,758 / 0,772 | 120/125 | 4/125 | 0,796 | −0,0286 | 29/125 |
| DIPPER tre frasi | 0,920 / 0,927 | 102/125 | 17/125 | 0,651 | −0,0262 | 45/125 |

Le parità di lunghezza sono quattro per la baseline 3B, nessuna per 14B, una per DIPPER passaggio e sei per DIPPER tre frasi. Il rapporto è modificato/originale usando le parole alfabetiche spaCy. Non è il rapporto fra conteggi di token dei modelli.

La configurazione DIPPER a tre frasi limita nettamente la compressione del passaggio intero: sulla stessa coppia di fonti la differenza media del rapporto è **+0,1620**. Produce più parole dell'altro preset in 102 casi, meno in 21 e lo stesso numero in due. Il 14B è invece più corto del 3B in 95 dei 124 confronti comuni, più lungo in 26 e uguale in tre. Non si attribuisce tutto il contrasto DIPPER alla sola finestra: cambiano contesto cumulativo e budget complessivo di generazione per finestra.

L'estrattore produce 105 feature; la RMS usa **91 feature trattenute**, con la formula `sqrt(mean(((phi_modificato - phi_originale) / sd_umani_calibrazione)^2))`. Le scale derivano dagli stessi 300 umani di calibrazione, senza macchine o test. Non è una distanza limitata alle quattro feature della shortlist. MATTR è riportato separatamente e non è fra quelle 91 feature della RMS. La Jensen–Shannon dei POS è anch'essa una misura distinta.

La baseline 3B aumenta MATTR in 118/124 casi e il 14B in 123/126; DIPPER passaggio lo aumenta in 30/125 e il contestuale in 25/125. Il 14B modifica maggiormente le feature, ma ciò non prova che il suo output sia più naturale o più umano. Anche il centroide umano non è un giudice di qualità: la baseline 3B e il 14B si avvicinano in circa metà dei casi, il DIPPER contestuale in 45/125 e quello sul passaggio in 29/125.

Sui **123 successi comuni** i rapporti medi rimangono 0,962 per 3B, 0,856 per 14B, 0,758 per DIPPER passaggio e 0,919 per DIPPER tre frasi; le RMS sono 0,832, 1,048, 0,796 e 0,649. I principali profili descrittivi non dipendono quindi soltanto dalle diverse righe escluse per fallimento. Questo controllo non trasforma le distanze in accuratezza o naturalezza.

## Similarità documentali, allineate e NLI storico

Medie sulle coppie valide di ciascun preset; il BERTScore documentale è grezzo, quello allineato in tabella è riscalato con baseline. Le scale non sono direttamente confrontabili come livelli di accuratezza.

| Preset | Coseno documento | BERTScore documento | NLI medio forward / reverse | Minimo coseno allineato bidirezionale | Minimo BERTScore allineato riscalato | Minimo NLI bidirezionale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline 3B | 0,937 | 0,946 | 0,785 / 0,696 | 0,553 | 0,470 | 0,0988 |
| 14B NF4 | 0,903 | 0,927 | 0,714 / 0,645 | 0,493 | 0,385 | 0,0903 |
| DIPPER passaggio | 0,925 | 0,948 | 0,693 / 0,602 | 0,509 | 0,360 | 0,0152 |
| DIPPER tre frasi | 0,951 | 0,964 | 0,863 / 0,731 | 0,674 | 0,562 | 0,1077 |

Forward indica supporto dell'originale alla riscrittura; reverse supporto della riscrittura all'originale. L'asimmetria può essere compatibile con omissioni, aggiunte o limiti di allineamento, ma non classifica da sola ogni fenomeno. Le similarità documentali rimangono elevate; i minimi allineati sono molto meno prossimi a uno. Per questo non si attribuisce al tipo di metrica un effetto che può dipendere dal livello di aggregazione.

Il DIPPER a tre frasi ha un profilo metrico più favorevole del passaggio intero su queste misure, insieme a minore compressione. Il 14B ha valori medi inferiori alla baseline 3B sulle similarità documentali e sull'NLI storico, pur avendo meno fallimenti tecnici. Sono osservazioni sui preset, non una graduatoria gold di conservazione. Il giudice di naturalezza valuta un'altra proprietà e non può validarle automaticamente.

Il minimo NLI è molto basso per molte riscritture, in particolare DIPPER passaggio. Medie del q10 bidirezionale: 0,229 per 3B, 0,169 per 14B, 0,0895 per DIPPER passaggio e 0,294 per DIPPER tre frasi. Le quote medie di unità con supporto sotto 0,5 sono rispettivamente forward/reverse 0,209/0,304, 0,283/0,341, 0,296/0,390 e 0,127/0,262. La soglia 0,5 qui descrive il supporto del classificatore sulle unità; non è una soglia validata di errore semantico documentale. Nessuno di questi valori è percentuale di significato conservato o perso.

Non si calcola una nuova AUC senza etichette indipendenti, non si trasferisce l'AUC 0,958 delle perturbazioni ai veri humanizer e non si applica la soglia storica OpAI 0,04075. Fusione, coreferenza e lunghezza possono influenzare soprattutto i minimi. La [lettura dei casi](case_review.md) distingue esempi di errore testuale da falsi allarmi plausibili, ma resta una review dell'assistente.

## NLI contestuale e copertura

Sono presenti **5.008 righe contestuali**: 626 coppie × due direzioni × quattro modalità. Il run dichiara DeBERTa-v3 base float32; nessun modello xsmall o quantizzazione sostitutiva. `single_all` considera esaustivamente le singole premesse, mentre il percorso storico usa retrieval `top_k_cosine`: già questo cambia il pool delle prove. `window2` e `window3` comprendono tutte le finestre più corte. `document` usa il testo esatto soltanto quando la coppia entra nel budget di 384 token, senza troncamento silenzioso.

| Preset | Single_all forward / reverse | Fino a due forward / reverse | Fino a tre forward / reverse | Documento forward / reverse |
| --- | ---: | ---: | ---: | ---: |
| Baseline 3B | 0,793 / 0,707 | 0,903 / 0,866 | 0,927 / 0,904 | 0,914 / 0,900* |
| 14B NF4 | 0,727 / 0,657 | 0,911 / 0,833 | 0,937 / 0,880 | 0,930 / 0,896* |
| DIPPER passaggio | 0,703 / 0,606 | 0,876 / 0,703 | 0,913 / 0,735 | 0,873 / 0,708 |
| DIPPER tre frasi | 0,867 / 0,737 | 0,926 / 0,846 | 0,940 / 0,879 | 0,874 / 0,831 |

\* Le medie reverse documentali hanno denominatori inferiori: **122/124** per 3B e **124/126** per 14B. Le altre modalità e tutte le direzioni forward hanno copertura completa. Le quattro righe incomplete restano con aggregato mancante:

- baseline 3B, `Essay:641:a8e05a96382cc9af`, 2/8 ipotesi eleggibili;
- baseline 3B, `Essay:245:198f5b62c8f94655`, 0/11;
- 14B, `Reuters:903:3d3ffb6dca295be8`, 0/11;
- 14B, `Reuters:992:d3932f02b4a9e76d`, 0/15.

Il [CSV a copertura comune](semantic_context_common_coverage.csv) confronta ogni modalità sugli stessi ID per preset e direzione. Per esempio, sui 122 reverse 3B completi in tutte le modalità, single_all/window2/window3/document danno 0,712/0,871/0,907/0,900; sui 124 reverse 14B danno 0,665/0,841/0,887/0,896. Non si inseriscono le coperture parziali in una media come fossero complete.

I minimi sono sensibili alla premessa: il forward medio dei minimi per DIPPER passaggio passa da 0,213 con singole esaustive a 0,696 con finestre fino a tre; il reverse passa da 0,0681 a 0,170. Per DIPPER tre frasi il minimo reverse medio passa da 0,197 a 0,462. La premessa più ampia può risolvere supporti distribuiti su più frasi, ma può anche far salire il supporto di affermazioni alterate.

**L'aumento con le finestre è in parte imposto dalla costruzione.** Per mean, q10 e minimo non c'è nessuna diminuzione oltre il roundoff in tutte le 626 coppie e le due direzioni, passando single_all→window2→window3: i candidati sono annidati e si sceglie un massimo. Il controllo è nel [CSV](semantic_context_nested_checks.csv); non è una misura di accuratezza.

Il documento intero non è una soluzione universale: persino sui **126 controlli identici** assegna entailment medio 0,8958 e minimo medio 0,6920, contro 0,9916 e 0,9817 delle singole esaustive. Le identità hanno invece coseno/BERTScore circa uno e variazioni stilometriche zero. Quindi un valore documentale o locale basso può dipendere dal classificatore e dal formato, mentre un valore alto non certifica che un errore sia assente.

## Guardie lessicali e fonti nuove

Le guardie segnalano cambiamenti negli inventari, senza stabilire se siano errori. Il numero di **documenti con almeno una differenza** è:

| Preset | Cautele | Negazioni | Numeri | Quantificatori |
| --- | ---: | ---: | ---: | ---: |
| Baseline 3B, n=124 | 39 | 32 | 50 | 77 |
| 14B, n=126 | 55 | 50 | 66 | 94 |
| DIPPER passaggio, n=125 | 30 | 19 | 65 | 66 |
| DIPPER tre frasi, n=125 | 23 | 15 | 50 | 57 |

Questi sono conteggi di documenti con valore della guardia >0, non somme delle sostituzioni. Le cautele diminuiscono come numero in 11, 23, 26 e 17 documenti rispettivamente. Riformulazioni lecite, contesti modali e quantificatori possono generare segnalazioni; non si chiamano questi conteggi tassi di errore semantico.

Sulle **84 fonti nuove**, i rapporti medi dei successi sono 0,955 per 3B, 0,870 per 14B, 0,767 per DIPPER passaggio e 0,925 per DIPPER tre frasi. Le RMS sono 0,839/1,059/0,805/0,656. NLI storico forward/reverse: 0,782/0,708, 0,706/0,639, 0,698/0,605 e 0,874/0,741. I profili principali persistono senza i pilot, con denominatori 82/84/83/83: è un controllo sulle nuove fonti, non una validazione delle metriche contro gold umana.

La conservazione sui veri humanizer rimane il punto da annotare: servono giudizi indipendenti, con errori concreti e casi di fusione. L'analisi sostiene la distinzione fra variazione stilistica, compressione, similarità e supporto inferenziale; non permette di sostituire una metrica automatica alla fedeltà osservata.

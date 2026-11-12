# Pilot DIPPER: 42 fonti e confronto appaiato con baseline 3B

In questo report **baseline 3B** indica Qwen2.5-3B-Instruct con il prompt fissato e decoding greedy; **DIPPER** indica la configurazione XXL NF4 sul passaggio intero senza prefisso, lessico 60/ordine 0. I risultati non descrivono ogni configurazione dei due modelli.

**Il pilot DIPPER è completo: 42 riscritture e 42 identità, tutte tecnicamente riuscite.** Rispetto a baseline 3B, DIPPER evade meno spesso OpenAI e più spesso Fakespot raw; sul voto di maggioranza entrambi hanno due successi, su fonti differenti. La compressione è molto più frequente con DIPPER. La lettura dei testi mostra omissioni e cambiamenti puntuali, ma anche limiti dell'NLI quando la riscrittura fonde più frasi.

L'analisi usa esclusivamente gli output ricevuti, senza nuove generazioni o inferenze. [Archivio e schede manuali](https://github.com/FrancescoM00/Humanization/blob/d2868b323938a2a9371c2a872ad86b5596d0f097/exports/kaggle_dipper_pilot/README.md), [dati per coppia](per_pair_summary.csv), [confronto dei due sistemi](comparison_with_qwen.csv), [confronto ASR appaiato](paired_comparison.md), [distribuzioni](metric_distributions.csv), [casi commentati](case_review.md), [integrità](integrity.md), [provenienza](provenance.json).

## Dati e condizioni

Le 42 fonti sono quelle del pilot baseline 3B: due per ciascuna delle 21 combinazioni di tre domini e sette generatori. Originali, 300 umani di calibrazione, umani test, soglie e scale stilometriche coincidono. I tre casi dello smoke sono inclusi e le loro riscritture sono rimaste identiche; non si sommano come nuovi dati indipendenti.

I passaggi iniziali hanno 90–180 parole alfabetiche e terminano a fine frase. Il risultato riguarda questi passaggi e questi generatori storici, non documenti lunghi o tutti gli LLM attuali.

DIPPER XXL è eseguito a 4 bit NF4, con diversità lessicale 60, ordine 0, decoding greedy, un solo tentativo e nessun feedback dei detector. Il protocollo usa l'intero passaggio senza prefisso contestuale: è una variante esplicita rispetto alle finestre contestualizzate del lavoro originale. Qwen2.5-3B è invece la baseline con prompt di humanization già fissato. Il confronto riguarda queste due pipeline concrete; non isola effetti di dimensione del modello, quantizzazione o istruzioni.

Gli input DIPPER hanno 181–276 token registrati, gli output 72–200; tutte le generazioni raggiungono EOS sotto i limiti di 512. Nessuna è identica all'input. L'accorciamento osservato non deriva da un taglio imposto dal limite di generazione. NLI mantiene DeBERTa-v3 base float32 su GPU.

## Evasione: l'ordine cambia con il detector

L'ASR è condizionata agli originali che il detector riconosceva AI. Il denominatore è identico fra i due sistemi.

| Detector | baseline 3B: successi / originali AI | ASR baseline 3B | DIPPER: successi / originali AI | ASR DIPPER |
| --- | ---: | ---: | ---: | ---: |
| OpenAI RoBERTa-base | 17/24 | 70,8% | 4/24 | 16,7% |
| Fakespot raw | 1/37 | 2,7% | 7/37 | 18,9% |
| RADAR | 2/28 | 7,1% | 3/28 | 10,7% |
| Voto di maggioranza | 2/31 | 6,5% | 2/31 | 6,5% |

Lo smoke DIPPER non aveva evasioni sui suoi tre input; il pilot ne mostra su altre fonti. Questo conferma che il collaudo su tre testi non poteva stimare l'efficacia generale. La macro-media dei tre detector è 15,4% per DIPPER contro 26,9% per baseline 3B, ma non è la percentuale di testi che evade l'intero pool.

Il confronto appaiato ricampiona insieme i due esiti della stessa fonte, entro il sottoinsieme degli originali AI. Con 5.000 ricampionamenti e seed 42, la differenza ASR DIPPER−baseline 3B è:

| Detector | Differenza | Intervallo bootstrap appaiato 95% |
| --- | ---: | --- |
| OpenAI | −54,2 punti percentuali | [−75,0; −29,2] |
| Fakespot raw | +16,2 punti percentuali | [+5,4; +27,0] |
| RADAR | +3,6 punti percentuali | [−7,1; +17,9] |
| Maggioranza | 0 punti percentuali | [−12,9; +12,9] |

Sono intervalli esplorativi, con checkpoint e calibrazione fissi, senza correzione per confronti multipli. Non includono l'incertezza delle soglie o del training e non dimostrano trasferibilità a nuovi domini. RADAR e maggioranza non mostrano una differenza ben determinata da questo pannello.

**Le due evasioni della maggioranza sono completamente diverse fra i sistemi.** Nessuna fonte viene elusa da entrambi. Un'ASR aggregata uguale non implica lo stesso comportamento sui singoli testi. Conteggi completi in [paired_asr_comparison.csv](paired_asr_comparison.csv).

## Anche le nuove rilevazioni contano

| Detector | Originali rilevati AI, su 42 | Riscritture DIPPER rilevate AI | AI → umano | Umano → AI |
| --- | ---: | ---: | ---: | ---: |
| OpenAI | 24 | 25 | 4 | 5 |
| Fakespot raw | 37 | 33 | 7 | 3 |
| RADAR | 28 | 33 | 3 | 8 |
| Maggioranza | 31 | 32 | 2 | 3 |

Qui «umano → AI» riguarda originali **macchina** che inizialmente sfuggivano al detector, non testi umani riscritti. Su OpenAI, RADAR e maggioranza il numero complessivo di rilevazioni aumenta dopo DIPPER, pur esistendo alcune evasioni. ASR e TPR complessivo descrivono due aspetti differenti.

Gli umani test sono gli stessi: falsi positivi 3/42 OpenAI, 0/42 Fakespot, 2/42 RADAR e maggioranza. Le soglie sono identiche ai run precedenti; i detector individuali hanno FPR empirico di calibrazione 15/300. Il pool resta interamente della famiglia RoBERTa. Le 14 fonti per dominio e le due per cella non permettono classifiche fini affidabili.

## Conservazione: valori documentali simili, profili locali differenti

Medie sulle stesse 42 fonti, con uguale peso per fonte. BERTScore documentale è l'approssimazione raw con matching di chunk usata dal protocollo. Forward indica originale → riscrittura, reverse riscrittura → originale.

| Metrica | baseline 3B | DIPPER |
| --- | ---: | ---: |
| Coseno documentale | 0,9282 | 0,9215 |
| BERTScore documentale F1 | 0,9445 | 0,9478 |
| ROUGE-L F1 | 0,5481 | 0,6415 |
| BLEU, scala 0–1 | 0,2896 | 0,3495 |
| NLI forward medio | 0,7899 | 0,6840 |
| NLI reverse medio | 0,6708 | 0,5949 |
| Minimo NLI forward | 0,3058 | 0,2349 |
| Minimo NLI reverse | 0,1176 | 0,0638 |
| Coseno allineato, minimo bidirezionale | 0,5505 | 0,4947 |
| BERTScore allineato riscalato, minimo bidirezionale | 0,4642 | 0,3571 |

Il leggero vantaggio di DIPPER nel BERTScore documentale e la maggiore sovrapposizione lessicale non dimostrano una fedeltà superiore. Anche la riduzione dei supporti NLI e delle similarità allineate non dà, senza annotazioni, una stima del tasso di errori o una classifica certa della conservazione. Le metriche hanno scale e sensibilità differenti.

La compressione coesiste con omissioni effettive; il supporto reverse è utile per individuarne alcune. Tuttavia la riscrittura può anche fondere frasi, rendendo insufficiente ciascuna premessa originale isolata. Non viene applicata la soglia storica 0,04075 e non viene calcolata AUC sui nuovi output in assenza di etichette indipendenti.

## Cosa mostrano i testi

La [review dell'assistente](case_review.md) comprende 12 casi: sei selezionati tramite hash delle fonti senza usare gli score e sei approfondimenti mirati. Non è una review umana cieca e non permette di contare quanti dei 42 testi siano semanticamente errati.

**Errori puntuali insieme all'evasione.** In Reuters/Dolly, DIPPER sostituisce «€80 million (£72.4 million)» con «80 million francs», omettendo anche altri dettagli; Fakespot viene eluso. In Essay/Claude, «up to 100 feet high» diventa «a height of thirty feet», con perdita del valore e del vincolo riportati; anche qui Fakespot viene eluso. Il forte segnale NLI è coerente con variazioni testuali verificabili, ma il suo score di contraddizione non è una prova logica indipendente. Questi esempi non significano che ogni evasione comporti errori.

**Un riassunto sostenuto può perdere molto.** In Essay/ChatGPT un passaggio passa da 152 a 59 parole, perde esempi e dettagli, ma ha forward 0,986 e reverse 0,438. Il forward alto è compatibile con una sintesi sostenuta dalla fonte, non con la conservazione di tutto il contenuto.

**Un forward quasi nullo può penalizzare la fusione.** Nel racconto Maria/Sofia, 15 frasi originali diventano cinque. Gran parte della trama rimane riconoscibile, pur con differenze di nome e intensità; ciascuna nuova frase combina più informazioni. Il forward medio è solo **0,00262**, mentre il reverse è 0,76640. Non si tratta di cinque invenzioni dimostrate: le singole premesse originali non sostengono da sole le ipotesi fuse. Il caso rende particolarmente urgente misurare i falsi allarmi dell'NLI sulle riscritture reali prima di interpretare le medie come qualità dei sistemi.

**Gli input rumorosi producono comportamenti diversi.** Le due consegne StableLM già identificate per baseline 3B restano consegne con DIPPER, ma perdono requisiti; baseline 3B invece le svolgeva. In Reuters/StableLM parte della richiesta compare cinque volte, di cui quattro consecutive alla fine: ridurre la ripetizione abbassa le parole senza una perdita proporzionale di informazione unica. Non tutto l'accorciamento equivale quindi a distruzione di contenuto.

**Esistono anche casi conservativi.** Essay/GPT4All passa da 161 a 158 parole e mantiene gran parte del contenuto, con reverse 0,986. Conserva però l'autoriferimento «As an AI language model». Fedeltà, rimozione di formule e naturalezza sono obiettivi distinti.

## Stile e lunghezza: DIPPER comprime sistematicamente

| Indicatore | baseline 3B | DIPPER |
| --- | ---: | ---: |
| Rapporto parole medio | 0,9752 | 0,7393 |
| Rapporto parole mediano | 0,9150 | 0,7664 |
| Testi più corti / più lunghi / stesso conteggio | 34 / 7 / 1 | 41 / 0 / 1 |
| Output sotto l'80% delle parole originali | 6/42 | 23/42 |
| RMS stilometrico medio | 0,8164 | 0,7765 |
| RMS stilometrico mediano | 0,7039 | 0,7451 |
| POS-JS medio | 0,1109 | 0,1216 |
| Δ MATTR50 medio | +0,0628 | −0,0199 |
| MATTR50 in aumento | 37/42 | 13/42 |
| Distanza dal centroide umano in diminuzione | 20/42 | 13/42 |
| Δ Flesch Reading Ease medio | −4,91 | +4,39 |

Il rapporto mediano DIPPER corrisponde a circa **−23,4% di parole**. In 33/42 fonti DIPPER produce meno parole di Qwen. Le lunghezze dei suoi output vanno da 59 a 158 parole, tutte senza interruzione al limite di generazione. Il profilo osservato comprende quindi una forte componente di sintesi, importante per interpretare evasione, stile e conservazione insieme.

La media RMS DIPPER è inferiore a baseline 3B, ma la mediana è superiore: non emerge un ordinamento uniforme della quantità di cambiamento. MATTR50 diminuisce in 29/42 casi DIPPER. Queste differenze non sono voti di qualità. Anche l'aumento medio di Flesch misura quella formula di leggibilità, non naturalezza; la sua variazione mediana è soltanto +1,52. Le scale sono comuni e stimate sui 300 umani di calibrazione.

## Sensibilità alle due consegne già note

Il risultato primario mantiene tutte le 42 fonti. Come analisi secondaria si escludono le stesse due fonti StableLM già segnalate nel report baseline 3B, senza cercare ulteriori esclusioni sulla base degli score DIPPER. Il sottoinsieme da 40 rimane sbilanciato e non è un nuovo dataset clean certificato.

| Misura DIPPER | Tutte le 42 fonti | Sottoinsieme di 40 |
| --- | ---: | ---: |
| ASR OpenAI | 4/24 | 3/23 |
| ASR Fakespot | 7/37 | 6/35 |
| ASR RADAR | 3/28 | 3/26 |
| ASR maggioranza | 2/31 | 1/29 |
| Rapporto parole medio | 0,7393 | 0,7546 |
| NLI forward medio | 0,6840 | 0,6796 |
| NLI reverse medio | 0,5949 | 0,5978 |

La compressione e il profilo medio NLI non dipendono soltanto da quelle due consegne. L'evasione di maggioranza perde invece uno dei soli due successi: il piccolo numeratore è sensibile a un caso. Il confronto completo dei due sistemi è in [sensitivity_summary.csv](sensitivity_summary.csv).

## Integrità, tempi e prossimo passo

La configurazione e i dati sono identici a quelli Qwen. I 1.278 score detector condivisi coincidono esattamente; riepiloghi e soglie sono stati ricalcolati dai dati salvati. Gli score identità sono coerenti, con soltanto arrotondamento BERTScore documentale fino a 7,17×10⁻⁹. Tutti gli stadi sono completi. Le 42 generazioni DIPPER sommano **481,33 secondi** registrati; i sei stadi semantici/allineati **63,99 secondi**, esclusi download, caricamento e altri blocchi del notebook.

Il confronto riguarda **42 fonti**, non 84 osservazioni indipendenti perché sono presenti due sistemi. Si conferma l'utilità di estendere il campione comune; le 126 fonti sono già riservate, mentre la proposta da 420 richiede una nuova coorte con i 300 gruppi di calibrazione esplicitamente fissi. Nessuna estensione è stata ancora eseguita e i preset non sono modificati dopo la lettura degli output.

Sono pronte [schede comuni](https://github.com/FrancescoM00/Humanization/blob/d2868b323938a2a9371c2a872ad86b5596d0f097/exports/kaggle_dipper_pilot/README.md): **45 coppie semantiche**, con baseline 3B e DIPPER sulle stesse 21 fonti più tre identità, e **24 testi per naturalezza** su sei fonti comuni, con umani, originali macchina e riscritture dei due sistemi. Le selezioni usano identificatori, non score; chiavi separate e giudizi vuoti. Le 24 coppie del precedente modulo baseline 3B sono incluse nella nuova scheda semantica e non vanno conteggiate come nuovi giudizi indipendenti se già annotate.

L'estensione migliorerà la precisione dell'evasione; le annotazioni serviranno a distinguere errori reali e falsi allarmi delle metriche, e a completare l'asse naturalezza. Fino ad allora i risultati mostrano profili differenti dei due preset, senza un vincitore generale sui quattro assi.

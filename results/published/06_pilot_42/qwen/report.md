# Pilot Kaggle: 42 riscritture baseline 3B e tre detector

In questo report **baseline 3B** indica Qwen2.5-3B-Instruct con il prompt fissato e decoding greedy; **DIPPER** indica la configurazione XXL NF4 sul passaggio intero senza prefisso, lessico 60/ordine 0. I risultati non descrivono ogni configurazione dei due modelli.

Il punto da verificare nell’estensione è la variazione complessiva di rilevabilità: Fakespot passa da 37 a 41 testi rilevati e RADAR da 28 a 35. L’ASR elevata sul detector storico OpenAI resta un risultato specifico di quel checkpoint. I controlli di lunghezza e il detector zero-shot sono definiti nel [protocollo v2](../../../docs/protocols/STUDY_V2_PROTOCOL.md), senza modificare questi score.

Il pilot è tecnicamente completo: **42 fonti macchina, 42 riscritture baseline 3B e 42 identità**, senza generazioni fallite. L'evasione si concentra sul detector storico OpenAI. Su Fakespot, RADAR e voto di maggioranza rimane bassa; dopo la riscrittura questi ultimi riconoscono anche diversi originali che prima sfuggivano. La conservazione del contenuto richiede una lettura più articolata delle sole medie: sono presenti cambiamenti reali, input anomali e limiti delle metriche.

Archivio originale e schede da annotare, [dati per coppia](per_pair_summary.csv), [riepilogo detector](detection_summary.csv), [transizioni](detection_transitions.csv), [distribuzioni](metric_distributions.csv), [casi commentati](case_review.md), [integrità](integrity.md), [provenienza](provenance.json).

## Disegno e stato dell'esperimento

Il campione comprende due fonti per ciascuna combinazione di tre domini e sette generatori: 14 fonti per dominio, sei per generatore. I 300 umani di calibrazione sono separati dai gruppi test e dalle 126 fonti riservate allo studio successivo. La coorte è quella congelata prima degli score; include i tre casi del precedente smoke, le cui riscritture sono rimaste identiche.

Il sistema è Qwen2.5-3B-Instruct con prompt fisso di humanization, una sola generazione greedy e nessun feedback dai detector. È una baseline di rewriting via prompt, non un servizio commerciale né un modello addestrato specificamente per evasione. I passaggi iniziali sono quelli storici a fine frase, di 90–180 parole; non sono documenti completi.

L'analisi riusa i file ricevuti. Non sono state eseguite nuove generazioni, predizioni di detector o inferenze delle metriche neurali. La revisione qualitativa è dell'assistente; le annotazioni umane e la naturalezza non sono ancora disponibili.

## Evasione: risultato molto diverso fra detector

ASR condizionata agli originali riconosciuti AI. Fra parentesi, l'intervallo bootstrap per fonte già prodotto dalla pipeline; soglie e modelli sono fissi nel ricampionamento.

| Detector | Originali riconosciuti AI | Successi baseline 3B | ASR | IC 95% ASR | Riscritture riconosciute AI |
| --- | ---: | ---: | ---: | --- | ---: |
| OpenAI RoBERTa-base | 24/42 | 17/24 | 70,8% | 52,0–87,5% | 8/42 |
| Fakespot raw | 37/42 | 1/37 | 2,7% | 0–10,8% | 41/42 |
| RADAR | 28/42 | 2/28 | 7,1% | 0–17,9% | 35/42 |
| Voto di maggioranza | 31/42 | 2/31 | 6,5% | 0–16,1% | 35/42 |

La macro-media ASR dei tre detector è **26,9%**. È una media di tre tassi con denominatori diversi, non la percentuale di testi che evade l'intero pool. Il voto di maggioranza costituisce una regola distinta.

Il 70,8% di OpenAI non significa che il sistema abbia ingannato tutti i detector sul 70,8% dei 42 testi. Quel detector riconosceva solo 24 originali; 17 di questi passano sotto la soglia. Le transizioni complete mostrano anche il movimento opposto:

| Detector | AI → umano | Umano → AI | TPR prima → dopo |
| --- | ---: | ---: | ---: |
| OpenAI | 17 | 1 | 57,1% → 19,0% |
| Fakespot raw | 1 | 5 | 88,1% → 97,6% |
| RADAR | 2 | 9 | 66,7% → 83,3% |
| Maggioranza | 2 | 6 | 73,8% → 83,3% |

In questa tabella anche gli elementi classificati inizialmente umani sono **testi macchina**: si tratta di falsi negativi dei detector, non di testi umani riscritti. Fakespot, RADAR e maggioranza segnalano dunque più testi dopo Qwen. La causa di questo comportamento non è stata identificata sperimentalmente.

Gli umani test restano inalterati: FPR 3/42 (7,1%) per OpenAI, 0/42 per Fakespot, 2/42 (4,8%) per RADAR e maggioranza. Sono controlli di riferimento, non una misura dell'effetto di riscrivere gli umani. Le soglie coincidono esattamente con quelle dello smoke e ciascun detector ha FPR di calibrazione 15/300; ciò non garantisce lo stesso tasso su un nuovo campione.

Per OpenAI l'ASR è 7/7 in Essay, 6/8 in Reuters e 4/9 in WritingPrompts. Questi denominatori sono piccoli: l'ordine fra domini rimane esplorativo. I [risultati per generatore](detection_by_generator.csv) conservano sempre il denominatore; sei fonti per generatore non permettono classifiche affidabili. Il pool iniziale resta composto da modelli della famiglia RoBERTa, e Fakespot è la variante a input grezzo documentata nel protocollo.

## Conservazione: distribuzioni, non verdetti

Medie sulle 42 riscritture, con uguale peso per fonte. BERTScore documentale è l'approssimazione con matching di chunk, non il confronto integrale senza limiti di contesto.

| Metrica | Media | Mediana |
| --- | ---: | ---: |
| ROUGE-L F1 documentale | 0,5481 | 0,5405 |
| BLEU documentale, scala 0–1 | 0,2896 | 0,2838 |
| Coseno documentale | 0,9282 | 0,9444 |
| BERTScore documentale F1, grezzo | 0,9445 | 0,9500 |
| NLI originale → riscrittura, medio | 0,7899 | 0,8414 |
| NLI riscrittura → originale, medio | 0,6708 | 0,7119 |
| Minimo NLI forward | 0,3058 | 0,0198 |
| Minimo NLI reverse | 0,1176 | 0,0075 |
| Coseno allineato, minimo bidirezionale | 0,5505 | 0,5931 |
| BERTScore allineato riscalato, minimo bidirezionale | 0,4642 | 0,5209 |

Le similarità documentali sono elevate, mentre le misure allineate mostrano escursioni più ampie. Il supporto reverse medio inferiore al forward è compatibile con omissioni e accorciamenti, ma risente anche di fusione delle frasi, coreferenza e insufficienza della premessa singola. **0,6708 non è una stima della percentuale di significato conservato.**

I minimi molto bassi non diventano automaticamente errori dello humanizer. Il caso Reuters dello smoke rimane un esempio: nome dell'azienda e crisi economica sono conservati in frasi diverse, mentre l'NLI valuta ogni premessa separatamente. Il classificatore può giudicare correttamente insufficiente la frase isolata; è il suo impiego come minimo documentale a produrre un allarme eccessivo.

Non sono state calcolate AUC di conservazione sul pilot: mancano etichette umane indipendenti. Non si applica la vecchia soglia OpAI 0,04075 e non si presenta il numero di score bassi come tasso di testi semanticamente errati.

## Un limite concreto dell'NLI sui vincoli quantitativi

Nel caso Essay/Claude `humanizer_6927e6c090a0effb9fead649`, la frase originale dice che gli tsunami possono raggiungere **“up to 100 feet”**, mentre la riscrittura dice **“tower over 100 feet”**. Il valore 100 è conservato, ma cambia il vincolo da “fino a” a “oltre”. NLI assegna medie forward/reverse 0,9832/0,9859 e minimi 0,9406/0,9501: l'alto supporto non segnala questa variazione.

Le guardie segnalano altre differenze nella coppia (numero e quantificatore cambiano di una unità), ma il loro inventario non rappresenta direttamente la relazione `up to`/`over`; non si può attribuire loro il riconoscimento specifico di quel vincolo. Un'eventuale guardia relazionale aggiuntiva va introdotta e validata come versione successiva, conservando gli score attuali.

Nel caso Reuters/Dolly `humanizer_8f156ecd5373a52262810d37`, **“more than 200,000 trades”** diventa **“nearly 200,000 trades”** e alcuni riferimenti temporali scompaiono. Qui il reverse medio è 0,1958, ma il calo non permette da solo di identificare quale cambiamento sia stato rilevato. La [review](case_review.md) distingue osservazioni testuali da interpretazione dei punteggi.

## Due input sono consegne: baseline 3B cambia compito

Due originali StableLM sono istruzioni di scrittura, non normale prosa espositiva. In Essay si richiede di discutere il marketing verso consumatori vulnerabili; in Reuters si richiede un elaborato sul mercato cinese e sulla soddisfazione dei clienti bancari. baseline 3B sviluppa le risposte invece di parafrasare le consegne, nonostante il prompt di sistema chiedesse di trattare il passaggio come materiale da riscrivere.

Questi sono i due casi con maggior espansione: **133→381 parole** e **164→274 parole**. Le medie NLI forward/reverse sono rispettivamente 0,0873/0,0320 e 0,0778/0,0654. Nel secondo output compaiono anche riferimenti bibliografici assenti dall'input e non verificati; non si afferma qui che siano falsi senza un controllo esterno.

I casi restano nel risultato principale dei 42 input congelati. La pulizia storica normalizzava spazi, terminazioni e lunghezza; non certificava che ogni output fosse prosa. La fedeltà all'input e la verità esterna sono aspetti distinti, e la qualità dell'input condiziona l'interpretazione dell'esperimento.

Come controllo esplorativo successivo alla lettura, escludendo solo questi due casi le medie diventano: coseno 0,9366, BERTScore 0,9476, NLI forward 0,8253 e reverse 0,7019. L'ASR resta concentrata su OpenAI (16/23), con 1/35 Fakespot, 2/26 RADAR e 2/29 maggioranza. La conclusione sui detector quindi non dipende soltanto dai due input anomali. Questo sottoinsieme di 40 è sbilanciato e non sostituisce il campione principale né costituisce un nuovo dataset clean certificato.

## Stile: cambiamento evidente, direzione non universalmente umana

RMS stilometrico medio 0,8164, mediano 0,7039; POS-JS medio 0,1109. Le scale sono stimate sui 300 umani di calibrazione, senza usare i testi macchina o test. MATTR50 aumenta in 37/42 casi, con delta medio +0,0628; le medie dei delta sono −0,0026 per `pos_num`, −0,0016 per `pos_cconj` e −0,0043 per `pos_propn`.

La distanza dal centroide umano diminuisce in 20 casi e aumenta in 22; la variazione mediana è +0,0086. Non emerge quindi un avvicinamento uniforme a quel riferimento. Il centroide è inoltre una descrizione pooled del campione umano, non una misura di naturalezza o una probabilità di origine umana.

Flesch Reading Ease medio passa da 47,82 a 42,91 (delta −4,91); Flesch–Kincaid Grade da 11,32 a 11,58. In media la riscrittura non migliora quindi quel particolare indicatore di leggibilità. Anche questo non determina il giudizio di naturalezza, che richiede la valutazione separata prevista.

La lunghezza mediana è **0,915 volte l'originale**, cioè circa −8,5%. Si accorciano 34 riscritture, sette si allungano e una mantiene il conteggio. La media dei rapporti è 0,975 perché le due grandi espansioni la spostano verso uno. Sei output sono sotto l'80% e due sopra il 120% della lunghezza iniziale. Escludendo i due casi-consegna, il rapporto medio è 0,9105. I conteggi usano i token alfabetici spaCy, come l'esperimento storico. Nessuna riscrittura è stata troncata artificialmente per correggere queste differenze.

## Controlli completati e cosa manca

Tutti gli 84 output hanno stato `ok`; è uno stato di completamento tecnico, non un giudizio semantico. Le identità hanno distanze e guardie zero, rapporto parole uno e coseno/BERTScore circa uno. NLI identità medio 0,9910, senza impostarlo artificialmente a uno. I riepiloghi dei detector sono stati ricalcolati dai dati registrati e coincidono; gli hash delle coppie e dei sorgenti corretti sono coerenti. Il [dettaglio](integrity.md) documenta le verifiche senza nuove inferenze.

Le 42 generazioni baseline 3B sommano **339,8 secondi** registrati, esclusi caricamento e download. I sei stadi semantici/allineati sommano **62,3 secondi**, esclusi download e altri blocchi. Non sono tempi dell'intero notebook.

La naturalezza resta da valutare. I moduli originali contengono 12 coppie semantiche e 28 testi singoli, tutti senza annotazioni. La review esplorativa dell'assistente copre 12 casi: sei selezionati tramite hash senza usare gli score e sei scelti per approfondire outlier o evasioni rare. Non è una review umana cieca e non produce un'accuratezza delle metriche.

## Prossimo passo concreto

È pronta una scheda semantica da 24 coppie: una riscrittura per ciascuna delle 21 celle dominio/generatore, più tre identità di controllo. La selezione dipende solo dagli ID e la scheda non contiene score o nomi dei sistemi. La chiave è separata. Occorre distinguere tipo di input, aggiunte, omissioni, numeri, vincoli quantitativi e cambi di forza delle affermazioni.

Compilare la scheda e i valutazione fuori dal perimetro della consegna fornirà un primo riferimento umano. Un singolo valutatore resta un controllo esplorativo; eventuali giudizi indipendenti vanno conservati separatamente. Non usare NLI o il verdetto dei detector per decidere l'etichetta.

Il confronto successivo utile è un secondo sistema, per esempio **DIPPER sugli stessi input**, iniziando dal suo smoke di tre fonti prima dei 42. L'espansione a 126 baseline 3B da sola aumenterebbe il campione ma non risolverebbe la dipendenza dal singolo humanizer o la mancanza di annotazioni. La naturalezza tramite giudice LLM grande rimane un esperimento distinto.

# Rilevabilità nello study: 126 fonti e controllo sulle 84 non viste

Analisi dei soli score ricevuti il 10 novembre 2026, senza nuovi modelli, generazioni o richieste API. I quattro riscrittori sono i preset fissati di baseline Qwen2.5-3B, Qwen2.5-14B NF4 con lo stesso prompt, DIPPER XXL NF4 sul passaggio intero e DIPPER XXL NF4 a tre frasi con prefisso cumulativo. I nomi brevi sotto non descrivono tutte le configurazioni delle rispettive famiglie.

**L'evasione dipende dal detector anche aggiungendo Binoculars.** Il 14B evade più della baseline 3B su OpenAI e Binoculars, ma entrambi risultano quasi sempre rilevati da Fakespot. L'aumento di rilevazione Fakespot si conferma sulle 84 fonti non viste; l'aumento RADAR osservato nel pilot è invece meno preciso in questo sottoinsieme. I controlli di lunghezza non riproducono il vantaggio dei due Qwen su OpenAI/Binoculars, mentre per DIPPER su OpenAI non emerge un vantaggio aggregato rispetto al prefisso nullo.

## Campione, fallimenti e denominatori

Il pannello comprende 126 fonti e 126 gruppi distinti: tre domini × sette generatori × sei fonti per cella. Le 42 fonti già analizzate nei pilot sono incluse; le **84 fonti nuove**, quattro per cella, vengono quindi riportate separatamente. Non sono una replica su un corpus esterno. I 300 umani di calibrazione e le soglie restano fissati e disgiunti dai test.

Esiti disponibili: 124/126 per baseline 3B, 126/126 per 14B e 125/126 per ciascun preset DIPPER. I quattro esiti non validi hanno stato `truncated`. Una generazione fallita **resta nel denominatore ASR se l'originale era rilevato**, senza diventare evasione. Per TPR e transizioni prima/dopo si usano invece le sole coppie con output valido: non si imputa un detector score ai fallimenti.

La tabella ASR mostra successi/originali già rilevati, non percentuali sull'intera coorte. Il voto di maggioranza riguarda soltanto i tre classificatori RoBERTa correlati; Binoculars è separato. Il suo ratio grezzo ha direzione AI bassa, senza conversione in probabilità.

## ASR sulle 126 fonti

| Detector | Baseline 3B | 14B NF4 | DIPPER passaggio | DIPPER tre frasi |
| --- | ---: | ---: | ---: | ---: |
| OpenAI RoBERTa | 45/73 = 61,6% | 60/73 = 82,2% | 10/73 = 13,7% | 6/73 = 8,2% |
| Fakespot | 1/109 = 0,9% | 1/109 = 0,9% | 11/109 = 10,1% | 8/109 = 7,3% |
| RADAR | 8/90 = 8,9% | 8/90 = 8,9% | 7/90 = 7,8% | 11/90 = 12,2% |
| Binoculars | 17/120 = 14,2% | 29/120 = 24,2% | 0/120 = 0% | 1/120 = 0,8% |
| Maggioranza RoBERTa | 4/94 = 4,3% | 6/94 = 6,4% | 7/94 = 7,4% | 6/94 = 6,4% |

Gli intervalli bootstrap 95% del protocollo includono: OpenAI 3B [50,7%; 72,6%], 14B [73,9%; 90,4%]; Binoculars 3B [8,3%; 20,8%], 14B [16,7%; 31,7%]. Il successo su OpenAI non si trasferisce a Fakespot/RADAR, e l'assenza di evasioni Binoculars nello smoke da tre fonti non si mantiene nello study.

I falsi positivi sui 126 umani test sono OpenAI 4/126, Fakespot 4/126, RADAR 3/126, Binoculars 6/126 e maggioranza 2/126. Sono risultati di test, distinti dal 5% empirico imposto sui 300 umani di calibrazione. Gli originali macchina sono rilevati rispettivamente in 73, 109, 90, 120 e 94 casi: i denominatori ASR differenti riflettono questa diversa copertura di partenza.

## Verifica sulle 84 fonti nuove

| Detector | Baseline 3B | 14B NF4 | DIPPER passaggio | DIPPER tre frasi |
| --- | ---: | ---: | ---: | ---: |
| OpenAI RoBERTa | 28/49 = 57,1% | 41/49 = 83,7% | 6/49 = 12,2% | 4/49 = 8,2% |
| Fakespot | 0/72 | 0/72 | 4/72 = 5,6% | 4/72 = 5,6% |
| RADAR | 6/62 = 9,7% | 6/62 = 9,7% | 4/62 = 6,5% | 8/62 = 12,9% |
| Binoculars | 9/80 = 11,3% | 21/80 = 26,3% | 0/80 | 1/80 = 1,3% |
| Maggioranza RoBERTa | 2/63 = 3,2% | 3/63 = 4,8% | 5/63 = 7,9% | 5/63 = 7,9% |

Questi esiti confermano il profilo per detector senza affidarsi soltanto ai 42 casi osservati in precedenza. Non autorizzano una conclusione su humanizer commerciali, generatori contemporanei assenti dal corpus o altri prompt di Qwen.

## Riscrivere può aumentare la rilevazione: distinguere i detector

Le transizioni seguenti riguardano le medesime fonti con generazione riuscita. Per esempio, `107→123 su 124` è il confronto appaiato Fakespot prima/dopo, non un confronto 109/126 contro 123/124 con denominatori diversi.

| Preset / detector | 126: rilevati prima → dopo / coppie valide | Variazione TPR, pp [IC95%] | 84 nuove: prima → dopo / coppie valide | Variazione TPR, pp [IC95%] |
| --- | ---: | ---: | ---: | ---: |
| 3B / Fakespot | 107→123 / 124 | +12,9 [7,2; 19,4] | 70→82 / 82 | +14,6 [7,3; 23,2] |
| 14B / Fakespot | 109→125 / 126 | +12,7 [7,1; 19,0] | 72→84 / 84 | +14,3 [7,1; 22,6] |
| 3B / RADAR | 88→99 / 124 | +8,9 [0,8; 16,9] | 60→64 / 82 | +4,9 [−4,9; 14,6] |
| 14B / RADAR | 90→100 / 126 | +7,9 [0,0; 15,9] | 62→66 / 84 | +4,8 [−4,8; 14,3] |
| 3B / Binoculars | 118→102 / 124 | −12,9 [−19,4; −6,5] | 78→69 / 82 | −11,0 [−18,3; −4,9] |
| 14B / Binoculars | 120→92 / 126 | −22,2 [−30,2; −15,1] | 80→59 / 84 | −25,0 [−34,5; −16,7] |

Fakespot ha 17 nuove rilevazioni e un'evasione per ciascun Qwen sulle 126; sulle 84 ne ha dodici nuove e zero evasioni. La stessa riscrittura può quindi ridurre la rilevabilità Binoculars e aumentarla Fakespot. La frase generale «riscrivere con un LLM rende più rilevabile» va circoscritta a detector e preset. L'aumento Fakespot non è soltanto un fenomeno StableLM, anche se metà delle dodici nuove rilevazioni sulle 84 fonti appartiene a quel generatore: le altre sono tre Claude, due Dolly e una ChatGPT. Le celle restano piccole e non sono etichette del tipo di input.

## Confronti appaiati fra i preset

Differenze ASR, in punti percentuali, sulla stessa selezione di originali già rilevati. I fallimenti rimangono insuccessi come nel protocollo. Si tratta di confronti esplorativi dei preset fissati, senza correzione per confronti multipli; non si attribuisce causalmente l'effetto alla sola dimensione o alla sola finestra.

| Differenza | Detector | 126 fonti, pp [IC95%] | 84 nuove, pp [IC95%] |
| --- | --- | ---: | ---: |
| 14B − baseline 3B | OpenAI | +20,5 [8,2; 32,9] | +26,5 [12,2; 40,8] |
| 14B − baseline 3B | Binoculars | +10,0 [3,3; 17,5] | +15,0 [6,3; 23,8] |
| 14B − baseline 3B | Fakespot | 0,0 [0,0; 0,0] | 0,0 [0,0; 0,0] |
| 14B − baseline 3B | RADAR | 0,0 [−5,6; 5,6] | 0,0 [−6,5; 6,5] |
| DIPPER tre frasi − passaggio | OpenAI | −5,5 [−13,7; 1,4] | −4,1 [−12,2; 4,1] |
| DIPPER tre frasi − passaggio | RADAR | +4,4 [−2,2; 11,1] | +6,5 [−1,6; 16,1] |
| DIPPER tre frasi − passaggio | Binoculars | +0,8 [0,0; 2,5] | +1,3 [0,0; 3,8] |

Il 14B mostra un vantaggio sul 3B limitato ad alcuni detector; non è una misura di fedeltà o naturalezza. Per DIPPER a tre frasi non emerge un vantaggio generale di evasione. I CSV riportano anche il confronto 3B/DIPPER contestuale e le differenze di TPR sulle fonti in cui entrambi i sistemi riescono.

## Controllo di lunghezza: confronti sullo stesso sottoinsieme eleggibile

Il prefisso nullo usa lo stesso conteggio alfabetico spaCy della riscrittura, anche a costo di un taglio a metà frase. Le riscritture più lunghe dell'originale e i riferimenti falliti sono ineleggibili; non si imputa una lunghezza. I sottoinsiemi sono quindi 99 fonti per 3B, 107 per 14B, 121 per DIPPER passaggio e 108 per DIPPER tre frasi sulle 126; sulle 84 sono rispettivamente 64, 71, 79 e 70. Dentro ognuno l'ASR usa soltanto gli originali già rilevati dal detector.

| Pannello / detector | Preset | Evasioni prefisso nullo | Evasioni riscrittore | Differenza riscrittore − nullo, pp [IC95%] |
| --- | --- | ---: | ---: | ---: |
| 126 / OpenAI | 3B | 1/62 | 39/62 | +61,3 [48,4; 72,6] |
| 126 / OpenAI | 14B | 7/64 | 54/64 | +73,4 [62,5; 84,4] |
| 126 / Binoculars | 3B | 0/96 | 13/96 | +13,5 [7,3; 20,8] |
| 126 / Binoculars | 14B | 1/102 | 27/102 | +25,5 [16,7; 34,3] |
| 84 nuove / OpenAI | 3B | 1/41 | 24/41 | +56,1 [41,5; 70,7] |
| 84 nuove / OpenAI | 14B | 4/42 | 36/42 | +76,2 [61,9; 88,1] |
| 84 nuove / Binoculars | 3B | 0/62 | 5/62 | +8,1 [1,6; 14,5] |
| 84 nuove / Binoculars | 14B | 0/68 | 19/68 | +27,9 [17,6; 38,2] |
| 126 / OpenAI | DIPPER passaggio | 10/72 | 10/72 | 0,0 [−9,7; 9,8] |
| 126 / OpenAI | DIPPER tre frasi | 5/64 | 5/64 | 0,0 [−9,4; 9,4] |
| 84 nuove / OpenAI | DIPPER passaggio | 7/48 | 6/48 | −2,1 [−14,6; 10,4] |
| 84 nuove / OpenAI | DIPPER tre frasi | 3/42 | 3/42 | 0,0 [−11,9; 11,9] |

Per Qwen l'accorciamento da solo non riproduce le evasioni sui due detector; il risultato persiste sulle fonti nuove. Non è però una stima causale del contributo stilistico, perché un prefisso elimina contenuto mentre una riscrittura può conservarlo o modificarlo in altro modo.

Per DIPPER/OpenAI non emerge un vantaggio aggregato sul prefisso. **ASR uguale non significa le stesse evasioni:** sulle 126, passaggio e controllo hanno otto successi esclusivi ciascuno, mentre contestuale e controllo hanno cinque successi ciascuno tutti disgiunti. Sulle 84 il contestuale e il suo prefisso hanno tre successi disgiunti per lato. Non si può quindi dire che la lunghezza abbia causalmente spiegato gli stessi casi.

## Rumore lessicale e combinazione col prefisso

Il rumore è una sola sostituzione da un dizionario fissato, con protezioni lessicali/NER; non è gold di equivalenza semantica. È applicabile a 70/126 originali e 42/84 nuovi, e i confronti con ciascun riscrittore usano anche l'intersezione con le generazioni valide.

Da solo, sui 70 casi eleggibili, produce evasioni OpenAI 6/41, Fakespot 2/60, RADAR 3/51, Binoculars 1/68 e maggioranza 2/52. Il rumore non è quindi sempre innocuo per il detector, ma ha un profilo diverso da Qwen. Sulle 84 nuove e sullo stesso sottoinsieme di riferimento, OpenAI ha 1/23 evasioni del rumore contro 15/23 del 3B e 2/24 contro 22/24 del 14B. Binoculars ha 1/39 contro 4/39 del 3B e 1/40 contro 9/40 del 14B.

Il prefisso con rumore ha ancora meno fonti eleggibili: 55 per 3B e DIPPER passaggio, 50 per 14B, 52 per DIPPER tre frasi. Sulle 126, OpenAI 3B dà 4/31 contro 20/31 riscritture; 14B 5/29 contro 27/29. Binoculars 3B dà 0/54 contro 8/54, 14B 2/49 contro 14/49. Le differenze appaiate sono positive con intervalli 95% rispettivamente [35,5; 67,7], [58,6; 89,7], [5,6; 24,1] e [10,2; 36,8] punti. Sulle 84, la differenza Binoculars del 3B ha invece intervallo [0,0; 22,6] e il suo sottoinsieme è piccolo: non si estende meccanicamente la precisione dei risultati complessivi.

## File e limiti della stima

- [Riepiloghi complessivi](detection_overall.csv): pannelli 126/84/42, denominatori, transizioni, CI e fallimenti.
- [Domini](detection_by_domain.csv) e [generatori](detection_by_generator.csv): analisi descrittive; nessuna graduatoria affidabile su sei o quattro fonti per cella.
- [Confronti appaiati fra riscrittori](detection_paired_writers.csv): differenze ASR e TPR, con denominatori distinti.
- [Controlli nulli](null_comparisons.csv): lunghezza, rumore, combinazione e confronti su eleggibilità comune; maggioranza derivata separatamente dalle tre decisioni RoBERTa.

Tutti gli IC primari usano 5.000 ricampionamenti dei gruppi di fonte mantenendo fissi checkpoint e soglie; non includono l'incertezza di calibrazione o di training. Per le 126 fonti gli ASR e i loro IC coincidono numericamente con i riepiloghi ricevuti. Gli IC delle nuove differenze sono esplorativi e non corretti per confronti multipli.

Con zero successi, il bootstrap può restituire [0; 0]: non prova un tasso di popolazione nullo. Nei CSV è incluso come riferimento supplementare un intervallo binomiale di Wilson, che assume prove indipendenti; i gruppi tutti distinti non dimostrano da soli quell'assunzione. Non sostituisce il protocollo e non aggiunge indipendenza fra detector. Né ASR, né TPR, né i controlli nulli misurano la conservazione del significato o la naturalezza.

# Confronto appaiato dell’evasione: DIPPER e Qwen

Confronto descrittivo sulle stesse 42 fonti del pilot, senza nuove generazioni o inferenze. Gli originali sono identici; gruppi di fonte, soglie dei detector e decisioni sugli originali coincidono entro 10⁻¹² per gli score. Ogni fonte ha un gruppo distinto. Si usano solo i file ricevuti, mantenendo le definizioni ASR del protocollo.

## Denominatore e coppie

Per ciascun detector si considerano soltanto gli originali già classificati AI. Il sottoinsieme è lo stesso per i due humanizer. Un’evasione richiede una generazione valida e uno score della riscrittura non superiore alla soglia fissata sugli umani di calibrazione. Per la maggioranza, AI significa almeno due voti AI su tre. Tutte le 42 generazioni di ciascun humanizer risultano valide; non sono escluse coppie per qualità del testo o per punteggio.

| Detector | Originali AI | Qwen successi | DIPPER successi | Entrambi | Solo Qwen | Solo DIPPER | Nessuno |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fakespot raw | 37 | 1 | 7 | 1 | 0 | 6 | 30 |
| OpenAI | 24 | 17 | 4 | 3 | 14 | 1 | 6 |
| RADAR | 28 | 2 | 3 | 1 | 1 | 2 | 24 |
| Maggioranza | 31 | 2 | 2 | 0 | 2 | 2 | 27 |

La maggioranza dà due successi a ciascun humanizer, ma nessuna fonte di successo è condivisa: due fonti vengono eluse solo da Qwen e altre due solo da DIPPER. L’uguaglianza dell’ASR aggregata non implica identità del comportamento. Numeratore e denominatore vanno sempre letti insieme.

## Differenza e incertezza esplorativa

La differenza è ASR(DIPPER) − ASR(Qwen). Il bootstrap ricampiona con rimpiazzo i gruppi di fonte all’interno del sottoinsieme baseline-AI, mantenendo insieme i due esiti della stessa fonte; qui ciascun gruppo contiene una fonte. Si usano 5.000 ricampionamenti, seed 42 per ciascun detector, intervallo percentile al 95%. I numeri sono calcolati sulla scala 0–1 e convertiti sotto in punti percentuali.

| Detector | ASR Qwen | ASR DIPPER | Differenza DIPPER−Qwen | Intervallo appaiato 95% |
| --- | ---: | ---: | ---: | --- |
| Fakespot raw | 2.7% | 18.9% | +16.2 pp | [+5.4; +27.0] pp |
| OpenAI | 70.8% | 16.7% | -54.2 pp | [-75.0; -29.2] pp |
| RADAR | 7.1% | 10.7% | +3.6 pp | [-7.1; +17.9] pp |
| Maggioranza | 6.5% | 6.5% | +0.0 pp | [-12.9; +12.9] pp |

DIPPER elude meno spesso OpenAI e più spesso Fakespot raw in questo pannello. RADAR e maggioranza hanno differenze poco determinate dal piccolo campione. Non si trasforma il risultato in una classifica universale: i detector hanno training differenti ma appartengono alla famiglia RoBERTa; i sistemi sono valutati con un singolo preset e i testi sono passaggi brevi di generatori storici.

Gli intervalli sono esplorativi, senza correzione per confronti multipli. Mantengono fissi checkpoint, soglie e sottoinsieme degli originali riconosciuti AI: non includono incertezza della calibrazione o dell’addestramento e non stimano l’incertezza in nuovi domini. Un bootstrap su pochi successi o su due sole fonti per cella dominio/generatore può avere copertura limitata. Non sono stati calcolati p-value.

L’evasione è un solo asse: non prova conservazione semantica, naturalezza o qualità della riscrittura. Le analisi qualitative e le annotazioni previste devono restare separate dai verdetti dei detector.

[CSV con conteggi e differenze](paired_asr_comparison.csv).

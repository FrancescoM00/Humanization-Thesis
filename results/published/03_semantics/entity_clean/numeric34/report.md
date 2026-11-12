# Controllo di robustezza: date, importi e percentuali

**L’NLI continua ad assegnare basso supporto alle alterazioni anche in questa famiglia più omogenea.** Il risultato è descrittivo su corruzioni controllate; non certifica ogni perturbazione né la superiorità sulle parafrasi corrette.

## Selezione e metodo

Selezione fissata dai metadati clean, senza usare gli score: **34 coppie e 34 fonti** fra le 145 entity applicate; 18 DATE, 6 MONEY, 10 PERCENT. CARDINAL, ORDINAL, PERSON e GPE sono esclusi. Nessuna coppia delle 34 è stata scartata dopo la lettura o la valutazione.

Sono riusati esclusivamente JSONL e CSV già disponibili: **zero nuove inferenze e zero modifiche ai dati**. Checkpoint e metriche sono quelli del run clean: DeBERTa-v3 base float32, MiniLM e BERTScore RoBERTa-base con il protocollo non riscalato del confronto storico. Il minimo bidirezionale è il minore fra minimo forward e reverse per documento.

La v1 è confrontata sulle stesse 34 fonti. Solo **12/34** hanno lo stesso tipo di entità nelle due politiche: la selezione DATE/MONEY/PERCENT riguarda la clean. Il confronto non isola un cambiamento dello stesso numero, della stessa frase o della stessa entità. IC 95%: bootstrap appaiato per fonte, 1000 ricampionamenti, seed 42; nessuna nuova soglia è calibrata.

## Stessa analisi del campione di 145, sulle 34 fonti

| misura | v1 | clean | mean_clean_minus_v1 | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| BERTScore F1 documento | 0.997558 | 0.999082 | 0.001524 | 0.000637 | 0.002636 |
| Coseno documento | 0.986625 | 0.998363 | 0.011738 | 0.001716 | 0.026700 |
| BERTScore F1 frase | 0.987657 | 0.992778 | 0.005121 | 0.002125 | 0.008119 |
| Coseno frase | 0.908060 | 0.954655 | 0.046596 | 0.016293 | 0.083657 |
| Entailment locale, originale → modificato | 0.013181 | 0.002572 | -0.010609 | -0.031625 | 0.000491 |
| Supporto minimo documento, bidirezionale | 0.004026 | 0.003436 | -0.000590 | -0.003632 | 0.001737 |

Il minimo NLI resta basso in entrambe le politiche. Le differenze fra v1 e clean non vanno interpretate come un miglioramento dell’accuratezza semantica: cambiano target e composizione delle sostituzioni. Gli IC delle differenze di entailment locale forward e minimo documentale includono zero: non emerge un vantaggio netto della clean rispetto alla v1 su queste 34 fonti. Le dieci misure complete, incluse le direzioni inverse, sono nei CSV summary e paired_deltas.

## Riferimento positivo: testo e frase identici

Ogni alterazione è accoppiata al controllo identità dell’intero originale per le misure documentali e al controllo identità della stessa frase bersaglio per le misure locali. Le identità sono state inferite realmente, non impostate a 1.

| misura | identity_mean | altered_mean | altered_min | altered_max | identity_vs_altered_auc |
| --- | --- | --- | --- | --- | --- |
| Entailment locale, originale → modificato | 0.991547 | 0.002572 | 0.000230 | 0.015787 | 1.000000 |
| Supporto minimo documento, bidirezionale | 0.984071 | 0.003436 | 0.000472 | 0.016151 | 1.000000 |
| Coseno documento | 1.000000 | 0.998363 | 0.983605 | 0.999964 | 1.000000 |
| Coseno frase | 1.000000 | 0.954655 | 0.712396 | 0.999103 | 1.000000 |
| BERTScore F1 documento | 1.000000 | 0.999082 | 0.996112 | 0.999882 | 1.000000 |
| BERTScore F1 frase | 1.000000 | 0.992778 | 0.970845 | 0.998577 | 1.000000 |

Entailment locale delle alterazioni: media **0.002572**, intervallo osservato [0.000230; 0.015787]. Sulle frasi identiche la media è **0.991547** e il minimo 0.982296: il calo non dipende solo dalla difficoltà intrinseca delle frasi.

Le identità sono controlli facili: anche coseno e BERTScore separano perfettamente queste identità dalle alterazioni. L’AUC contro identità, riportata soltanto come diagnostica secondaria, non dimostra un vantaggio discriminativo dell’NLI. Per quello servono riscritture corrette e abbastanza diverse: il confronto equo precedente resta il riferimento.

## Risultato per famiglia

| entity_type | n_pairs | local_nli_forward_mean | nli_bidirectional_min | embedding_cosine | local_embedding_cosine | bert_chunk_f1 | local_bert_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DATE | 18 | 0.001636 | 0.003760 | 0.997679 | 0.935849 | 0.999427 | 0.994157 |
| MONEY | 6 | 0.006264 | 0.003408 | 0.999182 | 0.972179 | 0.998217 | 0.987892 |
| PERCENT | 10 | 0.002042 | 0.002868 | 0.999102 | 0.977993 | 0.998980 | 0.993226 |

I tre gruppi confermano il basso entailment locale. Le numerosità 18/6/10 sono piccole e non giustificano una graduatoria di affidabilità fra date, importi e percentuali.

## Direzionalità, lettura dei testi e limiti

Un caso percentuale trasforma **“more than 10 percent” → “37%”**: il modificato è più specifico e può implicare l’originale. L’entailment locale forward è 0,014795, il reverse 0,994922. Il reverse alto è semanticamente plausibile, non un fallimento automatico né prova di contraddizione. Le sostituzioni possono togliere qualificatori o cambiare valuta, oltre al valore numerico.

La lettura delle 34 frasi da parte dell’assistente rileva anche **1940’s → 1998’s**, con ambiguità di decennio/forma. Non sono 34 contraddizioni gold: questa famiglia evita molte ambiguità PERSON/GPE ma mantiene limiti di contesto. Molte modifiche sono ampie; tutti i sei importi arrivano a somme da 26/47/83 milioni. Il controllo non misura l’affidabilità su piccoli scarti aritmetici, errori numerici realistici o humanizer nuovi.

**Conclusione per la tesi:** il basso supporto NLI e l’alta similarità persistono sul sottoinsieme DATE/MONEY/PERCENT; il risultato non è sostenuto soltanto dalle sostituzioni di persone o luoghi più problematiche. È un controllo di robustezza favorevole sulle perturbazioni considerate, con validità semantica e generalizzazione da verificare separatamente.

## Riproduzione e file

```bash
.venv/bin/python scripts/analyze_clean_numeric_subset.py
```

`cases.csv` contiene tutti i 34 ID, tipi, valori sostituiti, frasi prima/dopo e punteggi. `paired_scores.csv` permette il confronto v1/clean per fonte; `summary.csv`, `paired_deltas.csv` e `by_category.csv` riportano statistiche e IC. I riferimenti identità sono in `identity_references.csv` e `identity_summary.csv`.

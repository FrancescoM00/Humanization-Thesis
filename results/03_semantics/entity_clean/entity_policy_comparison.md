# Confronto fra sostituzioni di entità v1 e clean

Le sole condizioni autorizzate a cambiare sono entity_substitution e il relativo controllo identità locale. La coorte e tutte le altre coppie, incluse fixture e OpAI, sono confrontate per ID e contenuto integrale.

**Invarianza verificata:** 2007 coppie non-entity esattamente identiche; 61 colonne di metriche entro atol=1e-06, rtol=1e-06. I contatori cumulativi di lavoro non sono metriche semantiche e sono esclusi.

## Copertura sulla stessa coorte

| policy | n_cohort_sources | n_applied | n_skipped | coverage | n_sources_common_to_both |
| --- | --- | --- | --- | --- | --- |
| v1 | 200 | 200 | 0 | 1.000000 | 145 |
| clean | 200 | 145 | 55 | 0.725000 | 145 |

I casi non applicabili della politica clean sono saltati esplicitamente, senza rimpiazzare le fonti. Il confronto principale seguente usa solo le fonti applicabili a entrambe le politiche; le medie all_applied nel CSV sono descrittive e possono riflettere selezione diversa.

## Retrieval mantenuto fisso sui controlli

Prima di pianificare l'NLI, 2007 coppie integralmente identiche hanno riusato il retrieval storico, dopo verifica di checkpoint, provenienza del codice, equivalenza numerica degli embedding, hash degli input, chunk, budget e top-k. Coseno e BERTScore sono stati ricalcolati; nessun punteggio NLI è stato usato per scegliere le evidenze. Le nuove sostituzioni clean e i loro controlli locali mantengono retrieval appena calcolato.

Prima del congelamento differivano 4 array candidati in 3 coppie; I testi delle evidenze differivano in 1 casi. Questo evita di cambiare un fattore estraneo alla policy a causa di pareggi numerici. Dettagli in unchanged_alignment_reuse.json; allineamento ricalcolato conservato separatamente.

Il vecchio hash del modulo embedding è diverso e quel sorgente non è archiviato: non si sostiene un'equivalenza AST del codice storico. Il modulo usato dall'NLI storico coincide con quello attuale; si riusa il suo piano di evidenze su input e chunk identici, con coseni ricalcolati equivalenti entro la tolleranza dichiarata.

## Metriche sulle stesse fonti

| metric | v1 | clean | mean_clean_minus_v1 | ci_low | ci_high | n_sources |
| --- | --- | --- | --- | --- | --- | --- |
| bert_chunk_f1 | 0.997241 | 0.998089 | 0.000848 | 0.000312 | 0.001343 | 145 |
| embedding_cosine | 0.977648 | 0.983924 | 0.006276 | -0.003814 | 0.016373 | 145 |
| local_bert_f1 | 0.984824 | 0.989241 | 0.004417 | 0.002683 | 0.006085 | 145 |
| local_embedding_cosine | 0.864937 | 0.882543 | 0.017605 | -0.000181 | 0.035804 | 145 |
| local_nli_bidirectional_mean | 0.014329 | 0.002677 | -0.011651 | -0.029189 | -0.000147 | 145 |
| local_nli_forward_mean | 0.017020 | 0.003059 | -0.013961 | -0.032435 | -0.001662 | 145 |
| local_nli_reverse_mean | 0.066031 | 0.022272 | -0.043759 | -0.082127 | -0.009640 | 145 |
| nli_bidirectional_min | 0.016414 | 0.007270 | -0.009144 | -0.027363 | 0.003134 | 145 |
| nli_forward_min | 0.019723 | 0.009033 | -0.010690 | -0.028554 | 0.002437 | 145 |
| nli_reverse_min | 0.067017 | 0.053399 | -0.013618 | -0.058711 | 0.028298 | 145 |

Gli IC sono bootstrap appaiati per gruppo sorgente. Un calo del supporto non dimostra da solo che la perturbazione sia semanticamente valida: nessun punteggio NLI viene usato come gold label. Le categorie di entità e la frase bersaglio possono cambiare fra politiche anche sulla stessa fonte.

Il minimo bidirezionale è min(nli_forward_min, nli_reverse_min). La probabilità locale principale è local_nli_forward_mean; sono riportati anche reverse e minimo delle due direzioni. Coseno e BERTScore sono distinti fra documento e frase alterata. BERTScore qui usa lo stesso protocollo non riscalato del run storico, per un confronto omogeneo.

## Distribuzione delle categorie

| policy | entity_type | replacement_key | n_pairs |
| --- | --- | --- | --- |
| v1 | CARDINAL | CARDINAL:digit | 4 |
| v1 | CARDINAL | CARDINAL:word | 10 |
| v1 | DATE | DATE:month | 1 |
| v1 | DATE | DATE:relative | 6 |
| v1 | DATE | DATE:weekday | 2 |
| v1 | DATE | DATE:year | 6 |
| v1 | GPE | GPE | 16 |
| v1 | GPE | GPE:article | 2 |
| v1 | LOC | LOC | 12 |
| v1 | LOC | LOC:article | 2 |
| v1 | MONEY | MONEY | 6 |
| v1 | ORDINAL | ORDINAL | 13 |
| v1 | ORG | ORG | 51 |
| v1 | ORG | ORG:article | 10 |
| v1 | PERCENT | PERCENT | 5 |
| v1 | PERSON | PERSON | 38 |
| v1 | TIME | TIME | 16 |
| clean | CARDINAL | CARDINAL:digit | 8 |
| clean | CARDINAL | CARDINAL:word | 15 |
| clean | DATE | DATE:calendar | 1 |
| clean | DATE | DATE:month | 2 |
| clean | DATE | DATE:weekday | 2 |
| clean | DATE | DATE:year | 13 |
| clean | GPE | GPE | 33 |
| clean | GPE | GPE:article | 3 |
| clean | MONEY | MONEY | 6 |
| clean | ORDINAL | ORDINAL | 4 |
| clean | PERCENT | PERCENT | 10 |
| clean | PERSON | PERSON | 48 |

La politica clean limita le categorie e può modificare più menzioni coreferenti. Non certifica la coreferenza o la verità dei fatti; le corruzioni restano condizioni sperimentali.

## Riproduzione

```bash
.venv/bin/python scripts/compare_entity_policy.py
```

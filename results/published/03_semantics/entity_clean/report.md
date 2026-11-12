# Validazione semantica con sostituzioni di entità corrette

Testi macchina originali: **200**; seed 42. Campione selezionato tra i testi idonei a tutte le trasformazioni, bilanciando dominio e generatore per quanto consentito dagli idonei. Questa selezione introduce un bias esplicito: le frasi ambigue o senza entità non vengono rappresentate. Le trasformazioni non applicabili sono contate, non sostituite con identità; vedere `eligibility_manifest.json`.

| perturbazione | applicate | saltate |
| --- | --- | --- |
| identity | 200 | 0 |
| negation | 200 | 0 |
| sentence_shuffle | 200 | 0 |
| truncate | 200 | 0 |
| random_deletion | 200 | 0 |
| entity_substitution | 145 | 55 |

## Metriche effettivamente eseguite

| stage | n_pairs | elapsed_seconds | peak_rss_mib |
| --- | --- | --- | --- |
| lexical | 2297 | 69.324 | 213.566 |
| embedding | 2297 | 187.087 | 703.586 |
| bertscore | 2297 | 926.629 | 1567.152 |
| nli | 2297 | 159.911 | 1915.453 |

| perturbation | metric | n_pairs | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| entity_substitution | lexical_rouge_l_f1 | 145 | 0.988 | 0.985 | 0.990 |
| entity_substitution | lexical_bleu | 145 | 0.976 | 0.970 | 0.981 |
| entity_substitution | embedding_cosine | 145 | 0.984 | 0.978 | 0.989 |
| entity_substitution | bert_chunk_f1 | 145 | 0.998 | 0.998 | 0.998 |
| entity_substitution | nli_forward_mean | 145 | 0.831 | 0.805 | 0.854 |
| entity_substitution | nli_reverse_mean | 145 | 0.837 | 0.812 | 0.862 |
| identity | lexical_rouge_l_f1 | 200 | 1.000 | 1.000 | 1.000 |
| identity | lexical_bleu | 200 | 1.000 | 1.000 | 1.000 |
| identity | embedding_cosine | 200 | 1.000 | 1.000 | 1.000 |
| identity | bert_chunk_f1 | 200 | 1.000 | 1.000 | 1.000 |
| identity | nli_forward_mean | 200 | 0.992 | 0.992 | 0.992 |
| identity | nli_reverse_mean | 200 | 0.992 | 0.992 | 0.992 |
| negation | lexical_rouge_l_f1 | 200 | 0.996 | 0.995 | 0.996 |
| negation | lexical_bleu | 200 | 0.985 | 0.983 | 0.986 |
| negation | embedding_cosine | 200 | 0.998 | 0.998 | 0.998 |
| negation | bert_chunk_f1 | 200 | 0.999 | 0.999 | 0.999 |
| negation | nli_forward_mean | 200 | 0.894 | 0.885 | 0.904 |
| negation | nli_reverse_mean | 200 | 0.898 | 0.888 | 0.907 |
| random_deletion | lexical_rouge_l_f1 | 200 | 0.921 | 0.921 | 0.921 |
| random_deletion | lexical_bleu | 200 | 0.693 | 0.691 | 0.696 |
| random_deletion | embedding_cosine | 200 | 0.982 | 0.980 | 0.983 |
| random_deletion | bert_chunk_f1 | 200 | 0.962 | 0.962 | 0.963 |
| random_deletion | nli_forward_mean | 200 | 0.929 | 0.921 | 0.937 |
| random_deletion | nli_reverse_mean | 200 | 0.534 | 0.512 | 0.553 |
| sentence_shuffle | lexical_rouge_l_f1 | 200 | 0.492 | 0.475 | 0.509 |
| sentence_shuffle | lexical_bleu | 200 | 0.968 | 0.966 | 0.969 |
| sentence_shuffle | embedding_cosine | 200 | 0.999 | 0.999 | 1.000 |
| sentence_shuffle | bert_chunk_f1 | 200 | 0.999 | 0.998 | 0.999 |
| sentence_shuffle | nli_forward_mean | 200 | 0.989 | 0.987 | 0.991 |
| sentence_shuffle | nli_reverse_mean | 200 | 0.987 | 0.983 | 0.990 |
| truncate | lexical_rouge_l_f1 | 200 | 0.781 | 0.771 | 0.789 |
| truncate | lexical_bleu | 200 | 0.576 | 0.560 | 0.591 |
| truncate | embedding_cosine | 200 | 0.958 | 0.950 | 0.963 |
| truncate | bert_chunk_f1 | 200 | 0.979 | 0.978 | 0.980 |
| truncate | nli_forward_mean | 200 | 0.992 | 0.992 | 0.992 |
| truncate | nli_reverse_mean | 200 | 0.669 | 0.659 | 0.679 |

Gli IC 95% ricampionano i documenti sorgente (1.000 bootstrap), assegnando uguale peso a ciascuna fonte. Non descrivono l’incertezza della correttezza semantica delle perturbazioni. `paired_identity_deltas.csv` confronta ogni trasformazione con l’identità della stessa sorgente; i confronti locali usano l’esatta frase target contro se stessa.

![Metriche lessicali](lexical_perturbations.png)

## Confronto delle metriche neurali e sfide locali

![Metriche sui documenti](semantic_perturbations.png)

| dataset | perturbation | local_bert_f1 | local_embedding_cosine | local_nli_forward_contradiction_mean | local_nli_forward_mean | local_nli_forward_neutral_mean |
| --- | --- | --- | --- | --- | --- | --- |
| controlled | entity_substitution | 0.989 | 0.883 | 0.941 | 0.003 | 0.056 |
| controlled | negation | 0.984 | 0.869 | 0.998 | 0.001 | 0.001 |
| local_control | identity_for_entity_substitution | 1.000 | 1.000 | 0.002 | 0.991 | 0.007 |
| local_control | identity_for_negation | 1.000 | 1.000 | 9.82e-04 | 0.994 | 0.005 |

![Variazioni locali](local_semantic_changes.png)

Le probabilità NLI locali riguardano soltanto la frase modificata. Una variazione grande locale può diluirsi nella media di un documento lungo. Le scale dei modelli sono diverse: BERTScore non riscalato può essere numericamente alto anche per frasi non correlate. Usare le differenze rispetto al controllo identità della stessa frase, senza soglie universali.

## OpAI-Bench: asimmetria e verifica delle trasformazioni

| operation | comparison | n_pairs | n_sources | changed | shorter | longer | median_length_ratio | embedding_cosine | bert_chunk_f1 | nli_forward_mean | nli_reverse_mean | forward_minus_reverse | delta_ci_low | delta_ci_high | pair_fraction_forward_gt_reverse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| opai_compress | adjacent_operation | 200 | 190 | 197 | 197 | 0 | 0.913 | 0.993 | 0.985 | 0.986 | 0.920 | 0.067 | 0.056 | 0.082 | 0.930 |
| opai_compress | cumulative_from_human_v0 | 200 | 190 | 200 | 196 | 3 | 0.917 | 0.988 | 0.975 | 0.972 | 0.913 | 0.059 | 0.047 | 0.071 | 0.850 |
| opai_expand | adjacent_operation | 200 | 188 | 195 | 0 | 194 | 1.190 | 0.989 | 0.983 | 0.680 | 0.986 | -0.306 | -0.333 | -0.279 | 0.010 |
| opai_expand | cumulative_from_human_v0 | 200 | 188 | 200 | 52 | 146 | 1.063 | 0.978 | 0.965 | 0.740 | 0.913 | -0.174 | -0.200 | -0.148 | 0.145 |

![Asimmetria OpAI](opai_asymmetry.png)

Forward−reverse positivo è nella direzione attesa per una compressione che perde informazione; negativo per un’espansione che ne aggiunge. I casi immutati o con direzione di lunghezza diversa da quella nominale restano nel campione principale e sono contati. Il confronto v0 è cumulativo e comprende tutte le modifiche precedenti. Gli IC ricampionano il documento umano sorgente, raggruppando i suoi editor.

## Copertura dell’esecuzione

Tutti e quattro gli stadi sono stati eseguiti su tutte le coppie preparate.

**Campione OpAI-Bench preparato:** {"status": "prepared", "n_pairs": 800}

Gli stadi disponibili sono stati eseguiti con i checkpoint reali; nessun valore è stato riempito sulla base del risultato atteso.

## Definizioni e interpretazione

- ROUGE-L F1 usa stemming Porter; BLEU è SacreBLEU 13a, smoothing esponenziale, use_effective_order=True, una reference, scala 0–1. Le versioni locali sono sulle frasi effettivamente modificate.
- Coseno: medie normalizzate e pesate per token delle rappresentazioni di frasi/chunk MiniLM, senza troncamento del documento.
- BERTScore locale: RoBERTa-base layer10, IDF=False, rescale_with_baseline=False. Per documenti lunghi si usa una distinta approssimazione con matching di chunk, non il BERTScore integrale troncato.
- NLI: DeBERTa-v3-base addestrato su MNLI/FEVER/ANLI. Forward significa originale→modificato; reverse modificato→originale. La mappa delle etichette viene letta dal config. Le probabilità non sono una prova logica.
- Per ogni frase/chunk ipotesi si recuperano fino a 3 candidati e si conserva il supporto massimo. Si riportano media, minimo, q10, quota sotto 0.5 e contraddizione; il massimo può sovrastimare il supporto e il retrieval può perdere evidenza distribuita. I target locali evitano retrieval e diluizione.
- Negazione: alterazione intenzionale, non gold contradiction verificata per tutti i casi. Entity swap: contenuto alterato o non supportato, non necessariamente contraddittorio. Troncamento e omissione: spesso neutralità nella direzione che richiede informazione persa.
- Shuffle conserva le proposizioni: un NLI aggregato per frase può legittimamente restare alto. Questo test segnala il limite rispetto a coerenza e ordine narrativo.
- Le versioni OpAI sono cumulative: confronti principali v3→v4 Compress e v4→v5 Expand; v0→v4/v5 sono secondari. Le espansioni senza nuove informazioni possono essere entailed in entrambe le direzioni.

## Ripresa

Comandi e versioni sono nel README e in SEMANTIC_EXPERIMENT.md. Ogni stadio gira in un processo separato; la tabella degli stadi riporta il massimo RSS effettivamente osservato, non una stima della sola dimensione dei pesi. La cache delle rappresentazioni token BERT viene limitata a 512MiB tra documenti.

File completi: `scores.csv`, `metric_summary.csv`, `directional_deltas.csv`, `preparation_manifest.json`, `*_run.json`. I testi delle coppie e i metadati esatti delle modifiche restano nella directory dati locale indicata dai comandi di esecuzione (`data/semantic_full/` per il run completo), ignorata da Git.

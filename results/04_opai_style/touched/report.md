# Stile sulle sole frasi toccate di OpAI

Nel test la distanza RMS per frase è Polish 0.366, Paraphrase 0.717, Style 0.656: Polish è più leggero, ma l’ordine completo richiesto non è confermato dalla metrica primaria. Quando si include la copertura degli edit, le medie diventano 0.152, 0.369, 0.425. Questo distingue intensità locale e quantità del documento riscritta; non si scartano le metriche che non restituiscono l’ordine atteso.

Sono usate le stesse 200 traiettorie dello studio documentale. Il rilascio Parquet non include la colonna `sentence_ids`, ma conserva `sentences` e `sentence_labels`. Il costruttore ufficiale mantiene un array ordinato per gli ID s0001… e impone una frase per ID. Su tutte le traiettorie incluse sono state verificate cardinalità costante v0..v5, ricostruzione del testo, ancore non modificate identiche al v0 e assenza di cambi fuori dalle posizioni marcate. Gli ID sono dunque **ricostruiti dalla lineage posizionale nativa verificata**, non presenti come campo nel Parquet né ottenuti con matching approssimato.

Le etichette native sono cumulative: a ogni stadio il builder riscrive tutte le frasi già selezionate più le nuove. L’analisi primaria native_targeted include tali posizioni; actually_changed include solo quelle effettivamente diverse fra le versioni adiacenti. Questo distingue il budget di editing dal numero di edit riusciti.

| scope | operation | n_pairs | mean_touched_sentences | mean_before_words | mean_distance_sentences | n_mattr_pair_defined |
| --- | --- | --- | --- | --- | --- | --- |
| actually_changed | compress | 200 | 9.280 | 201.455 | 9.160 | 192 |
| actually_changed | expand | 200 | 11.200 | 203.285 | 11.010 | 192 |
| actually_changed | paraphrase | 200 | 5.100 | 109.100 | 4.995 | 184 |
| actually_changed | polish | 200 | 2.750 | 60.840 | 2.695 | 104 |
| actually_changed | style | 200 | 7.555 | 161.190 | 7.430 | 188 |
| native_targeted | compress | 200 | 10.305 | 219.785 | 10.105 | 198 |
| native_targeted | expand | 200 | 12.460 | 227.730 | 12.210 | 199 |
| native_targeted | paraphrase | 200 | 5.390 | 114.345 | 5.275 | 192 |
| native_targeted | polish | 200 | 3.500 | 74.890 | 3.415 | 138 |
| native_targeted | style | 200 | 8.400 | 178.375 | 8.235 | 200 |

## Distanze scalari predefinite

La metrica primaria è la radice della media dei quadrati di Δφ/σ sulle singole frasi allineate e sulle dimensioni fisse, poi aggregata sulle sole frasi toccate. Non viene divisa per il numero di frasi dell’intero documento. Le σ sono apprese soltanto dalle frasi native v0 delle fonti di calibrazione, con uguale peso totale per fonte; sono esclusi conteggi globali, TTR/hapax, MATTR50, SD/CV di frase e proxy retorici Wikipedia. Le feature costanti nella calibrazione sono eliminate una sola volta per tutte le operazioni. Frasi con meno di cinque parole lessicali non entrano nella distanza primaria e vengono contate.

Si riportano anche la norma sul passaggio concatenato e la distanza Jensen–Shannon delle distribuzioni POS del passaggio (base 2, intervallo 0–1). Tutte le 105 feature, inclusa la shortlist, sono comunque estratte sui passaggi toccati concatenati. MATTR resta NaN sotto 50 parole: non viene riempito con zero o cambiando finestra.

La diagnostica distance_rms_coverage moltiplica la RMS delle frasi toccate per sqrt(n_frasi_toccate_valide/N_frasi_documento). Equivale a distribuire l’energia quadratica degli edit sull’intero documento, assegnando delta zero alle frasi intatte. Separa così la copertura spaziale dall’intensità per frase; non sostituisce la metrica primaria e non ritocca le scale. Il numero di frasi toccate cresce per costruzione tra le operazioni, quindi un ordine di questa diagnostica può riflettere anche il budget.

Per l’energia pesata per copertura i due scope coincidono: i target invariati hanno delta zero. I bootstrap impiegano gli stessi ricampionamenti tra scope per ogni metrica/operazione, evitando differenze Monte Carlo artificiose quando gli score sono identici.

L’ordine Polish < Paraphrase < Style Rewrite è un’ipotesi di coerenza, non una gold label e non viene imposto scegliendo pesi dopo i risultati. Le operazioni possono modificare dimensioni differenti; uno stile più formale non comporta necessariamente maggiore distanza POS.

## Distanze nel test separato

| operation | metric | n_pairs | n_sources | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| compress | distance_rms_sentences | 103 | 99 | 0.659 | 0.609 | 0.709 |
| compress | distance_rms_pooled | 103 | 99 | 0.257 | 0.233 | 0.282 |
| compress | distance_pos_js_pooled | 103 | 99 | 0.090 | 0.080 | 0.099 |
| compress | distance_rms_coverage | 103 | 99 | 0.469 | 0.436 | 0.500 |
| expand | distance_rms_sentences | 103 | 99 | 0.530 | 0.489 | 0.573 |
| expand | distance_rms_pooled | 103 | 99 | 0.201 | 0.184 | 0.222 |
| expand | distance_pos_js_pooled | 103 | 99 | 0.080 | 0.072 | 0.088 |
| expand | distance_rms_coverage | 103 | 99 | 0.415 | 0.384 | 0.445 |
| paraphrase | distance_rms_sentences | 103 | 99 | 0.717 | 0.644 | 0.788 |
| paraphrase | distance_rms_pooled | 103 | 99 | 0.320 | 0.290 | 0.349 |
| paraphrase | distance_pos_js_pooled | 103 | 99 | 0.100 | 0.090 | 0.110 |
| paraphrase | distance_rms_coverage | 103 | 99 | 0.369 | 0.332 | 0.406 |
| polish | distance_rms_sentences | 103 | 99 | 0.366 | 0.317 | 0.416 |
| polish | distance_rms_pooled | 103 | 99 | 0.210 | 0.178 | 0.241 |
| polish | distance_pos_js_pooled | 103 | 99 | 0.056 | 0.048 | 0.066 |
| polish | distance_rms_coverage | 103 | 99 | 0.152 | 0.129 | 0.173 |
| style | distance_rms_sentences | 103 | 99 | 0.656 | 0.581 | 0.731 |
| style | distance_rms_pooled | 103 | 99 | 0.250 | 0.215 | 0.303 |
| style | distance_pos_js_pooled | 103 | 99 | 0.072 | 0.064 | 0.081 |
| style | distance_rms_coverage | 103 | 99 | 0.425 | 0.378 | 0.476 |

IC 95% bootstrap per documento sorgente, dopo la media degli editor. La scala è appresa sulle sole fonti di calibrazione; i risultati test non partecipano al suo apprendimento.

## Ordine nelle stesse traiettorie

| metric | quantity | n_trajectories | n_sources | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| distance_rms_sentences | strict_order | 103 | 99 | 0.338 | 0.247 | 0.419 |
| distance_rms_sentences | nonstrict_order | 103 | 99 | 0.338 | 0.247 | 0.434 |
| distance_rms_sentences | para_minus_polish | 103 | 99 | 0.352 | 0.282 | 0.426 |
| distance_rms_sentences | style_minus_para | 103 | 99 | -0.061 | -0.126 | 0.016 |
| distance_rms_pooled | strict_order | 103 | 99 | 0.116 | 0.061 | 0.187 |
| distance_rms_pooled | nonstrict_order | 103 | 99 | 0.116 | 0.056 | 0.182 |
| distance_rms_pooled | para_minus_polish | 103 | 99 | 0.110 | 0.074 | 0.146 |
| distance_rms_pooled | style_minus_para | 103 | 99 | -0.070 | -0.111 | -0.016 |
| distance_pos_js_pooled | strict_order | 103 | 99 | 0.106 | 0.051 | 0.162 |
| distance_pos_js_pooled | nonstrict_order | 103 | 99 | 0.116 | 0.061 | 0.182 |
| distance_pos_js_pooled | para_minus_polish | 103 | 99 | 0.044 | 0.031 | 0.056 |
| distance_pos_js_pooled | style_minus_para | 103 | 99 | -0.028 | -0.039 | -0.016 |
| distance_rms_coverage | strict_order | 103 | 99 | 0.611 | 0.515 | 0.702 |
| distance_rms_coverage | nonstrict_order | 103 | 99 | 0.611 | 0.520 | 0.697 |
| distance_rms_coverage | para_minus_polish | 103 | 99 | 0.217 | 0.182 | 0.250 |
| distance_rms_coverage | style_minus_para | 103 | 99 | 0.057 | 0.009 | 0.114 |

La frazione strict_order conta Polish < Paraphrase < Style entro la stessa traiettoria; nonstrict_order ammette pareggi. I due delta misurano Paraphrase−Polish e Style−Paraphrase. Questi risultati distinguono ordine delle medie e ordine nei singoli esempi.

## Shortlist sui passaggi toccati

| operation | feature | n_pairs | n_sources | mean_delta | delta_ci_low | delta_ci_high | q_bh_all_features_operations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| polish | mattr_50 | 138 | 129 | 0.009 | 0.006 | 0.013 | 1.48e-05 |
| polish | pos_cconj | 200 | 188 | -9.63e-04 | -0.002 | 9.21e-05 | 0.184 |
| polish | pos_num | 200 | 188 | -2.54e-04 | -7.32e-04 | 2.22e-04 | 0.436 |
| polish | pos_propn | 200 | 188 | -4.42e-04 | -0.002 | 9.60e-04 | 0.662 |
| paraphrase | mattr_50 | 192 | 180 | 0.023 | 0.018 | 0.028 | 2.89e-18 |
| paraphrase | pos_cconj | 200 | 188 | -0.002 | -0.003 | -0.001 | 8.08e-04 |
| paraphrase | pos_num | 200 | 188 | -1.79e-04 | -7.98e-04 | 3.65e-04 | 0.700 |
| paraphrase | pos_propn | 200 | 188 | 8.33e-04 | -4.02e-04 | 0.002 | 0.393 |
| style | mattr_50 | 200 | 188 | 0.005 | 0.001 | 0.008 | 0.015 |
| style | pos_cconj | 200 | 188 | -0.002 | -0.003 | -0.001 | 7.25e-06 |
| style | pos_num | 200 | 188 | -4.21e-04 | -9.08e-04 | 6.76e-05 | 0.206 |
| style | pos_propn | 200 | 188 | 5.13e-05 | -0.001 | 0.001 | 0.982 |
| compress | mattr_50 | 198 | 186 | 0.008 | 0.005 | 0.012 | 7.33e-06 |
| compress | pos_cconj | 200 | 188 | 0.005 | 0.004 | 0.006 | 4.29e-18 |
| compress | pos_num | 200 | 188 | 0.002 | 0.001 | 0.003 | 5.78e-10 |
| compress | pos_propn | 200 | 188 | 0.009 | 0.008 | 0.011 | 1.09e-19 |
| expand | mattr_50 | 199 | 187 | 0.010 | 0.007 | 0.013 | 3.60e-10 |
| expand | pos_cconj | 200 | 188 | -9.15e-04 | -0.002 | 2.48e-04 | 0.259 |
| expand | pos_num | 200 | 188 | -0.003 | -0.004 | -0.003 | 7.99e-24 |
| expand | pos_propn | 200 | 188 | -0.014 | -0.016 | -0.012 | 2.21e-29 |

Gli effetti sulle feature sono esplorativi e usano le fonti come unità; BH corregge le 105×5 ipotesi entro scope. Le parti toccate hanno lunghezze e contenuti diversi: questo disegno rimuove la diluizione nel documento intero, non trasforma la variazione in una prova di naturalezza.

## Documento intero e passaggi sugli stessi esempi validi

Il confronto seguente usa esattamente gli stessi pair_id ed editor per le due misure, con entrambe le versioni finite sia nel documento intero sia nel passaggio toccato. È essenziale per MATTR50: non si confronta il sottocampione di passaggi con almeno 50 parole con tutti i documenti. Si mediano prima gli editor della stessa fonte e si ricampiona il contrasto appaiato Δtarget−Δfull. Le differenze di scala e di passaggio osservato rimangono: un contrasto è descrittivo, non una stima causale della sola diluizione.

| operation | feature | n_pairs_common | n_sources_common | mean_delta_target | mean_delta_full_same_pairs | mean_paired_target_minus_full | contrast_ci_low | contrast_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compress | pos_num | 200 | 188 | 0.002 | 0.001 | 9.34e-04 | 6.01e-04 | 0.001 |
| compress | pos_cconj | 200 | 188 | 0.005 | 0.002 | 0.003 | 0.002 | 0.003 |
| compress | mattr_50 | 198 | 186 | 0.008 | 0.004 | 0.005 | 0.002 | 0.007 |
| compress | pos_propn | 200 | 188 | 0.009 | 0.004 | 0.005 | 0.004 | 0.006 |
| expand | pos_num | 200 | 188 | -0.003 | -0.002 | -8.99e-04 | -0.001 | -6.57e-04 |
| expand | pos_cconj | 200 | 188 | -9.15e-04 | -4.22e-04 | -4.92e-04 | -9.53e-04 | -9.84e-07 |
| expand | mattr_50 | 199 | 187 | 0.010 | 0.009 | 0.001 | -5.67e-04 | 0.003 |
| expand | pos_propn | 200 | 188 | -0.014 | -0.009 | -0.005 | -0.006 | -0.004 |
| paraphrase | pos_num | 200 | 188 | -1.79e-04 | -3.98e-06 | -1.75e-04 | -6.21e-04 | 2.36e-04 |
| paraphrase | pos_cconj | 200 | 188 | -0.002 | -6.36e-04 | -0.002 | -0.003 | -8.83e-04 |
| paraphrase | mattr_50 | 192 | 180 | 0.023 | 0.008 | 0.015 | 0.011 | 0.019 |
| paraphrase | pos_propn | 200 | 188 | 8.33e-04 | 2.66e-04 | 5.67e-04 | -4.47e-04 | 0.002 |
| polish | pos_num | 200 | 188 | -2.54e-04 | -3.55e-05 | -2.18e-04 | -5.90e-04 | 1.45e-04 |
| polish | pos_cconj | 200 | 188 | -9.63e-04 | -1.17e-04 | -8.45e-04 | -0.002 | 7.78e-05 |
| polish | mattr_50 | 138 | 129 | 0.009 | 0.001 | 0.007 | 0.004 | 0.011 |
| polish | pos_propn | 200 | 188 | -4.42e-04 | -1.25e-05 | -4.29e-04 | -0.002 | 6.10e-04 |
| style | pos_num | 200 | 188 | -4.21e-04 | -2.29e-04 | -1.92e-04 | -4.73e-04 | 7.91e-05 |
| style | pos_cconj | 200 | 188 | -0.002 | -9.15e-04 | -0.001 | -0.002 | -8.40e-04 |
| style | mattr_50 | 200 | 188 | 0.005 | 0.003 | 0.002 | -7.08e-04 | 0.004 |
| style | pos_propn | 200 | 188 | 5.13e-05 | 1.19e-04 | -6.78e-05 | -6.76e-04 | 5.34e-04 |

File: `distance_summary.csv`, `ordinal_summary.csv`, `trajectory_orders.csv`, `touched_feature_values.csv`, `aligned_sentence_distances.csv`, `feature_summary.csv`, `matched_scope_comparison.csv`, `standardizer.json`, `manifest.json`. I testi restano in `data/opai_touched_style/`. Riproduzione: `.venv/bin/python scripts/run_opai_touched_style.py`; aggiungere `--reuse-features` per usare le feature già estratte.

## Lettura diagnostica di cinque inversioni

Casi estremi selezionati per diagnosi (cinque fonti diverse del test con Style−Paraphrase più negativo nella metrica primaria). **Non sono un campione rappresentativo e non stimano un tasso d’errore.** Sono stati letti i tre passaggi nativi; nessun peso o campione è stato modificato dopo la lettura.

### 1. news-8bcb76ac2d5940a5 / qwen3-8b

Distanze Polish / Paraphrase / Style: 0.000 / 1.459 / 0.000; posizione s0007.

Estratti v1 → v2 → v3: «"You need to approach every game to win it."» → «Every game should be approached with the goal of winning,» → «Every game should be approached with the goal of winning,».

Style lascia invariati tutti i target di questa traiettoria: distanza zero è corretta, ma rende impossibile l’ordine stretto. L’etichetta dell’operazione non garantisce un edit riuscito.

### 2. reports-2bf4ad59f83db429 / qwen3-8b

Distanze Polish / Paraphrase / Style: 0.000 / 1.458 / 0.200; posizione s0003.

Estratti v1 → v2 → v3: «Political advertising must also comply with requirements overseen by the FEC…» → «Political advertisements are also subject to regulations managed by the FEC.» → «Political advertisements are also subject to regulations enforced by the Fed…».

Paraphrase cambia la costruzione modale e sintattica (must comply → are subject to); Style interviene soprattutto espandendo FEC e cambiando un verbo. Una distanza strutturale maggiore nel primo passo è plausibile.

### 3. abstracts-2710ff93a5533f4d / gpt-5.4

Distanze Polish / Paraphrase / Style: 0.424 / 1.617 / 0.616; posizione s0011.

Estratti v1 → v2 → v3: «generative modeling, uncertainty estimation, planning, etc.)» → «generative modeling, uncertainty estimation, planning, and so on)» → «generative modeling, uncertainty estimation, planning, among others)».

La posizione nativa è un frammento di elenco terminato da etc., non una frase grammaticale completa. Cambi piccoli di punteggiatura e parole funzionali pesano molto su un’unità breve. IDs/lineage validi non garantiscono segmentazione linguistica ideale.

### 4. reports-843e551986585c39 / gemini-2.5-flash

Distanze Polish / Paraphrase / Style: 0.476 / 1.221 / 0.275; posizione s0009.

Estratti v1 → v2 → v3: «For selected non-Annex I nations--Brazil, China, India, Indonesia, Malaysia,…» → «GAO discovered that most inventories from the chosen non-Annex I nations—Bra…» → «GAO found that most inventories from the chosen non-Annex I nations—Brazil, …».

Paraphrase sposta l’attribuzione GAO in apertura e riorganizza il periodo; Style conserva tale struttura e sostituisce pochi termini. Il passo formale incrementale è più piccolo della ristrutturazione precedente.

### 5. essays-8b1f33abb750e48c / gpt-5.4

Distanze Polish / Paraphrase / Style: 0.892 / 1.454 / 0.542; posizione s0003.

Estratti v1 → v2 → v3: «The human race uses more and more technology, but technology does not unders…» → «Although humans rely increasingly on technology, it does not comprehend huma…» → «Although humans are becoming increasingly dependent on technology, it does n…».

Paraphrase passa dalla coordinazione con but a una subordinata con Although; Style conserva la subordinata e cambia soprattutto il lessico. Le due operazioni agiscono su assi e storie diverse.

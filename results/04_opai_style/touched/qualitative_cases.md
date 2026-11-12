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

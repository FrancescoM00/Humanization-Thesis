# Migrazione e conservazione del lavoro precedente

La nuova repository deriva da `FrancescoM00/Humanization` al commit `d2868b323938a2a9371c2a872ad86b5596d0f097`. Il repository originale rimane separato, con codice, risultati, cronologia e organizzazione precedenti. Non vengono spostati file fuori dall'originale e non viene cambiato il suo ramo `main`.

## Organizzazione

Il percorso principale segue le sezioni 1.2–1.9 del PDF indicato nella [mappa](MAPPATURA_TESI.md). La logica condivisa è sotto `src/`; i CLI rimangono piatti sotto `scripts/` per evitare copie dello stesso algoritmo per capitolo. I risultati sono raggruppati per esperimento. Le nuove esecuzioni hanno output sotto `runs/`.

Le implementazioni scientifiche sono state adattate negli import, nei percorsi e nei riferimenti al codice. Un adattatore esplicito, `src/humanization_paths.py`, traduce i percorsi degli input dei protocolli storici senza riscrivere quei protocolli. I notebook usano il nuovo progetto, i CSV locali e i preset organizzati in `configs/`.

Sono esclusi gli strumenti e gli output del giudice di naturalezza, le sue celle nei notebook, gli smoke e i tentativi di ripresa, i backup dell'ambiente e i pesi. L'estensione sulle cautele rimane separata in `supplements/` perché successiva al PDF. La nota che uno score stilistico non misuri naturalezza può rimanere come limite interpretativo; non costituisce un esperimento aggiuntivo.

## Provenienza

I tre CSV appena forniti dall'autore sono copiati esattamente in `data/raw/`. La prima acquisizione del progetto originale era avvenuta tramite un mirror: la verifica successiva delle copie ufficiali dimostra che i contenuti coincidono, senza riscrivere la storia del primo download.

I CSV numerici e gli input congelati conservano i byte degli originali. Gli hash del codice e i percorsi dentro i manifest storici identificano l'esecuzione originale, non questa nuova organizzazione. Non vengono sostituiti retroattivamente con nuovi hash. Alcuni report Markdown sono presentati con le sezioni fuori perimetro rimosse: sono copie editoriali per la consegna, non nuovi risultati scientifici.

[`migration_manifest.json`](migration_manifest.json) registra la mappa delle cartelle e gli hash sorgente/destinazione degli artefatti copiati. Le modifiche al codice e le nuove analisi hanno la cronologia Git di questa repository. La validazione verifica campioni e risultati effettivi, oltre agli import.

## Riproduzione delle tabelle

Il nuovo comando `scripts/analyze_humanizer_results.py` rende esplicita l'analisi dei pilot e dello studio, precedentemente eseguita durante la lavorazione degli archivi ricevuti. Legge riscritture e punteggi intermedi, ricostruisce le metriche e confronta i risultati con i CSV consegnati. Le formule e i semi bootstrap sono quelli recuperati dalle analisi originali: non vengono scelti per migliorare i risultati.

Le annotazioni umane ancora da ricevere non sono sostituite con giudizi dell'assistente. Non sono stati rilanciati riscrittori, detector o modelli NLI per la riorganizzazione.

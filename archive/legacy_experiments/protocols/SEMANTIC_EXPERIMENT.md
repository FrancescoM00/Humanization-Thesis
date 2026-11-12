# Protocollo della baseline semantica v1

Questo documento riproduce la baseline v1. Le estensioni sono nei protocolli [controlli positivi e sottili](SEMANTIC_CONTROLS.md), [confronto equo e aggregatori](FAIRNESS_EXPERIMENT.md) ed [entity clean](ENTITY_CLEAN_EXPERIMENT.md). La revisione testuale ha rilevato errori e ambiguità nelle perturbazioni originali: i risultati non equivalgono a una validazione gold.

Questa pipeline verifica empiricamente la sensibilità delle metriche alle alterazioni. L'ipotesi che BERTScore/coseno restino alti mentre NLI diminuisce **non è codificata nei risultati**. Ogni valore richiede esecuzione del relativo modello.

Il percorso completo è stato eseguito: [interpretazione](../../results/03_semantics/baseline/interpretazione.md) e [report con grafici](../../results/03_semantics/baseline/report.md). I risultati sono in `results_semantic_full/`: 2.407 coppie, comprendenti 1.200 condizioni controllate, 400 controlli locali, 7 sanity check e 800 coppie OpAI (200 per ciascuno dei quattro confronti adiacenti/cumulativi). Il primo output soltanto lessicale è recuperabile nella cronologia Git.

## Installazione e modelli

Da questa radice, con Python 3.12 e ambiente virtuale:

```bash
.venv/bin/python -m pip install -r requirements-semantic-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python scripts/download_semantic_models.py
.venv/bin/python scripts/run_semantic_experiment.py
```

I modelli sono pubblici; non servono token né credenziali nuove:

| Stadio | Checkpoint |
| --- | --- |
| Coseno e recupero frasi | sentence-transformers/all-MiniLM-L6-v2 |
| BERTScore | FacebookAI/roberta-base, layer 10 |
| NLI | MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli |

Il downloader risolve la revisione esatta, scarica solo file necessari tramite HTTPS verificato, confronta gli hash LFS quando disponibili e salva `DOWNLOAD_MANIFEST.json`. Nessun remote code viene abilitato. I file restano in `data/models/`, ignorati da Git.

Per il download la rete deve consentire `huggingface.co`, `us.aws.cdn.hf.co`, `cas-bridge.xethub.hf.co`, `cdn-lfs.huggingface.co` e `cdn-lfs-us-1.hf.co`. I tre checkpoint eseguiti sono fissati in `results_semantic_full/models_provenance.json`. Il downloader accetta `--model embedding|bertscore|nli --revision SHA` per ripetere il download della revisione registrata.

Ogni stadio esegue in un processo separato per rilasciare la RAM prima del successivo. Device CPU; il run completo usa quattro thread, batch 4 per BERTScore e batch 8 per NLI. Picchi RSS e versioni sono salvati in `*_run.json`; rappresentano la memoria effettivamente osservata nel processo, non la sola dimensione dei pesi.

## Campione e perturbazioni

La sorgente è il campione MGTBench macchina originale ripulito della prima fase, **non** i prefissi del controllo stilometrico di lunghezza. Si verificano prima gli idonei tra i 900 testi macchina. Si selezionano 200 documenti con bilanciamento dominio×generatore per quanto permesso dall'idoneità; tutte e cinque le trasformazioni devono essere realmente applicabili. Il manifest conserva candidati, esclusioni, ragioni e bias di selezione.

- **Negazione:** inserimento grammaticale su una proposizione affermativa riconosciuta dal parser; evita negazioni esistenti, scope complessi e soggetti quantificati. È una contraddizione intenzionale, non una gold label verificata su ogni testo.
- **Entità:** sostituzione consistente di un'entità con un'altra dello stesso tipo, usando NER e dizionari prefissati; gestione di articoli e duplicati. La nuova proposizione può essere neutrale/non supportata invece di contraddittoria.
- **Shuffle:** permutazione dell'ordine delle frasi, mantenendo le proposizioni. È un controllo di coerenza, non necessariamente di verità.
- **Troncamento:** conservazione di circa 70% delle frasi complete, senza tagliare a metà frase.
- **Deletion:** rimozione deterministica del 15% delle parole alfabetiche, conservando offset e lista degli edit; può alterare significato e grammaticalità.

Seed 42, ID stabili SHA-256 e metadati degli edit permettono l'audit. Si aggiungono identità documentali, identità delle stesse frasi target per i confronti locali e sette fixture manuali separate. La selezione di testi idonei introduce un bias: i risultati non rappresentano automaticamente tutte le generazioni MGTBench.

## Metriche, documenti lunghi e direzioni

ROUGE-L F1 con stemming Porter; BLEU SacreBLEU 13a, smoothing esponenziale, `use_effective_order=True`, originale come unica reference, scala 0–1. Sono calcolati su documento e, per negazione/entityswap, sulla frase modificata.

Coseno: media pesata per token dei vettori normalizzati MiniLM di frasi/chunk. BERTScore locale standard con IDF e rescaling disattivati; per documenti lunghi si usa un'approssimazione esplicita con matching di chunk e aggregazioni distinte di precision/recall. Non si presenta questa approssimazione come BERTScore integrale.

La segmentazione copre tutto il testo. Frasi oltre il budget vengono divise in ulteriori chunk; nessun default `truncation=True` può eliminare silenziosamente la clausola alterata. Il budget documentale è 160 token per chunk; NLI ripartisce ulteriormente per rispettare 384 token della coppia.

**Forward**: originale come premessa, modificato come ipotesi. **Reverse**: modificato come premessa, originale come ipotesi. Le etichette entailment/neutral/contradiction vengono lette dal config, senza presumere l'ordine.

Per ogni ipotesi si recuperano fino a tre frasi/chunk di evidenza e si calcola il massimo supporto NLI. Si registrano media, minimo, q10, quota di ipotesi con supporto <0.5, neutralità e contraddizione. Questo è supporto aggregato fra frasi, **non una prova logica di entailment dell'intero documento**: non combina evidenza distribuita fra più frasi e il massimo può gonfiare il supporto. Le frasi modificate vengono anche confrontate direttamente, evitando retrieval e diluizione.

Una negazione in un documento lungo può lasciare quasi invariata la media, pur essendo riconosciuta localmente. Lo shuffle può conservare entailment alto perché conserva il contenuto proposizionale. Compressione/deletion possono ridurre reverse per perdita di informazione senza aumentare contradiction.

Ogni perturbazione viene confrontata con l'identità **della stessa sorgente**, con IC bootstrap per sorgente; i controlli locali usano l'esatta frase originale contro se stessa. Non si confrontano campioni di difficoltà diverse come se fossero appaiati.

## OpAI-Bench: Compress ed Expand

Fonte ufficiale: [VILA-Lab/OpAI-Bench](https://github.com/VILA-Lab/OpAI-Bench), commit `739d647bab87c7d3c7a86d3df6c008945af5c4a2`; dataset [OpAI-Bench1/OpAI-Bench](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench). Il repository non include i dati, distribuiti su Hugging Face.

```bash
.venv/bin/python scripts/download_opai.py
.venv/bin/python scripts/run_semantic_experiment.py --opai-file data/opai/records.jsonl
```

La revisione del dataset è risolta e fissata a SHA prima del download. Sono accettati anche export locali ufficiali JSONL/JSON/CSV/Parquet tramite `--opai-file`; schema, ID e versioni ambigue causano un errore, non un allineamento inventato.

Il run usa la revisione `072c03c7051409346aaba05168d183cf6d863ee9`, split `test`. Il downloader verifica gli SHA-256 LFS dei 16 Parquet ufficiali in `viewer/default/test` della stessa revisione (circa 71 MB, anziché l'export CSV molto più grande), conserva gli originali e proietta i 19.016 record v0/v3/v4/v5. Per ripetere la stessa fonte: `scripts/download_opai.py --revision 072c03c7051409346aaba05168d183cf6d863ee9`. Provenienza e checksum in `opai_manifest.json`.

Le versioni sono cumulative. Il confronto principale è **v3→v4 per Compress** e **v4→v5 per Expand**, campionando fino a 200 traiettorie per operazione e bilanciando gli strati disponibili. Il confronto secondario v0→v4/v5 misura anche le trasformazioni precedenti; v0 è umano. Gli output vengono allineati per documento, dominio e generatore; frasi locali soltanto quando esistono ID stabili espliciti, mai dalla sola posizione.

L'ipotesi per Compress è forward>reverse se si perde informazione. Per Expand ci si aspetta reverse>forward **soltanto se si aggiunge informazione**: se l'espansione esplicita ciò che era già implicato, l'entailment può legittimamente restare bidirezionale. Rapporti di lunghezza ed etichette operative non sono gold label NLI.

## Riprendere senza perdere lavoro

Il percorso nativo PyTorch è sempre disponibile, ma il modello base su CPU richiede molti confronti. Il run completo usa invece lo **stesso checkpoint float32** esportato in ONNX, senza quantizzazione, con controllo indipendente di parità su probabilità PyTorch reali. I confronti vengono deduplicati globalmente e ordinati per numero di token: cambiano scheduling e padding, non le evidenze o l'aggregazione. Le probabilità PyTorch già salvate vengono riutilizzate soltanto sul medesimo checkpoint.

Per riprodurre il percorso CPU completo (ogni comando va dalla radice del progetto):

```bash
.venv/bin/python scripts/run_semantic_experiment.py --stage prepare --opai-file data/opai/records.jsonl --data-dir data/semantic_full --output results_semantic_full
.venv/bin/python scripts/run_semantic_experiment.py --stage lexical --data-dir data/semantic_full --output results_semantic_full
.venv/bin/python scripts/run_semantic_experiment.py --stage embedding --data-dir data/semantic_full --output results_semantic_full --threads 4
.venv/bin/python scripts/run_semantic_experiment.py --stage bertscore --data-dir data/semantic_full --output results_semantic_full --threads 4
.venv/bin/optimum-cli export onnx --model data/models/nli --task text-classification --device cpu --dtype fp32 --opset 17 data/models/nli_onnx
.venv/bin/python scripts/plan_nli_jobs.py
.venv/bin/python scripts/initialize_nli_reference_cache.py
.venv/bin/python scripts/precompute_nli_onnx.py
.venv/bin/python scripts/run_semantic_experiment.py --stage nli --data-dir data/semantic_full --output results_semantic_full --threads 4 --require-nli-cache
.venv/bin/python scripts/run_semantic_experiment.py --stage report --data-dir data/semantic_full --output results_semantic_full
```

In ripresa si saltano preparazione, export e inizializzazione già completati. Il planner produce 88.900 coppie di evidenza uniche in questo campione. `--require-nli-cache` impedisce inferenze mancanti o fallback inattesi durante l'aggregazione. I metadati finali di run e i controlli di parità e BERTScore restano nei risultati; gli strumenti temporanei di profiling sono recuperabili nella cronologia Git. ONNX non viene chiamato bit-per-bit identico: si registra l'errore numerico osservato e la concordanza delle classi.

Il report rifiuta score di un campione diverso verificando l'hash delle coppie. Embedding e rappresentazioni BERT sono deduplicati per frase; NLI memorizza le coppie uniche e salva checkpoint delle probabilità nella directory dati scelta, ad esempio `data/semantic_full/nli_probability_cache.jsonl`, associati al modello. Non assegna artificialmente entailment 1 alle identità: anche i controlli vengono inferiti dal modello.

Risultati completi: `results_semantic_full/report.md`, `scores.csv`, `metric_summary.csv`, `paired_identity_deltas.csv`, `directional_deltas.csv`, manifest e controlli numerici. L'esito della verifica finale è conservato in [validation.json](../../results/03_semantics/baseline/validation.json). I testi completi restano in `data/semantic_full/`, ignorati da Git. Le fixture con modelli simulati verificano il software e **non sostituiscono la validazione dei checkpoint reali**.

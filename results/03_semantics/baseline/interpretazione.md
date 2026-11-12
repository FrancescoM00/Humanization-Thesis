# Conservazione semantica: interpretazione del run completo

**Aggiornamento:** il successivo [controllo con positivi e perturbazioni sottili](../extension/interpretazione.md) include una [revisione testuale di 40 casi](../extension/review40.md), che ha individuato errori NER e ambiguità. I dati originali sono conservati; l'alta contraddizione prodotta dall'NLI non valida da sola la correttezza delle perturbazioni. La revisione è dell'assistente, non annotazione umana.

L'esperimento è stato eseguito con i checkpoint reali MiniLM, RoBERTa-base e **DeBERTa-v3 base**, mantenendo il modello NLI richiesto. La negazione controllata lascia quasi invariata la similarità documentale; l'NLI della frase modificata rileva invece una netta perdita di supporto. **La media NLI del documento diluisce questo segnale**: il crollo locale non va presentato come crollo dell'intero documento.

## Campione e protocollo

Sono stati selezionati 200 testi macchina MGTBench: 67 Essay, 67 Reuters e 66 WritingPrompts, distribuiti sui sette generatori. Fra i 900 candidati, 538 consentivano tutte le perturbazioni. Questa selezione esclude testi senza un target grammaticale o un'entità idonei e limita la generalizzabilità. Ogni condizione ha 200 trasformazioni applicate, senza sostituzioni con identità.

Il run comprende 1.200 coppie controllate (identità e cinque trasformazioni), 400 controlli identità delle esatte frasi target, 7 fixture manuali separate e 800 coppie ufficiali OpAI: **2.407 coppie**, non 2.407 osservazioni indipendenti. IC 95% da 1.000 bootstrap per documento sorgente; in OpAI gli editor dello stesso originale sono raggruppati e ciascuna sorgente riceve uguale peso.

## Perturbazioni: documento intero

Medie su 200 sorgenti per condizione; NLI forward = originale → modificato, reverse = modificato → originale.

| Condizione | ROUGE-L | BLEU | Coseno MiniLM | BERTScore chunk | NLI forward | NLI reverse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Identità | 1,000 | 1,000 | 1,000 | 1,000 | 0,992 | 0,992 |
| Negazione | 0,996 | 0,985 | 0,998 | 0,999 | 0,894 | 0,898 |
| Sostituzione entità | 0,984 | 0,968 | 0,969 | 0,996 | 0,798 | 0,813 |
| Shuffle frasi | 0,492 | 0,968 | 0,999 | 0,999 | 0,989 | 0,987 |
| Troncamento | 0,781 | 0,576 | 0,958 | 0,979 | 0,992 | 0,669 |
| Rimozione casuale parole | 0,921 | 0,693 | 0,982 | 0,962 | 0,929 | 0,534 |

Per i documenti BERTScore è un'approssimazione esplicita con matching e aggregazione di chunk, non BERTScore standard sull'intero testo. Il coseno usa vettori di frasi/chunk aggregati con pesi per token. L'NLI recupera fino a tre evidenze per ipotesi e aggrega il massimo supporto; non certifica entailment logico del documento.

Lo shuffle conserva le proposizioni: il supporto NLI elevato è coerente con il protocollo e ne mostra il limite rispetto all'ordine narrativo. ROUGE-L è molto più sensibile all'ordine globale. Troncamento e rimozione mostrano una perdita soprattutto nella direzione che richiede l'informazione eliminata.

## Perturbazioni: frase modificata

| Condizione | Coseno locale | BERTScore locale | NLI entailment forward (IC 95%) | NLI contradiction forward |
| --- | ---: | ---: | ---: | ---: |
| Negazione | 0,8688 | 0,9843 | 0,00113 [0,00099; 0,00129] | 0,9976 |
| Sostituzione entità | 0,8476 | 0,9817 | 0,01354 [0,00422; 0,02677] | 0,8372 |

Il controllo identità delle stesse frasi ha entailment 0,994 per i target di negazione e 0,992 per quelli di entità. I delta appaiati con IC sono in [paired_identity_deltas.csv](paired_identity_deltas.csv). Tutti i 200 target di negazione usano BERTScore standard; per le entità 197 sono standard e 3 richiedono l'approssimazione a chunk, identificata per riga in `bertscore.csv`.

Il coseno locale diminuisce: non è completamente insensibile alla modifica. BERTScore raw resta alto, ma le scale non sono calibrate fra metriche e non autorizzano soglie universali; anche la fixture non correlata ha BERTScore circa 0,865. La negazione è una contraddizione intenzionale, non una gold label annotata per ogni caso. Un'entità sostituita può produrre neutralità o contenuto non supportato: qui la neutralità locale media è 0,149, e il basso entailment non dimostra falsità fattuale.

Sul documento, il minimo supporto medio scende da 0,983 dell'identità a 0,064 con negazione e 0,018 con entity swap. È sensibile agli errori locali, ma può essere fragile rispetto a una sola frase difficile: va considerato insieme a media, q10, quota senza supporto e controlli, senza sceglierne una soglia da questi soli risultati.

## OpAI-Bench: direzione dell'informazione

Confronti principali adiacenti, 200 coppie per operazione; forward e reverse seguono la stessa definizione precedente.

| Operazione | Transizione | Sorgenti distinte | NLI forward | NLI reverse | Forward − reverse, IC 95% |
| --- | --- | ---: | ---: | ---: | ---: |
| Compress | v3 → v4 | 190 | 0,9864 | 0,9197 | +0,0668 [0,0559; 0,0793] |
| Expand | v4 → v5 | 188 | 0,6798 | 0,9857 | −0,3059 [−0,3318; −0,2806] |

L'asimmetria media è nella direzione attesa: l'originale supporta la compressione più del contrario; l'espansione supporta il precedente più del contrario. In Compress il reverse resta elevato, quindi non si osserva un'implicazione strettamente unidirezionale per tutti i casi. Le operazioni nominali non sono gold label: 197/200 compressioni sono effettivamente più corte; 194/200 espansioni sono più lunghe. Tutti i casi, anche immutati, restano nell'analisi principale. Un'espansione che esplicita soltanto contenuto già implicito può essere supportata in entrambe le direzioni.

Il confronto secondario con v0 umano è cumulativo: include anche modifiche precedenti. I delta sono +0,0588 per Compress (IC [0,0466; 0,0726]) e −0,1736 per Expand (IC [−0,2007; −0,1485]); non vanno attribuiti alla sola ultima operazione. Fonte e SHA-256 dei 16 Parquet ufficiali sono in [opai_manifest.json](opai_manifest.json).

## Verifiche e riproducibilità

Sono stati calcolati **88.900 confronti NLI unici**. Il checkpoint base è stato eseguito float32 su CPU, anche tramite esportazione ONNX senza quantizzazione. Su 128 confronti reali il massimo scarto di probabilità rispetto al riferimento PyTorch indipendente è 0,00000560, con concordanza completa delle classi e della soglia entailment 0,5. Su 32 confronti reali locali, BERTScore differisce dalla libreria standard al massimo di 0,000000298. Questi controlli verificano l'implementazione numerica, non l'accuratezza semantica sull'intero corpus.

Il massimo RSS osservato del processo NLI è 2.636,73 MiB (2,58 GiB); BERTScore 1.631,91 MiB, embedding 699,88 MiB. Gli stadi girano in processi separati. Il tempo registrato del percorso NLI è circa 94 minuti, esclusi profiling, download ed export. I 56 test software passano; `pip check` non rileva dipendenze incompatibili. [validation.json](validation.json) verifica copertura completa dei punteggi, probabilità, cache e conservazione di tutti i caratteri non whitespace nella segmentazione per il retrieval.

[Report completo con IC e grafici](report.md), [punteggi per coppia](scores.csv), [riepilogo](metric_summary.csv), [asimmetria OpAI](opai_asymmetry.csv), [revisioni dei modelli](models_provenance.json). I comandi sono in [SEMANTIC_EXPERIMENT.md](../../../docs/protocols/SEMANTIC_EXPERIMENT.md). I testi e i pesi restano in `data/`, esclusi dal pacchetto dei risultati.

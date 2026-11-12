# Controlli positivi, perturbazioni sottili e stile OpAI

Questa estensione confronta le alterazioni intenzionali con riscritture presumibilmente conservative. Le etichette operative OpAI e le regole di perturbazione sono **proxy semantici, non gold label umane**. La revisione qualitativa viene svolta leggendo i testi, indipendentemente dai punteggi delle singole coppie; è una revisione dell'assistente e viene dichiarata come tale.

## Controlli positivi e confronto appaiato

I 16 Parquet ufficiali già scaricati contengono tutte le versioni. Il precedente `records.jsonl` proiettava soltanto v0/v3/v4/v5; il nuovo loader legge i Parquet locali senza un altro download. Si selezionano 100 traiettorie complete, bilanciate per dominio ed editor: 100 Polish (v0→v1), 100 Paraphrase (v1→v2) e 100 Style Rewrite (v2→v3).

Il confronto principale evita di separare soltanto OpAI da MGTBench: per ciascun positivo prima→dopo si prova una corruzione del medesimo testo dopo. Il negativo è prima→dopo-corrotto. Lo stesso originale, la stessa operazione, lo stesso editor e la stessa sorgente compaiono così nelle due classi. Le trasformazioni inapplicabili sono saltate esplicitamente; nessuna quota è riempita con identità. Le coppie immutate del benchmark restano nel campione positivo e sono contate.

I negativi MGTBench precedenti e le nuove perturbazioni sottili restano confronti secondari, con il limite esplicito del corpus diverso dai positivi. Shuffle e identità non entrano nell'AUC primaria del cambiamento di significato: lo shuffle può cambiare coerenza conservando le proposizioni.

## Perturbazioni sottili

`semantics/subtle_perturbations.py` aggiunge quattro tipi: numero quantitativo cambiato (esclusi date e ordinali), rafforzamento di un quantificatore, rimozione di una cautela epistemica, aggiunta di un orario specifico a un evento già asserito. Si conservano edit, offset, target locale, condizioni di applicabilità ed esclusioni. Un quantificatore rafforzato o una cautela eliminata possono essere non supportati senza essere contraddittori. L'orario aggiunto non va trattato come un fatto verificato.

## Numerosità delle sfide sottili

| Tipo | MGTBench | OpAI appaiato |
| --- | ---: | ---: |
| Numero cambiato | 50 | 50 |
| Quantificatore rafforzato | 50 | 43 |
| Cautela rimossa | 39 | 24 |
| Dettaglio aggiunto | 13 | 5 |

Le quote inferiori al massimo richiesto riflettono l'applicabilità conservativa, non edit sostituiti con identità. Nel test OpAI rimangono 24 numeri, 15 quantificatori, 9 cautele e un solo dettaglio: quest'ultimo è descrittivo e non consente una conclusione separata. Il confronto complessivo contiene inoltre 189 negazioni e 251 entity swap appaiati. Si affianca quindi al confronto complessivo un confronto secondario sui soli sottili, usando i positivi esattamente collegati e gli stessi split.

## Scelta dell'aggregatore

L'unità di split è la sorgente originale, condivisa da editor, versioni e perturbazioni. I punteggi non partecipano alla selezione delle sorgenti. Il confronto valuta media, q10, minimo e quota non supportata in ciascuna direzione e in forma bidirezionale. Per la conservazione bidirezionale si usa il supporto minore fra forward e reverse; per la quota non supportata si usa la maggiore.

L'aggregatore viene scelto sulla calibrazione; l'AUC successiva è calcolata sul test separato. Si riportano anche tutti gli aggregatori per trasparenza, senza sceglierne uno guardando il test. Una soglia fissata sui positivi di calibrazione mirando al 10% di falsi allarmi viene valutata sul test. Si tratta di falsi allarmi rispetto alle **etichette presunte**, non su parafrasi certificate. IC bootstrap e pesi usano la sorgente, non le molte varianti come repliche indipendenti. Gli score hanno orientamento fissato; le AUC sotto 0,5 non vengono ribaltate.

Forward basso è compatibile con aggiunte non supportate o modifiche; reverse basso con omissioni o modifiche. Nessuna direzione identifica da sola la causa di una divergenza, perché entrano in gioco anche retrieval e limiti del modello.

## BERTScore riscalato

La baseline ufficiale inglese di RoBERTa-base, layer 10, ha F=0,814463. Lo script applica esattamente la trasformazione della libreria con `rescale_with_baseline=True`: `(raw − baseline)/(1 − baseline)`, separatamente per P, R e F. La trasformazione dei punteggi già calcolati è equivalente a ripetere l'inferenza con quel flag: non occorre ricalcolare i vettori. Si esportano sia il valore riscalato sia `raw − baseline`.

Per i documenti il punteggio rimane l'approssimazione con matching di chunk; il rescaling non lo trasforma in BERTScore standard del documento intero. Essendo affine crescente, il rescaling non cambia ranking, correlazione Spearman o AUC dello stesso score. La baseline non è una soglia universale di significato conservato.

## Esecuzione

Da `/workspace/Humanization`, nell'ambiente già installato:

```bash
.venv/bin/python scripts/prepare_subtle_perturbations.py
.venv/bin/python scripts/prepare_semantic_extension.py
```

La preparazione rifiuta di sovrascrivere un campione già consolidato. Per riutilizzare le probabilità NLI e i riferimenti già calcolati, copiare una sola volta i due file nella nuova directory dati, preservando il checkpoint:

```bash
cp -n data/semantic_full/nli_probability_cache.jsonl data/semantic_extension/
cp -n data/semantic_full/nli_pytorch_reference.jsonl data/semantic_extension/
bash scripts/run_semantic_controls.sh
```

Gli stadi neurali sono sequenziali e mantengono DeBERTa-v3 base float32. I 1.607 controlli precedenti sono riusati solo dopo aver verificato che i testi dietro i loro ID siano invariati. Tutti i confronti nuovi vengono realmente inferiti.

Per la sola analisi stilometrica OpAI:

```bash
.venv/bin/python scripts/run_opai_style.py
```

Usa 200 traiettorie comuni alle cinque operazioni, le 105 feature originali e differenze dopo−prima. Include un controllo su prefissi a fine frase con conteggio lessicale identico. Effetti, IC e test usano la sorgente originale come unità; una variazione delle feature non dimostra naturalezza o umanizzazione.

## Risultati

- `results_semantic_extension/aggregator_report.md`: AUC, falsi allarmi e scelta dell'aggregatore.
- `results_semantic_extension/auc_by_stratum.csv`: dettagli per perturbazione, dominio, editor e operazione positiva.
- `results_semantic_extension/review40.md`: giudizi testuali sui 40 casi, inclusi errori e ambiguità.
- `results_semantic_extension/positive_review9.md`: piccolo audit dell'assunzione di conservazione nei controlli OpAI.
- `results_semantic_extension/metric_spearman.csv`: correlazioni fra metriche, incluso BLEU.
- `results_opai_style/report.md`: movimenti delle feature nei cinque stadi, documenti interi e controllo di lunghezza.

I testi e i pesi restano in `data/`, esclusi dai pacchetti dei risultati. Le analisi precedenti sono conservate.

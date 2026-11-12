# Verifica della rappresentazione stilometrica compatta

Specifica fissata l'11 novembre 2026, prima dei nuovi fit. L'analisi segue una precedente esplorazione degli stessi dati: non è una conferma indipendente della shortlist storica. I parametri eseguibili e l'hash del campione sono in [protocol.json](../../results/02_stylometry/shortlist/protocol.json).

## Domanda

Quanto segnale si conserva usando quattro caratteristiche invece del pannello completo? Una selezione fatta soltanto sui dati di addestramento funziona meglio di quattro caratteristiche casuali? Quali caratteristiche ricorrono e quanto trasferiscono a domini o generatori esclusi?

## Campione e modelli

Si riutilizzano esclusivamente le feature già salvate in `results_length_matched/features.csv`: 1.800 testi, 900 abbinamenti con conteggio lessicale identico, 300 per dominio. Le associazioni umano/macchina non garantiscono stesso argomento o derivazione per parafrasi. I gruppi del campione congelato sono 900; ogni gruppo contiene un umano e una macchina. Nessuna estrazione aggiuntiva, generazione o inferenza neurale.

| Configurazione | Definizione |
|---|---|
| C1 Full | Tutte le 105 feature |
| C2 Fixed | `pos_num`, `pos_cconj`, `mattr_50`, `pos_propn`, già scelte sull'intero campione; confronto esplorativo |
| C3 Training | Quattro feature selezionate esclusivamente sul training di ciascuna partizione |
| C4 Random | Media e distribuzione di 100 quaterne casuali distinte, estratte senza consultare etichette o valori |
| C4 Random matched | 100 quaterne distinte con tre POS e una feature lessicale, come composizione della shortlist |

Si aggiungono tre riferimenti: pannello da 100 feature senza i cinque indicatori diagnostici sensibili alla lunghezza, pannello stilometrico storico da 93 feature e solo numero di parole. Le etichette “100” e “93” si riferiscono alle colonne disponibili, non a una nuova selezione supervisionata. Gli AUC precedenti non sono copiati: anche questi riferimenti vengono valutati sulle nuove partizioni comuni.

Tutti i modelli usano imputazione con mediana, `StandardScaler` e regressione logistica L2, `C=1`, `class_weight=balanced`, `solver=lbfgs`, `tol=1e-4`, `max_iter=2000`. Imputazione e scale sono apprese sul solo training. Un'eventuale mancata convergenza viene riportata, senza scegliere nuovi parametri in base agli esiti.

## Selezione interna al training

C3 ordina per valore assoluto del g di Hedges calcolato sui soli valori finiti di training. Sono ammissibili feature con almeno due osservazioni finite per classe e variabilità comune positiva. Gli spareggi seguono il nome della feature. Si accettano progressivamente quattro colonne evitando correlazioni assolute di Spearman almeno 0,85 con quelle già scelte; la correlazione usa valori imputati con mediane del solo training. Se non si ottengono quattro colonne il programma si ferma, senza rilassare la regola.

Questa è una nuova regola semplice: non replica i filtri storici di coerenza fra domini e generatori. Non ottimizza soglie o iperparametri e non richiede una seconda cross-validation per sceglierli. Il nome C3 indica selezione interna a ciascun training esterno, non una ricerca del migliore algoritmo.

## Partizioni e riferimenti casuali

- Cross-validation aggregata sui tre domini: `StratifiedGroupKFold`, cinque fold e cinque semi fissi, 42, 137, 271, 811 e 2026. Prima dello split i gruppi sono rinominati con una permutazione determinata dal seme: è necessario perché, con un umano e una macchina in ogni gruppo, il solo shuffle di scikit-learn 1.7.2 produceva le stesse partizioni. Le configurazioni condividono tutte le partizioni. Le AUC sono riportate anche separatamente per dominio.
- Leave-one-domain-out: tre confronti, training su due domini e test sul terzo. Ogni gruppo con un testo nel test è rimosso interamente dal training.
- Leave-one-generator-out: sette confronti, training sugli altri sei generatori e test sul settimo e sugli umani associati. Nessun gruppo del test né esempio del generatore escluso entra nel training.
- Le quaterne casuali sono fissate una volta con seme 20261111, senza usare le etichette, e riutilizzate in ogni confronto. Le due distribuzioni sono mostrate integralmente, senza scegliere le migliori. C4 matched controlla la prevalenza delle parole funzionali nel pannello completo.

## Risultati e incertezza

La misura principale è la media delle AUC dei domini; per la cross-validation si mediano le AUC delle cinque ripetizioni. Il test aggregato è secondario. Nel leave-generator-out si riporta anche la media delle sette AUC. La media dei riferimenti casuali riguarda le AUC delle quaterne, non le probabilità: non si crea un ensemble.

Gli intervalli al 95% e le differenze appaiate usano 2.000 ricampionamenti dei gruppi, stratificati per dominio o, nel leave-generator-out, per generatore. Il seme è 20261112. Ogni estrazione è identica per le configurazioni e tiene insieme tutte le ripetizioni dello stesso testo. Gli intervalli sono condizionati alle predizioni già ottenute: non ricostruiscono tutta l'incertezza dell'addestramento. I fold, i semi e le frequenze di selezione non sono campioni indipendenti. Gli intervalli sono puntuali ed esplorativi; non vengono usati come prove confermative corrette per tutti i confronti.

Si riportano C3 meno C1, C2 e ciascun C4; C2 meno C1 e ciascun C4, etichettando i confronti C2 come esplorativi. Le distribuzioni fra quaterne casuali sono distinte dagli intervalli bootstrap. Non si fissa un margine di equivalenza o non inferiorità: una differenza non significativa non dimostra prestazioni equivalenti.

La stabilità è la frequenza di selezione delle colonne e delle famiglie nelle 25 partizioni CV, tre partizioni di dominio e sette di generatore. Le feature correlate possono sostituirsi: non ricorrere non equivale automaticamente a essere inutile.

## Limiti delle conclusioni

C3 verifica una procedura con selezione confinata al training. C2 descrive una shortlist già scelta sui medesimi dati e richiede gruppi nuovi per una conferma indipendente. Né il matching della lunghezza né questa validazione eliminano le differenze di argomento: numerali e nomi propri possono rifletterle. La capacità di distinguere umano e macchina rimane distinta dall'utilità come misura del cambiamento stilistico o della qualità di una riscrittura.

## Esecuzione

```bash
.venv/bin/python scripts/run_shortlist_validation.py --protocol results_shortlist_validation/protocol.json --output results_shortlist_validation
.venv/bin/python scripts/analyze_shortlist_validation.py --results results_shortlist_validation
```

I dati e i risultati storici restano invariati. Le nuove predizioni, selezioni e tabelle sono conservate nella cartella dedicata.

Correzione di implementazione v1.1: un primo tentativo, con le cinque partizioni identiche, è stato escluso prima di calcolarne le AUC. Si è corretta soltanto la randomizzazione dei gruppi, senza cambiare feature, classificatori, quaterne casuali o criteri di valutazione. Il protocollo aggiornato è fissato prima dei fit finali; la traccia del tentativo è in `implementation_note.json`.

## Controllo aggiuntivo: 100 candidate anziché 105

Su richiesta dell'autore si aggiunge una verifica separata, specificata prima dei nuovi fit nel [protocollo dedicato](../../results/02_stylometry/shortlist_length_robustness/protocol.json). Non modifica i risultati precedenti. Si fittano soltanto 35 modelli C3 con 100 candidate, escludendo `n_words`, `n_characters`, `n_sentences`, `ttr`, `hapax_ratio`. La selezione di quattro colonne, la soglia assoluta di Spearman 0,85 e la regressione logistica restano identiche.

Le partizioni vengono lette da `results_shortlist_validation/split_membership.csv`, senza rigenerarle. Le predizioni C3 con 105 candidate sono riutilizzate esattamente dall'esperimento precedente. Imputazione, standardizzazione e selezione delle nuove quattro feature usano soltanto il training delle stesse 25 partizioni CV, tre di dominio e sette di generatore. Le quaterne casuali non vengono ricalcolate.

Si confrontano AUC e differenze appaiate C3-100 meno C3-105, con la stessa misura macro e 2.000 ricampionamenti per gruppo. I pesi bootstrap e gli intervalli del riferimento C3-105 devono riprodurre quelli precedenti. Si riportano frequenze di selezione di tutte le colonne, permanenza di MATTR e comparsa di nuove candidate.

Si tratta di un controllo dell'insieme candidato: non dimostra che le 100 misure rimaste siano tutte indipendenti dalla lunghezza. Una perdita di prestazione non prova da sola che il segnale precedente fosse spurio: per esempio, TTR misura anche diversità lessicale e i conteggi delle parole erano già pareggiati. Nessun margine di equivalenza è fissato e non si promuove automaticamente una nuova shortlist. Il controllo rimane distinto dalla valutazione della humanization.

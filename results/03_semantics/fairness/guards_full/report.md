# Guardie lessicali complementari al supporto NLI

Gli score sono continui. Le guardie segnalano cambiamenti lessicali di impegno, non verità fattuale o entailment logico. Inventari e normalizzazione sono stati fissati prima della valutazione. Positivi OpAI e negativi artificiali sono etichette proxy, non gold umane.

## Confronto sul test separato

I due classificatori L2 (C=1) usano soltanto la calibrazione del confronto OpAI appaiato: quattro guardie, oppure minimo NLI e quattro guardie. Standardizzazione e fitting sono cal-only, con pesi uguali per fonte entro ciascuna classe. Nessuna ricerca di iperparametri. Il confronto sui sottili trasferisce gli stessi modelli, senza rifitting.

| cohort | metric | n_preserved | n_altered | auc | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| opai_matched_primary | risk_nli_min | 141 | 265 | 0.958 | 0.933 | 0.980 |
| opai_matched_primary | risk_guards_logistic | 141 | 265 | 0.719 | 0.671 | 0.773 |
| opai_matched_primary | risk_nli_guards_logistic | 141 | 265 | 0.938 | 0.903 | 0.971 |
| opai_matched_subtle_secondary | risk_nli_min | 42 | 49 | 0.900 | 0.813 | 0.964 |
| opai_matched_subtle_secondary | risk_guards_logistic | 42 | 49 | 0.732 | 0.628 | 0.840 |
| opai_matched_subtle_secondary | risk_nli_guards_logistic | 42 | 49 | 0.869 | 0.799 | 0.944 |

## Cautele rimosse

| cohort | metric | n_altered | auc | ci_low | ci_high | secondary_sensitivity |
| --- | --- | --- | --- | --- | --- | --- |
| opai_matched_primary | risk_nli_min | 9 | 0.571 | 0.320 | 0.698 | 0.133 |
| opai_matched_primary | guard_hedge_change | 9 | 0.444 | 0.167 | 0.720 | 0.267 |
| opai_matched_primary | risk_guards_logistic | 9 | 0.496 | 0.273 | 0.704 | 0.133 |
| opai_matched_primary | risk_nli_guards_logistic | 9 | 0.511 | 0.302 | 0.729 | 0.133 |
| opai_matched_subtle_secondary | risk_nli_min | 9 | 0.571 | 0.320 | 0.698 | 0.133 |
| opai_matched_subtle_secondary | guard_hedge_change | 9 | 0.444 | 0.167 | 0.693 | 0.267 |
| opai_matched_subtle_secondary | risk_guards_logistic | 9 | 0.496 | 0.273 | 0.702 | 0.133 |
| opai_matched_subtle_secondary | risk_nli_guards_logistic | 9 | 0.511 | 0.276 | 0.716 | 0.133 |

Nove casi test e poche fonti rendono questo risultato esplorativo. Il rilevatore di cautela conta modali e parole epistemiche: può cogliere la regola sintetica senza generalizzare a forme indirette. La validazione sulle stesse famiglie di regole non costituisce un test indipendente sugli humanizer.

L’audit descrittivo dei conteggi mostra che in 4/9 negativi hedge test la corruzione riporta il numero di cautele più vicino a quello dell’originale: l’edit OpAI positivo aveva già modificato quel numero. Un conteggio documentale può quindi cancellare modifiche in punti diversi, oppure penalizzare di più la riscrittura presunta buona. In 2/9 casi si elimina soltanto possibly da could possibly, lasciando could: non è una gold prova di passaggio dalla possibilità all’asserzione. Sono osservazioni diagnostiche post-score sui testi, non una nuova etichettatura umana; nessun caso viene rimosso né le regole vengono cambiate per migliorare il test. Il confronto incrementale output-buono→output-corrotto conferma l’applicazione della regola, ma richiederebbe un riferimento buono non disponibile nell’uso reale X→humanizer(X). Dettagli per ID in `hedge_count_audit.csv`.

## Diagnostica sulle rimozioni dirette MGTBench

Un controllo secondario conserva i 39 casi original-machine→corruzione senza un precedente edit OpAI. L’AUC usa i positivi OpAI e quindi rimane confusa dal corpus; non è una nuova stima indipendente. Nessun modello viene rifittato su questo sottocampione. `direct_hedge_diagnostic.csv` riporta AUC e numerosità per split; `hedge_loss_alerts.csv` confronta qualsiasi perdita di hedge con la soglia secondaria cal-only. La copertura dell’edit conta la regola applicata, non la correttezza semantica.

Copertura lessicale dei 39 edit: 39/39 presentano un calo nel conteggio di hedge. Il lessico v1 è incompleto: per esempio non include maybe e can; l’uguaglianza di conteggi non garantisce uguaglianza della modalità o della proposizione. I conteggi aggregati non dimostrano che una corruzione abbia ripristinato la cautela della stessa frase.

## Allarmi sui positivi presunti

Le soglie sono soltanto diagnostiche, calibrate sui positivi di calibrazione al 10% nominale. Non vengono proposte come soglie universali o come risultato da trasferire a DIPPER. Le probabilità della regressione sono score del classificatore proxy, non probabilità di significato conservato.

| operation | metric | threshold | n_alerts | n_presumed_preserved | false_positive_rate |
| --- | --- | --- | --- | --- | --- |
| all | risk_nli_min | 0.959 | 11 | 153 | 0.072 |
| all | risk_guards_logistic | 0.635 | 18 | 153 | 0.118 |
| all | risk_nli_guards_logistic | 0.751 | 12 | 153 | 0.078 |

## Interpretazione delle guardie

- Hedges: differenza assoluta nel conteggio totale, con modali may/might/could solo se POS/tag MD; May mese non conta. Si conservano perdita e guadagno separatamente. Modalità deontica/abilità e verbi epistemici restano ambigui.
- Negazioni: cambiamento del multiset canonico; not e n’t sono normalizzati. Non ricostruisce scope o doppia negazione.
- Numeri: cambiamento del multiset dei valori normalizzati e delle unità semplici. 20→30 viene rilevato anche a conteggio invariato; 3/three, 1,000/1000 e twenty percent/20% sono normalizzati nel sottoinsieme supportato. Nessun ragionamento aritmetico generale.
- Quantificatori: multiset di categorie lessicali fissate, con all/every/each nella stessa categoria universale. Non certifica la relazione logica fra quantificazioni.

File: `guard_scores.csv` (conteggi e delta per tutte le coppie), `guard_model_scores.csv` (score continui), `metric_auc.csv`, `auc_by_condition.csv`, `positive_all_alerts.csv`, `model_manifest.json`. I testi completi non sono esportati nei CSV.

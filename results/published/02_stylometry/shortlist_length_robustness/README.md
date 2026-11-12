# C3: robustezza con 100 feature candidate

Controllo aggiuntivo dell'11 novembre 2026. Entrambi i classificatori usano **quattro feature**: cambia soltanto l'insieme da cui selezionarle, da 105 a 100. Si escludono `n_words`, `n_characters`, `n_sentences`, `ttr`, `hapax_ratio`; non si sostiene che le restanti misure siano tutte indipendenti dalla lunghezza.

- **[Report Markdown](report.md)**, **[PDF](report.pdf)** e **[ZIP](https://github.com/FrancescoM00/Humanization/blob/d2868b323938a2a9371c2a872ad86b5596d0f097/results_shortlist_length_robustness/report_robustezza.zip)**.
- [Protocollo fissato prima dei nuovi fit](protocol.json), [specifica generale](../../../docs/protocols/SHORTLIST_VALIDATION.md), [esperimento precedente](../shortlist/README.md).
- [AUC e intervalli](metrics.csv), [differenze appaiate](paired_differences.csv).
- [Frequenze complete](selection_frequency.csv), [confronto delle selezioni](selection_comparison.csv), [famiglie](selection_family_frequency.csv), [selezioni di ogni training](selections.csv).
- [Coefficienti e scale di training](coefficients.csv), [audit degli split](split_audit.csv), [manifest fit](manifest.json), [manifest statistico](statistics_manifest.json), [riepilogo](summary.json).

`predictions.npz` contiene gli array `cv` (5,1800,2), `leave_domain_out` e `leave_generator_out` (1,1800,2). Le colonne sono C3_105 e C3_100, nell'ordine di `model_registry.csv`; i testi seguono `documents.csv`. La prima colonna è identica alle predizioni C3 precedenti, senza refit. `bootstrap_auc.npz` contiene le AUC dei ricampionamenti, con lo stesso ordine di configurazioni.

Sono effettuati soltanto **35 nuovi fit**. I gruppi e le partizioni sono letti dal precedente `split_membership.csv`, senza rigenerarli. Il bootstrap riproduce le AUC e gli intervalli del riferimento precedente. Lo ZIP include i risultati del nuovo controllo; per rieseguire l'analisi serve anche l'esperimento precedente nel repository.

## Riproduzione

Dalla radice del progetto:

```bash
.venv/bin/python scripts/run_shortlist_length_robustness.py --protocol results_shortlist_length_robustness/protocol.json --output results_shortlist_length_robustness_replica
.venv/bin/python scripts/analyze_shortlist_length_robustness.py --results results_shortlist_length_robustness_replica
```

Il runner copia il protocollo nel nuovo output e rifiuta di sovrascrivere un run completato. Per rigenerare soltanto tabelle, figure e Markdown usare il secondo comando sul risultato originale: non ripete i fit.

Il PDF è un'esportazione con Pandoc e XeLaTeX:

```bash
cd results_shortlist_length_robustness
pandoc report.md --standalone --pdf-engine=xelatex --include-in-header=pdf_header.tex -V papersize=a4 -V geometry:margin=2cm -V fontsize=10pt -V mainfont='Latin Modern Roman' -V monofont='Latin Modern Mono' -V lang=it-IT --resource-path=. --output=report.pdf
```

Il controllo riguarda la robustezza della selezione, non una nuova misura di humanization o una conferma indipendente di una shortlist fissa.

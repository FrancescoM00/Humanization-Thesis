# Verifica della shortlist: materiali

Esperimento completato l'11 novembre 2026 su 1.800 testi già rappresentati da feature. Il confronto non richiede nuovi modelli linguistici o GPU.

- **[Report leggibile](report.md)** e **[PDF](report.pdf)**.
- **[ZIP leggero](https://github.com/FrancescoM00/Humanization/blob/d2868b323938a2a9371c2a872ad86b5596d0f097/results_shortlist_validation/report_shortlist.zip)** con report, grafici, tabelle e specifica.
- [Protocollo eseguibile](protocol.json) e [spiegazione del metodo](../../../docs/protocols/SHORTLIST_VALIDATION.md).
- [AUC e intervalli](metrics.csv), [differenze appaiate](paired_differences.csv).
- [Frequenze delle 105 feature](selection_frequency.csv), [famiglie](selection_family_frequency.csv), [selezioni per training](selections.csv).
- [Quaterne casuali](random_subsets.csv), [loro AUC](random_subset_metrics.csv), [distribuzioni](random_reference_summary.csv).
- [Coefficienti C2/C3 e scale di training](coefficients.csv).
- [Identificativi dei testi](documents.csv), [registro modelli](model_registry.csv), [audit split](split_audit.csv), [appartenenza ai gruppi](split_membership.csv).
- `predictions.npz`: probabilità complete, array `cv` (5,1800,206), `leave_domain_out` e `leave_generator_out` (1,1800,206). L'ordine dei testi è `documents.csv`, quello dei modelli è `model_registry.csv`.
- `bootstrap_auc.npz`: AUC ricampionate; l'ordine delle configurazioni è in [summary.json](summary.json).
- [Manifest del fit](manifest.json), [manifest statistico](statistics_manifest.json), [nota sul tentativo escluso](implementation_note.json).

**C2 non è una conferma indipendente delle quattro colonne:** erano già state selezionate sull'intero campione. C3 valuta una nuova regola limitata al training. Gli abbinamenti umano/macchina controllano le parole, non garantiscono stesso contenuto. Questi limiti valgono anche per il trasferimento a domini e generatori esclusi.

## Riproduzione

Dalla radice del repository, con le dipendenze Python già installate:

```bash
.venv/bin/python scripts/run_shortlist_validation.py --protocol results_shortlist_validation/protocol.json --output results_shortlist_validation_replica
cp results_shortlist_validation/protocol.json results_shortlist_validation_replica/protocol.json
.venv/bin/python scripts/analyze_shortlist_validation.py --results results_shortlist_validation_replica
```

Il fit rifiuta di sovrascrivere un run già completato. Il CSV in ingresso è quello storico, verificato con l'hash registrato. Le predizioni e le tabelle complete sono versionate nel repository; il solo ZIP leggero non contiene l'input delle feature né le due matrici NPZ.

Per rigenerare soltanto tabelle, grafici e Markdown del run esistente, usare il secondo script con `--results results_shortlist_validation`. Non ripete i fit. Il PDF è un'esportazione del Markdown con Pandoc e XeLaTeX, non una dipendenza dell'esperimento:

```bash
cd results_shortlist_validation
pandoc report.md --standalone --pdf-engine=xelatex --include-in-header=pdf_header.tex -V papersize=a4 -V geometry:margin=2cm -V fontsize=10pt -V mainfont='Latin Modern Roman' -V monofont='Latin Modern Mono' -V lang=it-IT --resource-path=. --output=report.pdf
```

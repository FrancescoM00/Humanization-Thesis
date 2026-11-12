# 02 — Componente stilometrica e selezione delle feature

**Riferimento: §1.3 del PDF.** Si verifica quali caratteristiche separino umano e macchina e quanto il segnale cambi fra domini e generatori. È una valutazione discriminativa, non una misura di qualità di uno humanizer.

## Metodo e risultati da leggere

[`src/stylometry/features.py`](../../src/stylometry/features.py) estrae 105 feature con spaCy piccolo, textstat e MATTR50: frasi, punteggiatura, POS, parole funzionali, diversità lessicale, leggibilità e proxy della guida Wikipedia. Le definizioni sono nel `feature_schema.json` dei risultati.

Sul campione a parole abbinate: MATTR50 umano/macchina 0,810/0,777, g = −0,684; punteggiatura 16,329/13,379 ogni cento parole, g = −0,542. La regressione logistica stilometrica iniziale raggiunge AUC entro dominio 0,876, 0,928 e 0,881; escludendo il dominio dal training scende a 0,670, 0,759 e 0,580. La shortlist storica `pos_num`, `pos_cconj`, `mattr_50`, `pos_propn` rimane esplorativa.

| Verifica successiva | AUC CV macro |
| --- | ---: |
| C1: tutte le 105 feature | 0,883 |
| C2: quattro feature storiche | 0,760 |
| C3: quattro selezionate sul solo training | 0,783 |
| Quattro casuali uniformi / per famiglie | 0,620 / 0,719 |
| C3: quattro selezionate da 100 candidate | 0,763 |

Nei due C3 entrano sempre quattro feature: 105 e 100 indicano il pannello candidato. Il secondo esclude `n_words`, `n_characters`, `n_sentences`, `ttr`, `hapax_ratio`. MATTR ricorre in tutti i 35 training. La prima regressione stilometrica e C1 non usano esattamente lo stesso pannello: i numeri dei due protocolli vanno tenuti distinti.

## Consultare e ricalcolare senza estrarre i testi

I risultati congelati sono in [`results/02_stylometry`](../../results/02_stylometry): `baseline`, `length_matched`, `shortlist`, `shortlist_length_robustness`. Comprendono feature numeriche, predizioni, partizioni, intervalli e report.

Per rigenerare le tabelle della shortlist dalle predizioni salvate, lavorando su una copia:

```bash
python -c "from shutil import copytree; copytree('results/02_stylometry/shortlist', 'runs/shortlist_analysis', dirs_exist_ok=True)"
python scripts/analyze_shortlist_validation.py --results runs/shortlist_analysis
python -c "from shutil import copytree; copytree('results/02_stylometry/shortlist_length_robustness', 'runs/shortlist_100_analysis', dirs_exist_ok=True)"
python scripts/analyze_shortlist_length_robustness.py --results runs/shortlist_100_analysis
```

Queste analisi usano le predizioni già calcolate; non addestrano nuovi modelli. Gli intervalli ricampionano i gruppi originali, mantenendo insieme le versioni e le ripetizioni.

## Ripetere estrazione e classificazione

Preparare prima i CSV come in [01_dataset](../01_dataset/README.md), installare l'ambiente stilometrico e il modello spaCy. Il campione completo e il controllo di lunghezza si ricostruiscono con i due comandi della guida dataset. Per rifare le selezioni usando le feature numeriche congelate:

```bash
python scripts/run_shortlist_validation.py --protocol results/02_stylometry/shortlist/protocol.json --output runs/shortlist_refit
python scripts/run_shortlist_length_robustness.py --protocol results/02_stylometry/shortlist_length_robustness/protocol.json --output runs/shortlist_100_refit
```

C3 ordina le feature per |g| di Hedges e scarta correlazioni assolute di Spearman almeno 0,85, sempre sul training. Imputazione e standardizzazione usano soltanto il training. Regressione L2, C=1, classi bilanciate, nessuna ricerca degli iperparametri. Il controllo a 100 candidate riusa le partizioni e il riferimento C3 a 105 congelati: esegue 35 nuovi fit.

Tutto gira su CPU; nessuna generazione LLM o chiamata API. Gli script risolvono i percorsi storici dei protocolli verso le cartelle nuove, preservando la verifica degli input e gli hash originali. Gli output nuovi restano sotto `runs/`.

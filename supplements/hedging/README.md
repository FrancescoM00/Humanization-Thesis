# Supplemento — Estensione sulle espressioni di cautela

Questo esperimento del **12 novembre 2026** è successivo al PDF usato per organizzare la consegna. Completa il precedente controllo sulle cautele; non sostituisce retroattivamente i risultati del §1.4.3.

## Domanda, dati e risultato

Si mantiene la soglia documentale storica esatta **0,040749866515398**, senza ricalibrarla sul nuovo test. La configurazione è MiniLM top-3 e DeBERTa-v3 base float32, con minimo bidirezionale documentale. Questa soglia non appartiene alla configurazione contestuale `window2` scelta successivamente.

Il pannello contiene 60 originali macchina (22 Essay, 16 Reuters, 22 WritingPrompts), con tre versioni ciascuno: rimozione completa della cautela, rafforzamento parziale e parafrasi conservativa. Sono **120 modifiche con intento di alterazione e 60 controlli**, non 180 fonti indipendenti. Le cautele erano già presenti negli originali.

| Condizione | AUC documentale | Segnalate alla soglia fissa |
| --- | ---: | ---: |
| Rimozione completa | 0,978 | 13/60 |
| Rafforzamento parziale | 0,733 | 0/60 |
| Complessivo | 0,856 | 13/120 |

L'AUC complessiva ha IC 95% [0,819; 0,891]; la sensibilità è 10,8%, IC [5,8%; 15,8%]. Si osservano 0/60 falsi allarmi sui controlli, con limite superiore Wilson circa 6,0%. L'NLI contiene un segnale, ma la soglia storica non copre adeguatamente queste modifiche, soprattutto quelle parziali.

Tutte le triplette e la riserva sono state lette dall'assistente prima dei punteggi. Quattro originali con errori chiari sono stati sostituiti; nel pannello finale 90 coppie sono giudicate valide e 90 qualificate per ambiguità. Non sono annotazioni umane. Le alterazioni chiaramente valide sono 63/120; con controlli validi abbinati rimangono 25 fonti e 49 alterazioni, AUC 0,844 e sensibilità ponderata 16,0%.

## File e analisi riproducibile

I materiali finali sono in [`results/`](results/): `pairs.jsonl`, `assistant_review.csv`, `nli.csv`, `metrics.csv`, report PDF/Markdown, manifest e registrazione delle sostituzioni. La cartella `pre_review/` conserva i dati prima della revisione; non è un altro pannello di test.

Per rigenerare l'analisi senza inferenze, copiare prima i metadati necessari al report:

```bash
python -c "from pathlib import Path; from shutil import copy2; p=Path('runs/hedging_analysis'); p.mkdir(parents=True, exist_ok=True); [copy2(Path('supplements/hedging/results')/n,p/n) for n in ['preparation_manifest.json','discussione.md','replacements.csv']]"
python scripts/analyze_hedging_controls.py --pairs supplements/hedging/results/pairs.jsonl --scores supplements/hedging/results/nli.csv --protocol supplements/hedging/results/protocol.json --review supplements/hedging/results/assistant_review.csv --output runs/hedging_analysis
```

Si ottengono AUC, sensibilità, falsi allarmi, 2.000 bootstrap per originale, analisi dei casi validi e diagnostica locale. Gli score non vengono usati per scegliere casi o una nuova soglia.

Il codice di costruzione è `src/semantics/hedging_controls.py`, con i CLI `prepare_hedging_controls.py` e `finalize_hedging_controls.py`. Quest'ultimo applica revisioni testuali già disponibili, non produce giudizi umani. Le nuove inferenze possono usare le coppie finali e gli stadi embedding/NLI descritti nella [guida semantica](../../experiments/03_conservazione_semantica/README.md); i confronti neurali originari sono 5.080. I modelli non sono richiesti per ricalcolare le tabelle dai punteggi inclusi.

Il [report completo](results/report.md) discute anche un limite direzionale: in molti casi il modello assegna più supporto dal testo cauto a quello certo che viceversa. La buona AUC bidirezionale non dimostra un ragionamento corretto sulla modalità. Le schede umane restano vuote; l'assenza di allarme non è una garanzia di fedeltà.

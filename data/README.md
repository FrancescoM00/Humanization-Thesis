# Dati del progetto

- `raw/mgtbench/`: i tre CSV originali forniti dall'autore, 1.000 righe ciascuno. Sono identici byte per byte ai dati degli esperimenti e non vengono risalvati durante la preparazione.
- `raw/opai/official_parquet/`: i 16 Parquet ufficiali OpAI utilizzati per gli esperimenti di editing.
- `resources/nltk_data/`: CMUdict con i file originali e le relative attribuzioni.
- `processed/`: coppie, coorti e score congelati per ricalcolare le analisi. La sottocartella `generated/` contiene eventuali nuovi campioni.
- `models/`: destinazione locale dei checkpoint, creata quando necessaria e esclusa dalla consegna.

Verifica dei CSV dalla radice del progetto:

```bash
python experiments/00_data.py --config configs/datasets.yaml --mode reproduce
```

`--mode compute` prepara anche un campione con gli stessi criteri di pulizia e campionamento. 

Per il flusso dei comandi vedere [reproducibility](../docs/reproducibility.md).

# Dati

`raw/` contiene i tre CSV MGTBench originali forniti dall'autore il 12 novembre 2026: `Essay_LLMs.csv`, `Reuters_LLMs.csv`, `WP_LLMs.csv`. Sono copie dei file indicati come ufficiali nella cartella Google Drive segnalata da TH-Bench. Ciascuno contiene 1.000 righe: un testo umano e gli output dei sette generatori, oltre ai metadati. Le versioni di una riga condividono un'origine; non si assume che siano parafrasi equivalenti.

Gli allegati coincidono byte per byte con i dati usati negli esperimenti già eseguiti. La prima acquisizione storica era avvenuta tramite un mirror GitHub fissato a un commit; il confronto successivo con le copie ufficiali è documentato in [provenance](../docs/provenance/mgtbench_drive_comparison.md). La nuova consegna usa direttamente le copie allegate, senza download MGTBench automatico.

```bash
python scripts/import_mgtbench.py --verify-only
python scripts/import_mgtbench.py --input-dir /percorso/copie_ufficiali --output data/raw
```

Il comando verifica nomi, colonne, numero di righe e SHA-256. File già identici sono mantenuti; un file diverso causa un errore prima delle copie. I valori attesi sono quelli degli esperimenti pubblicati, non di eventuali versioni future del dataset. `SOURCE.json` registra provenienza e hash. Il controllo usa soltanto la libreria standard Python e non richiede GPU o modelli.

I CSV non vengono risalvati o ripuliti: ordine delle righe, celle e codifica restano invariati. Pulizia, campionamento e abbinamento della lunghezza producono file separati durante gli esperimenti. Le directory generate sotto `data/`, inclusi i pesi in `data/models/`, sono ignorate da Git. Le coppie congelate necessarie alla rianalisi sono invece in [`inputs/`](../inputs/README.md).

OpAI-Bench è un dataset distinto e non è sostituito dai CSV MGTBench. Il relativo download è descritto negli esperimenti [03](../experiments/03_conservazione_semantica/README.md) e [04](../experiments/04_stile_opai/README.md). I termini dei corpora rimangono quelli delle fonti; vedere [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

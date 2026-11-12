# OpAI: dati originali

`official_parquet/` contiene i 16 file ufficiali utilizzati negli esperimenti: quattro domini e quattro generatori, con le versioni di editing e gli array di frasi e label nativi. `SOURCE.json` conserva la provenienza dell'acquisizione originale, inclusi i riferimenti al derivato JSONL storico. Quel derivato non è necessario per leggere direttamente i Parquet in questo progetto.

L'esperimento `03_opai_editing.py` legge questi file per il nuovo calcolo delle feature; in modalità `reproduce` usa invece le feature già estratte in `results/published/04_opai_style`. I dati originali non sono sostituiti da esempi sintetici.

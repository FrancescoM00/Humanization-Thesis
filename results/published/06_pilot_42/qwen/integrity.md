# Integrità del pilot ricevuto

Verifica sui file registrati, senza nuove inferenze di humanizer, detector o metriche neurali.

- Lo ZIP originale è conservato in `exports/kaggle_pilot/humanization_pilot.zip`, SHA-256 `7a35ca521ccc4e2147b42e479fe2ac6bde1ef676ddb462137a27af046960f272`.
- `semantic/pairs.jsonl`, `rewriting_cuda0/pairs.jsonl` e `rewriting_cuda0/rewrites.jsonl` sono identici byte per byte: 84 righe, ID unici, stati tutti `ok`, 42 identità e 42 Qwen.
- La coorte contiene 42 fonti test, due per ciascuna delle 21 celle. I 300 umani di calibrazione hanno gruppi disgiunti dal test e dalle 126 fonti riservate allo studio. I file dei tre pannelli coincidono con quelli congelati nello smoke.
- Tutti i sei stadi semantici/allineati sono completi per gli stessi 84 ID; i manifest riportano il corretto hash delle coppie. I metadati BERTScore/NLI/allineamento corrispondono ai sorgenti corretti archiviati, i cui AST coincidono con il codice del progetto.
- Il runtime registra commit `71dd61596e259a9fda89cd7d1ef4909b84c5ee79`, Python 3.13 e PyTorch `2.11.0+cu130`, su T4. NLI è configurato come il medesimo checkpoint base float32 su CUDA; l'archivio non include i pesi per un nuovo confronto byte per byte dei modelli.
- I riepiloghi detector e macro sono stati ricalcolati con il codice del progetto e coincidono con quelli ricevuti entro 1e-10. Le soglie coincidono esattamente con lo smoke; gli umani di calibrazione sono gli stessi.
- Le tre riscritture dello smoke e le relative identità hanno gli stessi testi nel pilot. Il pilot le include e non costituisce una validazione indipendente da esse.
- Le identità hanno RMS/POS-JS/delta stilometrici zero, rapporto parole uno, similarità circa uno; NLI identità medio 0,9910, con medie individuali 0,9801–0,9961 e minimo fra unità 0,8666. Le probabilità non sono impostate a uno.
- Le colonne `local_*` mancanti sono coerenti con l'assenza di target locali annotati. Lo stato `ok` descrive il completamento tecnico, non la conservazione semantica.
- I form originali sono vuoti: 12 coppie semantiche e 28 testi di naturalezza. Nessuna annotazione umana o misura di naturalezza viene ricavata da questi campi.

Il contenuto dell'archivio non è stato riscritto. Le tabelle derivate e le osservazioni qualitative sono separate dai dati originali. Non è emerso un nuovo blocco tecnico della pipeline.

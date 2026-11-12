# Input congelati per la riproduzione

Questi file sono gli input degli esperimenti già eseguiti e i punteggi intermedi necessari a rigenerare le tabelle senza ripetere le inferenze. I CSV MGTBench originali sono separati in `data/raw/`.

| Cartella | Contenuto |
| --- | --- |
| `semantic_baseline/` | Coppie del primo esperimento semantico |
| `semantic_extension/` | Positivi OpAI, perturbazioni sottili, split e coppie appaiate |
| `semantic_clean/` | Pannello con sostituzioni di entità più conservative |
| `semantic_fairness/` | Errori distribuiti, positivi aggiuntivi e allineamenti |
| `pilot_42/qwen/`, `pilot_42/dipper/` | Coorti, riscritture e punteggi dei due pilot |
| `study_126/` | Coorti, riscritture, punteggi detector, NLI e stile dello studio esteso |
| `nli_calibration/` | Cinque input congelati della selezione NLI contestuale |

I file sono copiati dall'archivio del progetto originale. I manifest storici mantengono hash e percorsi validi per quel codice e quel momento dell'esecuzione. La nuova mappatura è in [`docs/migration_manifest.json`](../docs/migration_manifest.json); non si attribuisce retroattivamente il nuovo codice ai vecchi run.

Le condizioni generate e le parafrasi OpAI non sono etichette umane gold. Le annotazioni umane reference dello studio sono ancora in raccolta. I pesi dei modelli, le cache probabilistiche complete, gli ambienti virtuali e le risposte del giudice di naturalezza non sono inclusi. Le nuove esecuzioni vanno in `runs/`, lasciando invariati questi input e i risultati consegnati.

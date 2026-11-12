# Input e score congelati

Questa directory contiene gli input degli esperimenti già eseguiti, separati dai dati originali e dai nuovi risultati.

| Directory | Contenuto |
| --- | --- |
| `semantic_baseline`, `semantic_extension`, `semantic_clean`, `semantic_fairness` | Coppie, etichette intenzionali, split e allineamenti dei controlli semantici |
| `pilot_42/qwen`, `pilot_42/dipper` | Coorti, riscritture e score individuali dei pilot |
| `study_126` | Coorti, riscritture e score dello studio esteso |
| `nli_calibration` | Pannello per la scelta della configurazione NLI |
| `naturalness_reference` | Giudizi remoti già raccolti, per diagnostiche offline; non sono etichette umane |
| `generated` | Eventuali campioni nuovi, esclusi dalla consegna e dal versionamento |

L'esperimento `07_evaluation.py --mode reproduce` ricalcola metriche e confronti dai punteggi individuali. Gli output vanno in `results/new_runs`; i riferimenti consegnati restano in `results/published`. Le configurazioni YAML e i protocolli di `configs/protocols/` indicano gli input con percorsi relativi. I manifest di esecuzione registrano parametri e SHA-256 dei file effettivamente usati.

Le perturbazioni costruite e le review dell'assistente non sono validazione umana indipendente. I pesi e le cache neurali complete non sono inclusi.

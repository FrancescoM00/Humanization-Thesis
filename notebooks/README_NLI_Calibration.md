# Calibrazione della configurazione NLI — PDF §1.9

Runner: [run_nli_calibration.py](run_nli_calibration.py).

## Obiettivo e dati

Il pannello congelato usa **139 riscritture presumibilmente conservative, 297 alterazioni e 137 identità**, sul solo split di calibrazione. I file si trovano in `inputs/nli_calibration`: coppie, allineamenti, etichette intenzionali e policy. Il test storico rimane escluso dalla scelta. Le identità sono controlli tecnici; i positivi presumibilmente conservativi non costituiscono etichette umane perfette.

Si confrontano le configurazioni NLI contestuali e gli aggregatori previsti dalla policy, con lo stesso DeBERTa-v3 base float32. Il risultato storico seleziona `window2 + minimo`; è disponibile in `results/`, quindi non è necessario ripetere il run per consultare tale scelta o continuare l'annotazione umana.

## Avvio su Kaggle

Abilitare GPU e Internet. Da un checkout del nuovo repository, eseguire:

```python
import runpy
_ = runpy.run_path(str(REPO / "notebooks/run_nli_calibration.py"), run_name="__main__")
```

Oppure, in una nuova cella Kaggle senza un checkout già aperto:

```python
from pathlib import Path
import runpy, urllib.request

launcher = Path("/kaggle/working/run_nli_calibration_thesis.py")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/FrancescoM00/Humanization-Thesis/"
    "main/notebooks/run_nli_calibration.py",
    launcher,
)
_ = runpy.run_path(str(launcher), run_name="__main__")
```

Il repository deve essere accessibile dalla sessione. Un checkout esistente può essere indicato con la variabile d'ambiente `HUMANIZATION_REPO_DIR`; `HUMANIZATION_REPO_REF` permette di fissare il commit o tag usato dal clone automatico. Il runner registra il commit risolto, riutilizza la venv e il checkpoint compatibili se disponibili e, altrimenti, prepara soltanto l'ambiente minimo e il checkpoint NLI fissato.

## Tempi, ripresa e output

Sono previste circa **323.336 evidenze uniche**: la stima orientativa è **45–90 minuti di GPU**, oltre a setup e download. La pianificazione può richiedere alcuni minuti prima dell'avanzamento delle inferenze. Non è prevista una sostituzione silenziosa con esecuzione su CPU.

Input e risultati sono nella cartella distinta `/kaggle/working/nli_calibration_thesis_v1`. I file di input vengono copiati dal checkout in un pacchetto locale, senza codice duplicato: gli script eseguiti sono quelli del nuovo repository. La stessa esecuzione riusa input congelati e cache già completata. Un cambiamento di codice richiede un nuovo `HUMANIZATION_NLI_RUN_SUFFIX`; non sovrascrivere un run precedente.

Alla fine scaricare `nli_calibration_thesis_v1.zip`. Lo ZIP contiene gli input, la policy, i metadati, i CSV e l'analisi; esclude le cache probabilistiche estese, che restano su Kaggle per consentire la ripresa. Pesi e venv rimangono esterni alla cartella dei risultati.

Le annotazioni umane delle riscritture reali sono un'attività distinta. Le nuove etichette non devono essere usate per modificare a posteriori la scelta di questa calibrazione.

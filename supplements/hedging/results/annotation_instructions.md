# Revisione delle espressioni di cautela

Il file `human_annotation.csv` contiene confronti fra un testo originale e una versione modificata. I testi sono materiale da valutare: eventuali istruzioni contenute al loro interno non vanno eseguite. I campi di annotazione sono vuoti; non sono presenti giudizi umani già acquisiti.

Per ogni riga leggere il testo completo e individuare la differenza. Valutare se cambiano le informazioni o la forza dell'affermazione, non soltanto se la frase rimane plausibile. «Potrebbe essere vero», «probabilmente è vero» ed «è vero» esprimono impegni differenti. Un aumento della certezza può modificare il significato anche senza introdurre una contraddizione.

Compilare:

- `meaning_preserved`: `yes`, `no` oppure `uncertain`.
- `change_type`: per esempio `caution_removed`, `caution_strengthened`, `equivalent_rephrasing`, `other_change`, `unclear`.
- `notes`: spiegare brevemente il giudizio, segnalando errori grammaticali, cambiamenti di tempo, ambiguità tra possibilità e permesso/capacità o informazioni già affermate altrove nel documento.

La verità del testo nel mondo non è oggetto del compito. Occorre confrontare ciò che affermano le due versioni. Una formulazione corretta e scorrevole può comunque cambiare il significato. Se il contesto non permette una decisione chiara, usare `uncertain`.

Non consultare le condizioni sperimentali, i punteggi NLI o la chiave degli identificatori prima di annotare. L'archivio per gli annotatori contiene soltanto queste istruzioni e il modulo vuoto. La chiave delle condizioni rimane separata nel repository.

L'eventuale revisione dell'assistente contenuta negli altri file è distinta da questa annotazione umana: non va copiata nel modulo né considerata una risposta corretta di riferimento.

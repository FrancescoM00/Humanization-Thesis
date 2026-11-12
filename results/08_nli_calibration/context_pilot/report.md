# NLI con premesse più ampie: prova sulle fusioni e sui controlli

**La fusione di frasi spiega una parte sostanziale dei falsi allarmi osservati.** Nel racconto Maria/Sofia prodotto dalla configurazione DIPPER XXL sul passaggio intero, il forward medio sale da 0,0038 con singole premesse a 0,7994 con finestre fino a tre unità. L'intero documento raggiunge 0,9820, ma dà anche allarmi su controlli identici: non emerge un aggregatore universalmente valido da questa piccola prova.

Sono state eseguite realmente **5.446 inferenze uniche su 19 coppie**, con lo stesso DeBERTa-v3 base, CPU float32, batch 2 e due thread, in circa 26,4 minuti. Input, allineamenti, selezione e metadati sono salvati in questa directory. Non sono nuove annotazioni umane e non viene stimata AUC. I 19 casi sono un pannello esplicativo mirato, non un campione casuale di accuratezza.

## Disegno

Per gli output Kaggle le ipotesi sono le stesse frasi/chunk degli allineamenti precedenti. Si confrontano quattro modalità di premessa:

- `single_all`: ogni unità originale separata, con massimo supporto su tutte le unità ammissibili.
- `window2`: tutte le unità singole e tutte le coppie contigue, senza riordinamento.
- `window3`: le precedenti più tutte le triple contigue.
- `document`: il testo completo esatto come unica premessa, soltanto se la coppia premessa/ipotesi entra nei 384 token effettivi del tokenizer.

Nessun testo viene troncato per rientrare nel budget. Se manca la copertura completa delle ipotesi, l'aggregato della direzione rimane mancante; non si media soltanto sui casi facili. Il documento non sostituisce automaticamente le finestre quando non entra.

Le finestre 2/3 sono insiemi di evidenze annidati: il massimo supporto può soltanto aumentare aggiungendo candidati. **L'aumento è anche una proprietà della costruzione, non una prova di migliore accuratezza.** Per questo sono inclusi conservazioni presunte, identità, omissioni ed errori controllati.

Il precedente retrieval top-3 GPU rimane in [historical_top3.csv](historical_top3.csv). Il confronto controllato fra contesti usa invece le quattro nuove modalità sullo stesso backend CPU; non si attribuisce ogni differenza rispetto alla pipeline storica al solo contesto.

## Fusione Maria/Sofia: cambia il verdetto forward

| Premessa | Forward medio | Forward minimo | Reverse medio | Reverse minimo |
| --- | ---: | ---: | ---: | ---: |
| Singole, tutte | 0,0038 | 0,0017 | 0,8256 | 0,0124 |
| Fino a 2 unità | 0,3719 | 0,0063 | 0,8528 | 0,0243 |
| Fino a 3 unità | 0,7994 | 0,0474 | 0,8574 | 0,0277 |
| Documento esatto | 0,9820 | 0,9243 | 0,7486 | 0,0004 |

Quindici frasi diventano cinque e gran parte della trama rimane distribuita fra più frasi originali. La review precedente aveva già identificato variazioni puntuali di nome e intensità: il caso non è una gold di equivalenza perfetta. Il nuovo risultato dimostra però che il forward quasi nullo dipende fortemente dall'unità di premessa. Anche la direzione reverse del documento rimane problematica: aumentare il contesto non risolve ogni difficoltà di coreferenza o ipotesi congiunta.

## Omissioni ed errori non scompaiono tutti

| Caso e misura | Singole | Fino a 2 | Fino a 3 | Documento |
| --- | ---: | ---: | ---: | ---: |
| Reuters, problemi legali omessi: minimo reverse | 0,000334 | 0,000447 | 0,000447 | 0,000475 |
| Numero 100→30 piedi: minimo forward | 0,000752 | 0,002265 | 0,004252 | 0,105587 |
| Euro→franchi: minimo forward | 0,005667 | 0,021106 | 0,021106 | 0,003940 |
| Entità controllata: minimo forward | 0,002084 | 0,002084 | 0,002084 | 0,000232 |

I valori bassi su questi casi indicano che ampliare la premessa non elimina ogni segnale degli errori. Sul numero 100→30, tuttavia, il documento aumenta molto il supporto. L'interpretazione logica resta quella del report precedente: «fino a cento» non esclude in assoluto onde di trenta, ma la riscrittura cambia l'informazione quantitativa riportata.

Nella negazione controllata il minimo forward passa da 0,0906 a 0,4582 con le finestre, mentre il reverse resta basso, circa 0,0152→0,0173. È un esempio della possibile perdita di sensibilità in una direzione; non si selezionano soltanto gli esempi nei quali il contesto aiuta. Le perturbazioni e gli OpAI restano etichette presunte da controllare, non giudizi indipendenti del significato.

## Documento intero: sensibilità al contesto anche sull'identità

Un controllo identità ha supporto medio 0,9922 con singole premesse e 0,9928 con finestre fino a tre, ma soltanto **0,6624 con il documento**, con minimo 0,2904. La presenza letterale di una frase in una premessa più lunga non garantisce che questo checkpoint la riconosca sempre come implicata.

Uno dei due esempi OpAI non entra nel budget completo per tutte le ipotesi: copertura 46,2% forward e 30,8% reverse, aggregati mancanti. Non viene considerato un fallimento semantico né eliminato per presentare soltanto documenti brevi.

## Implicazione per lo studio da 126

La precedente AUC 0,958 riguarda il suo protocollo controllato e non è trasferita agli humanizer. Nel nuovo studio si conservano gli score storici e quelli contestuali come protocolli distinti; non si applica una soglia derivata da questi 19 esempi. Le finestre fino a tre sono una candidata da valutare con le annotazioni, mentre il documento intero è un confronto supplementare con limiti evidenti di affidabilità e copertura.

Le ipotesi di «similarità che sovrastima la conservazione» richiedono confronto con etichette, sensibilità/falsi allarmi o AUC sullo stesso riferimento. La differenza fra medie di coseno, BERTScore e NLI, su scale diverse, non dimostra da sola una sovrastima.

[Score per modalità](contextual_nli.csv), [unità e supporti](contextual_nli_units.jsonl), [selezione](selection.json), [metadati](contextual_nli_run.json). Il comando è `scripts/score_contextual_nli.py`; la cache contestuale è separata dalle inferenze storiche.

# Lettura illustrativa di dodici riscritture nello study v2

Questa è una lettura dell'assistente dei testi e degli score ricevuti, non un'annotazione umana cieca, una gold semantica o una validazione del giudice di naturalezza. Sono considerate **dodici coppie fonte/sistema**, tutte appartenenti alle **84 fonti non presenti nei pilot da 42**: quattro generazioni fallite e otto output `ok`. Le otto coppie valide sono state scelte per illustrare aggiunte, omissioni, fusione di frasi e divergenze fra metriche. Non sono un campione casuale e non permettono di stimare la frequenza degli errori nell'intero studio.

I nomi brevi identificano i preset fissati: baseline Qwen2.5-3B FP16 con prompt, Qwen2.5-14B NF4 con lo stesso prompt, DIPPER XXL NF4 sul passaggio intero e DIPPER XXL NF4 a finestre di tre frasi con prefisso riscritto cumulativo. Non descrivono tutte le configurazioni delle rispettive famiglie.

## Quattro fallimenti osservabili: risposte alle consegne e ripetizioni

| Fonte | Generatore originale | Riscrittore | Stato e comportamento |
| --- | --- | --- | --- |
| `Essay:223:eceeae3f088c81ef` | StableLM | Baseline 3B | `truncated`, 512 token senza EOS; sviluppa il saggio richiesto all'interno dell'input e introduce analisi strategiche nuove |
| `Reuters:992:d3932f02b4a9e76d` | StableLM | Baseline 3B | `truncated`, 512 token senza EOS; risponde a una lista di domande sulla crisi finanziaria anziché riscriverla |
| `WritingPrompts:184:b88567fa0a190a58` | GPT4All | DIPPER passaggio | `truncated`, 512 token senza EOS; entra nella ripetizione di «the screams of the dying» |
| `WritingPrompts:184:b88567fa0a190a58` | GPT4All | DIPPER tre frasi | `truncated` già nella prima delle tre finestre attese, 512 token senza EOS; ripete «the screams of the screams» e poi «the horns of the horns» |

I metadati non indicano un errore CUDA per questi quattro esiti. Il budget di uscita viene raggiunto senza terminazione. Il troncamento è la manifestazione tecnica comune, ma il comportamento testuale è differente: la baseline 3B sviluppa risposte sostanziali a consegne interne, mentre DIPPER degenera in ripetizione sullo stesso racconto di battaglia.

Nel primo Essay l'originale comincia con «Write an essay in 2900 words» e presenta dati su Diversicare. L'output parte parafrasando questi dati, poi aggiunge inefficienze operative, invecchiamento del personale, necessità di formazione e strategie finanziarie. Queste informazioni non sono tutte nell'input e il testo si interrompe mentre sviluppa nuove opportunità. In Reuters la fonte contiene soltanto domande e consegne; la baseline produce risposte su dot-com, Grande Depressione, TARP e responsabilità politiche. Non è una parafrasi della lista.

Nel racconto GPT4All l'originale comprende un giovane soldato, una lancia e un momento di esitazione. Le due versioni DIPPER si bloccano nella descrizione iniziale del campo di battaglia e non arrivano a riscrivere il resto. La finestra di tre frasi non impedisce questo specifico fallimento, che avviene prima di qualunque prefisso cumulativo.

Gli esiti sono conservati come fallimenti. Non ricevono un voto di naturalezza e non vanno trasformati in bassi score semantici arbitrari o rimossi dal conteggio delle generazioni. L'analisi semantica/stilometrica dei soli output `ok` ha denominatori differenti per sistema: **124 baseline 3B, 126 14B e 125 per ciascun DIPPER**. Le fonti con tutti e quattro gli output validi sono 123; nelle 84 nuove sono 81. Il confronto sui soli successi deve quindi essere accompagnato dai fallimenti e, quando si confrontano direttamente le medie, da un controllo sul campione comune.

## Otto output validi: i punteggi e il testo

La tabella riporta coseno documentale e medie NLI forward/reverse. I valori non sono percentuali di significato conservato; scale e aggregazioni differenti non possono essere confrontate come accuratezze. Le finestre fino a tre comprendono anche quelle più corte e prendono il massimo: l'aumento di supporto è in parte imposto dalla costruzione.

| Fonte | Preset | Coseno | NLI storico F / R | NLI finestre fino a tre F / R |
| --- | --- | ---: | ---: | ---: |
| `Reuters:289:4d4cd4a992f95fbc` | DIPPER passaggio | 0,939 | 0,331 / 0,127 | 0,702 / 0,213 |
| `Essay:489:51777f8d998ccb8e` | Baseline 3B | 0,894 | 0,123 / 0,343 | 0,397 / 0,683 |
| `Essay:489:51777f8d998ccb8e` | 14B | 0,909 | 0,293 / 0,256 | 0,654 / 0,799 |
| `Reuters:633:11827711b469c049` | DIPPER passaggio | 0,904 | 0,348 / 0,125 | 0,975 / 0,499 |
| `Reuters:633:11827711b469c049` | DIPPER tre frasi | 0,950 | 0,995 / 0,356 | 0,997 / 0,866 |
| `Essay:261:99fdf43504523caf` | 14B | 0,937 | 0,460 / 0,254 | 0,985 / 0,976 |
| `Reuters:721:b779ed927af8e5c8` | 14B | 0,905 | 0,635 / 0,084 | 0,994 / 0,768 |
| `Reuters:889:e2783d23c690fec1` | DIPPER tre frasi | 0,979 | 0,989 / 0,864 | 0,993 / 0,922 |

### Reuters 289: numeri e valute cambiano, il coseno rimane alto

La fonte StableLM riporta i risultati di Komercni Banka con importi in CZK e conversioni in USD. DIPPER passaggio cambia ripetutamente **USD in EUR**, la crescita degli attivi da **26,8 a 26,7 miliardi** e la conversione del totale degli attivi da **USD 43,75 miliardi a EUR 4,3 miliardi**. Il formato del totale in CZK diventa inoltre «1,035,775 billion», diverso da «1,035.7 billion». Il riferimento al secondo trimestre **2017** diventa «the second quarter of this year», perdendo l'ancoraggio esplicito all'anno.

Non serve verificare le notizie nel mondo reale per osservare che le due versioni differiscono in queste stringhe e quantità. Il coseno documentale è ancora 0,939, mentre NLI storico è basso in entrambe le direzioni e il reverse con finestre rimane 0,213. È un esempio di similarità tematica elevata con alterazioni numeriche, non una stima della sensibilità dell'NLI ai numeri in generale.

### Essay 489: una consegna può produrre un output `ok` senza essere parafrasata

Anche questa nuova fonte StableLM comincia con una richiesta: «Write an essay in 300 words», seguita da informazioni sul Food Safety Modernization Act. Entrambi i modelli Qwen eliminano la consegna e producono una trattazione dell'argomento. Lo stato `ok` indica terminazione tecnica, non rispetto dell'istruzione di riscrittura.

La baseline 3B aggiunge in particolare che uno studio nel *Journal of Food Protection* avrebbe trovato una diminuzione degli episodi di malattie alimentari dopo FSMA. Lo studio e quel risultato non sono menzionati nell'input; questa lettura non ne verifica esternamente esistenza o correttezza. Aggiunge anche proposte su sensibilizzazione pubblica, risorse e tecnologie. Il 14B evita quel riferimento bibliografico, ma introduce una valutazione di efficacia e miglioramenti futuri nell'applicazione della legge. Sono due profili diversi di esecuzione della consegna interna, non una vittoria automatica del 14B sulla fedeltà.

Le nuove fonti mostrano che il problema degli input-consegna non si esaurisce nei due casi già notati nel pilot. Si mantiene la politica del risultato primario con tutti gli input; un eventuale secondario sulla prosa deve usare classificazioni del tipo sul solo originale e rendere espliciti i denominatori. Non si eliminano gli input perché gli output o gli score sono sfavorevoli.

### Reuters 633: recupero di contenuto con DIPPER contestuale e limite della premessa singola

La fonte StableLM contiene una cronologia internamente problematica: prospetta una fusione fra McDonnell Douglas e Lockheed Martin e poco dopo afferma che si erano già fuse nel 1995. La valutazione di conservazione riguarda quello che è scritto, non una verifica storica della notizia o una correzione automatica dell'originale.

DIPPER passaggio elimina la fusione del 1995 e il successivo buon andamento di Lockheed Martin. Il preset a tre frasi conserva invece entrambe le informazioni. Nel primo output il reverse con finestre è 0,499; nel secondo sale a 0,866. Nel secondo rimane omesso il riferimento a una «solid foundation» basata su McDonnell Douglas, quindi non si parla di equivalenza perfetta.

Per il passaggio intero il forward storico è soltanto 0,348, ma raggiunge 0,975 con finestre: l'output riunisce informazioni della premessa distribuite su più frasi. In questo caso un basso forward storico non significa che tutto il contenuto dell'output sia inventato. Il recupero di informazioni nel nuovo preset è visibile nel testo, ma non si attribuisce causalmente alla sola finestra: cambiano anche contesto e budget complessivo di generazione.

### Essay 261 e Reuters 721: NLI storico basso su riorganizzazioni ampie

Nel testo Claude su Braudel il 14B conserva nome, scuola delle Annales, *longue durée*, titolo del libro, 1949, XVI secolo, Filippo II e i fattori commercio/agricoltura/trasporti. Riorganizza però le proposizioni e introduce formulazioni più forti come «revolutionized» e «introducing the concept». La conservazione di molta informazione centrale rende poco plausibile leggere il reverse storico 0,254 come quota di informazione rimasta; con finestre sale a 0,976. Neppure quest'ultimo valore certifica l'assenza di cambiamenti nella forza o nell'attribuzione delle affermazioni.

Nel Reuters GPT4All su Manchester United il 14B mantiene nome, possibile acquisizione, Igor Makarov, £1,1 miliardi/$1,7 miliardi, crescita superiore a un terzo, prestito da $750 milioni e consorzio cinese. Conserva anche formulazioni di incertezza. Il reverse storico è 0,084, contro 0,768 con finestre: la formulazione e l'organizzazione possono penalizzare il supporto a premessa singola anche con molti dettagli conservati. La compressione degli scenari di acquisizione e joint venture resta da valutare indipendentemente; non è una gold parafrasi perfetta.

### Reuters 889: supporto NLI alto con perdita dell'anno e rafforzamento delle affermazioni

Nel Reuters GPT4All su NatWest, DIPPER tre frasi conserva la quota **51%**, ma trasforma **«Q3 2019» in «the third quarter of this year»**. Senza un ancoraggio temporale aggiuntivo nel passaggio, l'anno esplicito viene perso. Inoltre l'originale presenta l'acquisizione come un'opportunità di ampliare prodotti e reti; l'output usa «will enable» e «will strengthen», rendendo l'esito più assertivo.

Il coseno è 0,979, il forward NLI storico 0,989 e quello con finestre 0,993. Si osserva quindi un limite opposto ai falsi allarmi da fusione: **un supporto alto può coesistere con deriva temporale e di modalità**. Il conteggio delle guardie e i contesti più larghi non certificano da soli la conservazione di queste relazioni. Non si dichiara l'intero passaggio falso o contraddittorio: il problema specifico è l'ancoraggio dell'anno e la forza delle promesse.

## Implicazioni per l'interpretazione

I dodici casi separano fallimento della generazione, esecuzione della consegna interna, errore numerico, omissione e difficoltà di allineamento. Questi fenomeni richiedono letture diverse; uno stato `ok`, un coseno elevato o un NLI elevato non bastano da soli a decidere la fedeltà. Le fonti di partenza possono inoltre essere incoerenti: correggere il mondo reale e conservare il significato del passaggio sono obiettivi differenti.

Le conclusioni quantitative devono usare tutti gli score con i denominatori corretti; questa selezione illustrativa non produce AUC, prevalenza di errori o una graduatoria di naturalezza. Le annotazioni umane indipendenti restano necessarie, anche per distinguere un'attenuazione legittima, una fusione fedele e una deriva sottile che le metriche non segnalano.

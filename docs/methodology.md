# Impostazione metodologica

Il lavoro valuta che cosa cambia quando un testo macchina viene riscritto. Le domande sono distinte: il detector continua a riconoscerlo? Il significato rimane? Lo stile si modifica? La prosa è percepita come naturale? Un risultato favorevole su una dimensione non garantisce un risultato favorevole sulle altre.

## Dataset

MGTBench fornisce testi inglesi dei domini Essay, Reuters e WritingPrompts, con testi umani e output di sette generatori. I CSV originali sono conservati in `data/raw/mgtbench/`. Gli abbinamenti della stessa fonte non implicano che umano e macchina siano parafrasi semanticamente equivalenti.

Il campione stilometrico comprende 300 testi umani e 300 macchina per dominio: 900 coppie e 1.800 testi complessivi. Il controllo della lunghezza usa prefissi che terminano a fine frase e hanno lo stesso numero di parole lessicali nella coppia. I gruppi collegati rimangono nello stesso fold.

OpAI-Bench contiene riscritture Polish, Paraphrase, Style Rewrite, Compress ed Expand. Le coppie e gli identificatori delle frasi permettono di studiare sia documenti interi sia porzioni effettivamente modificate. Le operazioni conservative forniscono controlli presunti; non sostituiscono una verifica umana del significato.

## Stilometria

L'estrattore produce 105 feature: lunghezza e variabilità delle frasi, punteggiatura, POS, parole funzionali, diversità lessicale, leggibilità e proxy dei segnali di scrittura artificiale. Le frequenze usano denominatori dichiarati; MATTR è calcolato su finestre di 50 parole.

Le differenze delle singole feature sono espresse anche con il g di Hedges, con segno macchina meno umano. La regressione logistica usa imputazione e standardizzazione apprese sul training. Le verifiche della shortlist effettuano la selezione delle feature all'interno del training. Validazione per gruppi, dominio escluso e generatore escluso rispondono a domande di generalizzazione diverse.

## Conservazione semantica

ROUGE-L e BLEU misurano sovrapposizione lessicale. Coseno e BERTScore confrontano rappresentazioni. NLI stima il supporto fra premessa e ipotesi nelle due direzioni. La suddivisione e il recupero delle evidenze rendono possibile il confronto di testi lunghi, ma introducono scelte di aggregazione che devono rimanere esplicite.

La validazione usa identità, perturbazioni controllate, riscritture OpAI e corruzioni distribuite. La costruzione delle alterazioni fornisce un intento sperimentale, non una gold label. Negazioni, entità, numeri, quantificatori, cautele e omissioni possono avere effetti differenti.

La soglia storica 0,04075 appartiene a uno specifico minimo documentale calibrato su OpAI. Non si trasferisce automaticamente ad altri aggregatori, ai punteggi locali o a riscrittori diversi. La configurazione contestuale `window2` è una scelta separata. I supporti NLI non sono percentuali di significato conservato; sulle cautele il comportamento direzionale può non seguire l'inferenza intuitiva attesa.

## Riscrittura e detection

I preset conservano modelli, revisioni, prompt e impostazioni di generazione dichiarati. Qwen 3B e Qwen 14B sono configurazioni specifiche. DIPPER sull'intero passaggio e DIPPER a finestre di tre frasi usano gli stessi pesi, ma differiscono nella procedura e nel contesto. Tre frasi non significa applicare DIPPER tre volte allo stesso testo.

La calibrazione dei detector utilizza umani separati dai gruppi di test. L'ASR condizionato misura la quota di evasioni fra gli originali inizialmente riconosciuti dal singolo detector. La variazione di TPR sull'intera coorte viene riportata separatamente. I fallimenti di generazione non diventano evasioni riuscite. Le soglie e le direzioni degli score rimangono fissate nel confronto.

Gli intervalli usano il ricampionamento dei gruppi di fonte e mantengono insieme le versioni appaiate. Le varianti dello stesso originale non sono osservazioni indipendenti. Le procedure statistiche e i semi sono dichiarati nelle configurazioni o nei manifest.

## Naturalezza

La naturalezza richiede un giudizio sulla prosa e non coincide con evasione o fedeltà. Il confronto a coppie può essere richiesto in entrambi gli ordini. Le risposte vanno riportate con copertura, instabilità e preferenza di posizione; scegliere lo stesso lato dopo lo scambio dei testi è un segnale da analizzare.

Il modulo è opzionale e l'accesso al servizio esterno è esplicito. Le etichette dell'assistente restano separate da annotazioni umane indipendenti. Un accordo fra modelli non costituisce validazione umana. Le reference umane dello studio sono ancora in raccolta; il progetto non inventa le etichette mancanti.

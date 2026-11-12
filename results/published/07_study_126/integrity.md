# Integrità del run Kaggle su 126 fonti

Audit del 10 novembre 2026 sui file ricevuti in `humanization_v2_study_protocol_v2.zip`: SHA-256 `7e55e8b510df4e26ca1f7d4a61648525a088523a8cb8983bb805fa9b16476cd8`, 27.194.720 byte compressi, 139 elementi comprese le directory, 106.156.670 byte non compressi. Sono stati ricalcolati riepiloghi da score già salvati, senza nuove generazioni, inferenze neurali o chiamate API.

**Il run contiene tutti i 630 esiti previsti, con 626 riusciti e quattro troncati.** Detector, controlli, stile e semantica sono presenti e coerenti con gli input effettivamente giudicabili. La naturalezza è invece parziale in tutti e tre i confronti: il completamento degli altri stadi non va esteso a questo asse.

## Coorte, split e continuità con i pilot

- Il pannello `study.jsonl` coincide byte per byte con quello già congelato, SHA-256 `84f9f6933b2ee72412d65eedbf8f1fbe70cf0139bcd3d6d395cb85250c0edb68`.
- Sono presenti 300 gruppi umani di calibrazione e 126 gruppi test distinti e disgiunti dalla calibrazione: sei fonti per ciascuna delle 21 celle dominio/generatore. Il pannello comprende le 42 fonti dei pilot e 84 ulteriori fonti; non è una replica interamente indipendente.
- Anche i file dei pannelli smoke e pilot sono identici ai precedenti. Configurazione v2, scale stilometriche e soglie dei detector coincidono con lo smoke verificato. Non emerge un ricampionamento della calibrazione.
- Sulle 42 fonti condivise, originali, riscritture e stati della baseline 3B e di DIPPER sul passaggio intero coincidono tutti con i rispettivi pilot. Non emerge una variazione dei preset durante l'estensione.
- Il runtime registra PyTorch `2.11.0+cu130`, due Tesla T4 e commit iniziale `3d028202a2200d1aca02b0e9f558d5f3eee1944a`. Le correzioni successive della naturalezza sono riconoscibili dal bundle incluso e dagli hash effettivi.

## Esiti di generazione e quattro fallimenti

I 630 ID sono unici e coprono cinque condizioni per ciascuna fonte. Il manifest `completed` descrive la disponibilità di un esito per ciascuna richiesta, non 630 riscritture riuscite.

| Condizione | `ok` | `truncated` | Fonte del fallimento |
| --- | ---: | ---: | --- |
| Identità | 126 | 0 | — |
| Baseline 3B | 124 | 2 | Essay 223 StableLM e Reuters 992 StableLM |
| 14B NF4 | 126 | 0 | — |
| DIPPER passaggio | 125 | 1 | WritingPrompts 184 GPT4All |
| DIPPER tre frasi | 125 | 1 | Stessa fonte WritingPrompts |

Nei due casi 3B l'originale contiene una consegna; il modello sviluppa una risposta invece di limitarsi alla riscrittura e raggiunge il budget di 512 token senza EOS. Nel racconto GPT4All entrambi i preset DIPPER producono lunghe ripetizioni e arrivano allo stesso limite senza EOS. Il preset contestuale si arresta nella prima delle tre finestre previste, con zero finestre completate. I testi parziali e i metadati restano conservati. Questi dati documentano il limite di generazione; non indicano errori CUDA o esaurimento della memoria.

Tutte e tre le fonti coinvolte nei quattro esiti troncati appartengono alle 84 fonti aggiuntive, non ai 42 casi dei pilot.

Le quattro righe non `ok` sono escluse da `rewriting/pairs.jsonl`, che contiene esattamente le 626 righe riuscite. `semantic/pairs.jsonl` e `semantic_context/pairs.jsonl` sono identici byte per byte a quel file, SHA-256 `537fceaa0ba5b8e29518b97bf789f25a8b98a5c45ba12797669eb5b24ffecfb6`.

## Detector e controlli nulli

Il file congiunto contiene 1.764 righe: 14 condizioni per ciascuna delle 126 fonti. Di queste, 1.752 sono `ok`, quattro sono le riscritture troncate e otto sono controlli marcati `reference_unavailable`, perché dipendono dalle quattro riscritture fallite. I target mancanti non vengono imputati.

Ogni detector riceve 2.304 record: 300 umani di calibrazione, 126 umani test, 126 originali macchina e 1.752 output validi. I tre classificatori hanno quindi 6.912 score complessivi; Binoculars ne ha 2.304. I fallimenti non ricevono score fittizi e rimangono insuccessi nel denominatore ASR quando l'originale è rilevato.

Ricalcoli indipendenti sugli score ricevuti confermano, entro 10⁻¹⁰ nella lettura CSV:

- tutte le 224 righe del riepilogo dei classificatori e le 56 righe macro;
- tutte le 56 righe del riepilogo Binoculars;
- le 40 righe del confronto dei controlli nulli, con corretta decisione AI verso il basso per Binoculars.

Le soglie ricalcolate sono identiche a quelle salvate e a quelle dello smoke. Binoculars conserva il ratio grezzo, distinto da una probabilità, e la maggioranza resta quella dei tre classificatori RoBERTa. I confronti nulli usano i casi eleggibili e gli stessi originali baseline-AI; il numero totale dei controlli non va confuso con il denominatore del confronto appaiato.

## Semantica, contesto e stile

Tutti i sei stadi semantici risultano `executed` sugli stessi 626 ID, con hash delle coppie coerente: lessicale, embedding, BERTScore, NLI, coseno allineato e BERTScore allineato. Stile, guardie ed export continuo contengono anch'essi 626 righe con gli stessi ID. Le scale sono stimate sui 300 umani di calibrazione, senza usare macchine o test.

Il NLI contestuale registra 199.126 confronti unici con DeBERTa-v3 base float32 su CUDA e 5.008 aggregati: 626 coppie × due direzioni × quattro modalità. La copertura è completa per tutte le 1.252 righe di ciascuna modalità `single_all`, `window2` e `window3`. Nel documento intero sono complete 1.248 righe su 1.252; quattro righe reverse hanno aggregati correttamente mancanti, senza media delle sole ipotesi che entrano nel budget. Sono i casi 3B Essay 641 ChatGPT-turbo, 3B Essay 245 StableLM, 14B Reuters 903 StableLM e 14B Reuters 992 StableLM.

Le finestre annidate aumentano il massimo supporto anche per costruzione. Completezza tecnica e copertura non dimostrano accuratezza semantica; non viene ricostruita una gold dai punteggi stessi.

## Annotazioni e provenienza del codice

Il sampler prepara 183 coppie su 46 fonti: 84 reference su 21 fonti e 99 diagnostiche su 25. Il target diagnostico di 100 non è raggiunto e il manifest lo dichiara. Il raggruppamento completo delle fonti e la disponibilità degli output validi non vengono alterati per arrivare artificialmente a cento. Entrambi i moduli degli annotatori contengono 183 righe con giudizi, tipi di errore e note vuoti. Non sono presenti nuove etichette umane da usare come riferimento.

Gli hash di riscrittura (`e9b77323…`) e Binoculars (`c6bdfb61…`) coincidono con i sorgenti del commit iniziale `3d02820`. Il modulo NLI contestuale conserva hash `14ade40e…`. I tre sorgenti del bundle `naturalness_resume` coincidono con quelli della correzione pubblicata: modulo naturalezza `e0159c84…` e driver `7033478a…`. Il manifest del giudice registra il medesimo hash del client. Non si attribuisce quindi tutto il run al solo checkout iniziale, ignorando la ripresa.

Il controllo dei file esportati non trova stringhe con i pattern delle credenziali Groq o Hugging Face. Non sono esportati pesi o venv. Questo è un controllo sui materiali ricevuti, non una certificazione di dati esterni all'archivio. I pesi effettivi non sono nello ZIP: l'audit verifica revisioni e metadati registrati, non ricontrolla indipendentemente i byte dei checkpoint.

Non emerge un errore aritmetico o una perdita silenziosa di casi nei riepiloghi verificati. Restano i limiti sostanziali dichiarati: quattro generazioni fallite, naturalezza parziale e sensibile all'ordine, assenza di annotazioni umane e necessità di separare le 84 fonti non viste per le ipotesi sviluppate sul pilot.

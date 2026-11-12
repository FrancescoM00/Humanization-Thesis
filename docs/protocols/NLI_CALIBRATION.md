# Scelta della configurazione NLI sullo split di calibrazione

Specificazione del **10 novembre 2026**, successiva alla lettura dello study da 126 fonti e precedente alle nuove inferenze di questo confronto e alla ricezione delle annotazioni reference. Non è una preregistrazione retroattiva dello study. La selezione è uno sviluppo dello strumento su etichette proxy; la sua validazione umana resta successiva e distinta.

## Campione e dati esclusi

Si conservano lo split per fonte e il collegamento positivo/negativo già definiti nell'esperimento OpAI:

- **139 positivi presunti e 297 alterazioni intenzionali**, da 46 gruppi di fonte;
- i negativi sono soltanto `opai_matched_negative` con `evaluation_split="calibration"`; i positivi sono quelli richiamati da `metadata.matched_positive_id`;
- si aggiungono **137 identità**, una per originale distinto entro il gruppo di fonte dei positivi selezionati. Chunk e testo vengono copiati esattamente in entrambe le direzioni;
- totale **573 coppie**, senza aggiungere fonti indipendenti. Le identità non entrano nell'AUC.

Le 141 coppie positive e 265 alterate dello storico test, su 48 fonti, **non partecipano alla selezione**. La preparazione può leggere i metadati dello split per verificare la disgiunzione, non i suoi score. Sono esclusi anche i positivi non collegati ai negativi del confronto principale. Non si tratta di un nuovo campione gold: parafrasi OpAI e perturbazioni restano etichette operative presunte.

Gli input sono in [exports/nli_context_calibration/](https://github.com/FrancescoM00/Humanization/blob/d2868b323938a2a9371c2a872ad86b5596d0f097/exports/nli_context_calibration/README.md): coppie, allineamenti esatti, etichette, policy e manifest. La preparazione rifiuta di sostituire input diversi nella stessa directory.

## Modello e modalità

Si riusa il medesimo DeBERTa-v3 **base float32**, `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, revisione `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`. Non si modifica il modello, non si usa quantizzazione e non si passa a xsmall. Il limite della coppia premessa/ipotesi è 384 token; nessun troncamento silenzioso.

Le quattro modalità sono quelle già implementate in [semantics/contextual_nli.py](../../src/semantics/contextual_nli.py):

| Modalità | Candidati a premessa |
| --- | --- |
| `single_all` | Tutte le unità singole |
| `window2` | Tutte le singole e le finestre contigue fino a due unità |
| `window3` | Tutte le singole e le finestre contigue fino a tre unità |
| `document` | Documento esatto, solo quando la coppia entra nel budget |

Le unità ipotesi sono identiche fra modalità. Si prende il massimo entailment fra i candidati ammissibili per ciascuna ipotesi, in entrambe le direzioni. Una modalità con ipotesi senza premessa ammissibile conserva la copertura incompleta e non produce un aggregato parziale spacciato per completo. Le finestre sono annidate: il supporto può aumentare per costruzione, senza dimostrare maggiore accuratezza.

## Sedici candidati e regola di scelta

Per ciascuna modalità si confrontano media, q10, minimo e quota delle ipotesi con supporto inferiore a 0,5: **16 candidati**. Lo score documentale di conservazione è il minimo delle due direzioni dell'aggregatore. Per la quota non supportata è `1 - max(quota_forward, quota_reverse)`, così valori alti indicano sempre conservazione presunta. Lo 0,5 definisce questa quota, non un verdetto di conservazione sui veri humanizer.

Un candidato è ammissibile come primario soltanto se:

1. ha copertura completa in entrambe le direzioni per tutte le 436 coppie etichettate e tutte le 137 identità;
2. la **media degli score documentali delle identità** è almeno 0,95;
3. il **decimo percentile della distribuzione degli score documentali delle identità** è almeno 0,90.

Questi ultimi due criteri sono soglie operative di coerenza, fissate prima delle nuove inferenze, non limiti universali di validità NLI. Il percentile fra documenti non va confuso con l'aggregatore q10 fra frasi. Le identità sono controlli necessari ma non dimostrano la correttezza sulle parafrasi, fusioni o omissioni.

Fra gli ammissibili si sceglie la massima **AUC aggregata sul solo split di calibrazione**, con positivi presunti = 1 e alterazioni = 0. Le parità entro `1e-12` preferiscono la premessa più corta, poi l'ordine media, q10, minimo, quota. Se nessuno passa, l'esito è `no_eligible_candidate`: non si forza una modalità primaria e non si allentano i requisiti dopo aver visto il risultato.

Si riportano tutti i candidati, ragioni di esclusione, distribuzione delle identità e AUC per famiglia di alterazione con i rispettivi positivi collegati. Gli intervalli AUC ricampionano **i 46 gruppi di fonte**, con 2.000 repliche e seed 42. Descrivono l'incertezza di sviluppo, senza correggere l'ottimismo della selezione. Non si calcola l'AUC globale eliminando solo i casi privi di copertura.

## Esecuzione e risultati ricevuti

La preparazione locale e il piano con tokenizer sono eseguiti. Il successivo ZIP di calibrazione è stato ricevuto il **10 novembre 2026**: tutte le 323.336 evidenze sono completate su GPU, con DeBERTa-v3 base float32 e circa 53,7 minuti di inferenza. Input e policy corrispondono a quelli congelati. [Risultati](../../results/08_nli_calibration/calibration/report.md), [audit](../../results/08_nli_calibration/calibration/integrity.md), [interpretazione](../../results/08_nli_calibration/calibration/interpretazione.md).

La regola seleziona **`window2` + `min`**, registrati in [selection.json](../../results/08_nli_calibration/calibration/selection.json). AUC di calibrazione **0,9734879**, intervallo bootstrap per fonte 95% **[0,9566194; 0,9873532]**. La media degli score delle 137 identità è **0,9710096** e il decimo percentile **0,9509247**: i criteri sono soddisfatti. Il minimo individuale è però 0,8666 e tre identità sono sotto 0,90; non si dichiara che tutte le identità abbiano score quasi unitario.

Sul medesimo pannello, `single_all + min` ha AUC 0,9470 e `window3 + min` 0,9712. La differenza fra due e tre unità è piccola e non dimostra superiorità fuori campione: la selezione segue il massimo puntuale già stabilito, non una nuova regola aggiunta dopo i risultati. Il confronto con lo storico 0,958 sul test 141/265 non misura un miglioramento diretto, perché split e percorso di scoring sono diversi.

Le singole e finestre hanno copertura completa. Il documento ha entrambe le direzioni complete soltanto per **77/436 coppie etichettate e 36/137 identità**: resta secondario ed escluso dalla scelta per copertura, non per un'AUC globale bassa calcolata su un sottoinsieme favorevole. Non si ricorre a troncamento.

Nel cloud corrente PyTorch è CPU e CUDA non è disponibile; lo scoring effettivo è quindi stato eseguito su Kaggle. Le vecchie cache coprivano circa il 3% del piano: non sono state mescolate probabilità di runtime CPU/GPU per quel piccolo risparmio.

La [guida Kaggle](https://github.com/FrancescoM00/Humanization/blob/d2868b323938a2a9371c2a872ad86b5596d0f097/kaggle/README_NLI_Calibration.md) riusa pesi e venv quando disponibili ed esegue il vecchio scorer, seguito dall'analizzatore nuovo. Il pacchetto di ritorno comprende input, policy, metadati, CSV e selezione; le cache di probabilità e le evidenze estese rimangono sulla macchina per riprendere, fuori dallo ZIP leggero. Non si eseguono humanizer, detector o chiamate Groq.

## Modalità primaria per la verifica umana

La scelta è fissata prima di ricevere le nuove annotazioni reference. Per ciascuna ipotesi si prende il massimo entailment fra una o due unità contigue ammissibili; il minimo fra tutte le ipotesi viene calcolato separatamente forward e reverse; lo score primario è il minimo delle due direzioni. Le unità sono frasi/chunk già fissati, non sempre frasi intere.

Gli aggregati `window2` erano già presenti nello study: la [tabella primaria](../../results/07_study_126/nli_primary_scores.csv) è stata ricavata senza nuove inferenze, con 626 score validi e quattro fallimenti espliciti. Tutte le 84 reference hanno lo score. L'applicazione è descrittiva e successiva allo studio; i risultati originari delle altre modalità sono conservati come secondari. Non si cambia a posteriori la modalità per accordarla alle etichette umane e non si introduce una soglia automatica.

## Annotazione successiva

Sono pronte le [84 coppie reference cieche](https://github.com/FrancescoM00/Humanization/blob/d2868b323938a2a9371c2a872ad86b5596d0f097/exports/kaggle_v2_study/reference/README.md), già selezionate senza score nello study. Conservano ID e testi, con chiavi separate. La scelta NLI va congelata prima di consultare le nuove etichette umane; i giudizi non verranno usati per scegliere il candidato più favorevole sullo stesso campione.

Prima di ricevere i giudizi, si fissa anche la lettura principale delle etichette: **yes = conservazione integrale, partial/no = alterazione** per l'AUC del punteggio continuo. Le tre classi rimangono distinte nelle distribuzioni; un confronto yes/no che escluda partial sarà soltanto secondario, con denominatore esplicito. Mancanti non diventano yes e, se manca una delle due classi, l'AUC resta non calcolabile. Gli intervalli ricampionano le 21 fonti, non trattano le 84 riscritture come fonti indipendenti. Un solo annotatore viene dichiarato come tale; se ne partecipano due, si conservano i giudizi separati e l'eventuale conciliazione avviene senza consultare gli score, senza scegliere il lettore più favorevole.

Le 99 diagnostiche restano un approfondimento separato. Non si trasferiscono allo study l'AUC di calibrazione o la soglia storica OpAI 0,04075. Anche dopo la scelta, gli score rimangono continui e le modalità secondarie rimangono disponibili. La taratura del giudice di naturalezza e la revisione statistica dei detector sono attività successive, non già eseguite da questo pacchetto.

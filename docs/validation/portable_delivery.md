# Verifica della consegna portabile

Verifica eseguita il 12 novembre 2026.

- Suite Python: **217 test passati**, uno saltato perché richiede CUDA.
- Rimosse le traduzioni automatiche dei vecchi nomi di cartella. I percorsi pubblicati si risolvono una sola volta; un doppio `published` è segnalato come errore di configurazione.
- Verificati gli input dichiarati dai protocolli eseguibili e i 16 Parquet ufficiali OpAI. I tre CSV MGTBench sono invariati.
- Controllate le impronte di **366 file scientifici**, tutti invariati, e le copie portabili dei documenti di riferimento. Conservati separatamente **75 documenti originali**, identici byte per byte.
- Ricalcolati i **35 classificatori** della selezione da 100 feature: selezioni, partizioni e predizioni coincidono con il riferimento; differenza massima nelle predizioni uguale a zero.
- Riprodotte le analisi delle shortlist, la calibrazione NLI e il controllo sulle cautele a partire dagli score inclusi.
- Riprodotta l'evaluation dello studio e del pilot: **209 + 15 controlli di colonne numeriche**, tutti entro la tolleranza di 1e-12; differenza massima 1,11e-16. I dettagli sono in [portable_study_check.csv](portable_study_check.csv) e [portable_pilot_check.csv](portable_pilot_check.csv).
- ZIP costruito da una lista esplicita di inclusione, senza archivio storico, Git, ambiente virtuale, pesi, credenziali o nuovi run. Verificati gli hash dei file estratti.
- Gli otto comandi accettano `--help` anche avviati dall'esterno della cartella estratta. Il controllo dati e la riproduzione dell'evaluation funzionano senza checkout Git.

Questa verifica non comprende nuove inferenze dei modelli o chiamate al giudice remoto. Le evidenze originali sono state conservate e la cronologia Git non è stata riscritta.

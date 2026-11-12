# Confronto dei CSV MGTBench forniti da Google Drive

I tre CSV caricati dall’autore, indicati come scaricati dalla [cartella Google Drive segnalata da TH-Bench](https://drive.google.com/drive/folders/1p4iBeM4r-sUKe8TnS4DcYlxvQagcmola), sono **identici byte per byte** ai file del mirror già utilizzati negli esperimenti.

Verifica eseguita il **2026-11-09T00:48:32+01:00**. Sono stati confrontati direttamente i byte e gli SHA-256, poi intestazioni, righe nello stesso ordine e tutte le celle CSV. Gli hash coincidono anche con i manifest storici di baseline, pilot e controllo della lunghezza.

| File | Righe dati | Colonne | Byte | Celle diverse |
| --- | ---: | ---: | ---: | ---: |
| Essay_LLMs.csv | 1000 | 10 | 17.216.305 | 0 |
| Reuters_LLMs.csv | 1000 | 12 | 13.727.537 | 0 |
| WP_LLMs.csv | 1000 | 10 | 13.216.282 | 0 |

## SHA-256

**`Essay_LLMs.csv`** — stesso hash sull’allegato, sul file locale e nei tre manifest:

```text
f7ae324296b3d81b2134e873406f70a457c0383168aa7a9de83b70051ca41707
```

**`Reuters_LLMs.csv`** — stesso hash sull’allegato, sul file locale e nei tre manifest:

```text
d3a31dd8fd3b04ec333ec29ad3a929618b5a88beaf2e8338b50594957a6fa32b
```

**`WP_LLMs.csv`** — stesso hash sull’allegato, sul file locale e nei tre manifest:

```text
ee787f528f2b5d4440cf9cd714ea4f81f9cf7b59086c37d90a376ba3db3a7278
```

## Conseguenze per la tesi

La precedente incertezza sull’identità dei tre CSV del mirror rispetto ai file Drive forniti dall’autore è risolta. La via del primo download resta il mirror fissato: il confronto successivo ne verifica il contenuto, senza riscrivere la cronologia della raccolta. **Non occorre rilanciare gli esperimenti e nessun risultato numerico cambia.**

Rimane distinta la preparazione locale dei campioni: pulizia, campionamento e matching sono operazioni del progetto e non diventano per questo una riproduzione dei file di attacco chiamati `clean` in TH-Bench.

I manifest dei run e `data/raw/SOURCE.json` sono conservati nel loro stato storico: il campo di verifica inizialmente falso descrive le conoscenze disponibili durante quei run. La verifica successiva è registrata nel [manifest di questo confronto](mgtbench_drive_comparison.json). Analogamente, gli avvisi nei report generati descrivono lo stato al momento dell’esecuzione; per la provenienza aggiornata fa fede questa nota.

## Ambito del controllo

Il confronto riguarda i tre CSV singoli allegati dall’autore. Non è stato effettuato un nuovo download diretto da Google Drive. Lo ZIP `old_datasets-20261108T234347Z-1-001.zip` supera il limite di trasferimento di 32 MiB e **non è stato ispezionato**; non è necessario per verificare l’identità dei tre CSV già disponibili separatamente.

Gli allegati non sono stati copiati su Git: coincidono con i dati già archiviati nel backup del progetto.

## Ripetere il controllo

Con i CSV di Drive scaricati in una cartella separata, dalla radice del progetto:

```bash
cmp data/raw/Essay_LLMs.csv /percorso/drive/Essay_LLMs.csv
cmp data/raw/Reuters_LLMs.csv /percorso/drive/Reuters_LLMs.csv
cmp data/raw/WP_LLMs.csv /percorso/drive/WP_LLMs.csv
```

Ciascun comando deve terminare con codice 0 e senza output. Gli hash da confrontare sono riportati sopra e nei manifest storici.

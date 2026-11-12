# Provenienza dei documenti portabili

I risultati numerici e i dati scientifici non sono stati modificati dalla pulizia del progetto. Le loro impronte sono elencate in `../validation/research_artifact_fingerprints.json`.

Alcuni documenti di riferimento contenevano percorsi della macchina sulla quale erano stati prodotti. La consegna include descrizioni portabili con percorsi relativi: sono copie per consultare e riprodurre i risultati, non registrazioni di nuove inferenze. Il file `portable_reference_documents.json` collega ogni copia alla SHA-256 del documento originale e ne riporta l'impronta corrente.

I protocolli eseguibili sono in `configs/protocols/`. Per le due analisi sulla selezione delle feature, `execution_protocol.json` ed `execution_manifest.json` descrivono gli input e le predizioni di riferimento nella struttura attuale. I nuovi fit generano invece i propri manifest e non sovrascrivono i risultati pubblicati.

I documenti originali sono conservati integralmente in un archivio di evidenze separato. La cronologia Git non è stata riscritta: il tag `pre-portable-delivery` identifica anche la versione precedente alla pulizia. L'archivio storico non fa parte dello ZIP destinato all'esecuzione.

Le attribuzioni scientifiche, le revisioni dei modelli e i termini dei dataset rimangono in `THIRD_PARTY_NOTICES.md` e nei file `SOURCE.json`.

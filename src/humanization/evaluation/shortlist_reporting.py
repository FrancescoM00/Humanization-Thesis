"""Report leggibile e figure autonome dalle tabelle salvate della shortlist."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from humanization.evaluation.reporting import markdown_table


LABELS = {
    "C1_full105": "C1: tutte le 105",
    "C2_fixed4": "C2: quattro storiche*",
    "C3_training4": "C3: quattro dal training",
    "C4_random": "C4: casuali uniformi",
    "C4_random_matched": "C4: casuali, stesse famiglie",
    "C1_full100": "Riferimento: 100",
    "C1_style93": "Riferimento: 93",
    "word_count_only": "Solo numero di parole",
}
MAIN = tuple(LABELS)[:5]
PROTOCOLS = {"cv": "Validazione ripetuta", "leave_domain_out": "Dominio escluso",
             "leave_generator_out": "Generatore escluso"}


def decimal(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


def interval(row):
    return f"[{decimal(row.ci_low)}; {decimal(row.ci_high)}]"


def write_report(results: Path | str, protocol_path: Path | str | None = None) -> Path:
    results = Path(results)
    metrics = pd.read_csv(results / "metrics.csv")
    differences = pd.read_csv(results / "paired_differences.csv")
    frequencies = pd.read_csv(results / "selection_frequency.csv")
    references = pd.read_csv(results / "random_reference_summary.csv")
    selections = pd.read_csv(results / "selections.csv")
    protocol = json.loads((Path(protocol_path) if protocol_path else results / "protocol.json").read_text())
    manifest = json.loads((results / protocol.get("fit_manifest_file", "manifest.json")).read_text())
    stats = json.loads((results / "statistics_manifest.json").read_text())
    def metric(name, config):
        return metrics[(metrics.protocol == name) & (metrics.scope == "macro") &
                       (metrics.configuration == config)].iloc[0]
    def difference(name, left, right):
        return differences[(differences.protocol == name) & (differences.scope == "macro") &
                           (differences.left_configuration == left) &
                           (differences.right_configuration == right)].iloc[0]
    def frequency(name, feature):
        row = frequencies[(frequencies.protocol == name) & (frequencies.feature == feature)].iloc[0]
        return f"{row.selection_count}/{row.n_fits}"
    def random_row(name, ref):
        return references[(references.protocol == name) & (references.scope == "macro") &
                          (references.reference == ref)].iloc[0]
    def transfer_table(name, scope):
        frame = metrics[(metrics.protocol == name) & (metrics.scope == scope) &
                        metrics.configuration.isin(MAIN[:3])]
        pivot = frame.pivot(index="stratum", columns="configuration", values="auc")
        pivot = pivot.reindex(columns=list(MAIN[:3])).map(decimal)
        pivot.columns = ["C1 Full", "C2 Fixed*", "C3 Training"]
        return markdown_table(pivot.rename_axis("Test escluso").reset_index())

    # Esporta le figure scientifiche standard separatamente dal report PDF.
    colors = ["#22577a", "#b6483e", "#21816b", "#777777", "#7953a3"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharey=True)
    y = np.arange(len(MAIN))
    for ax, (name, title) in zip(axes, PROTOCOLS.items()):
        for i, (config, color) in enumerate(zip(MAIN, colors)):
            row = metric(name, config)
            ax.errorbar(row.auc, i, xerr=[[row.auc-row.ci_low], [row.ci_high-row.auc]],
                        fmt="o", color=color, capsize=3, markersize=5)
            ax.text(row.ci_high + .009, i+.1, f"{row.auc:.3f}", fontsize=8)
        ax.axvline(.5, ls="--", color="#aaaaaa", lw=1)
        ax.set_xlim(.48, .98)
        ax.set_ylim(len(MAIN)-.5, -.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("AUC macro e IC 95% per gruppo", fontsize=9)
        ax.grid(axis="x", alpha=.15)
    axes[0].set_yticks(y, [LABELS[c] for c in MAIN], fontsize=9)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(results / f"auc_comparison.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    observed = frequencies[frequencies.protocol == "cv"].copy()
    names = observed.sort_values(["selection_count", "feature"], ascending=[False, True]).head(8).feature.tolist()
    for name in protocol["fixed_features"]:
        if name not in names:
            names.append(name)
    rates = observed.set_index("feature").loc[names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(np.arange(len(names)), rates.selection_rate,
            color=["#b6483e" if flag else "#22577a" for flag in rates.historical_shortlist])
    ax.set_yticks(np.arange(len(names)), names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    for i, row in enumerate(rates.itertuples()):
        ax.text(row.selection_rate+.015, i+.1, f"{row.selection_count}/{row.n_fits}", fontsize=9)
    ax.set_xlabel("Frazione dei training CV in cui C3 seleziona la feature")
    ax.set_title("Stabilità descrittiva: rosso = shortlist storica", fontsize=11)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(results / f"selection_stability.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    primary_rows = []
    for config in MAIN:
        row = metric("cv", config)
        primary_rows.append([LABELS[config], decimal(row.auc), interval(row)])
    primary = markdown_table(pd.DataFrame(primary_rows, columns=["Configurazione", "AUC", "IC 95% per gruppo"]))
    transfer_rows = []
    for name in PROTOCOLS:
        transfer_rows.append([PROTOCOLS[name], *[decimal(metric(name, c).auc) for c in MAIN]])
    transfer = markdown_table(pd.DataFrame(transfer_rows, columns=["Protocollo", "C1", "C2*", "C3", "C4 uniforme", "C4 famiglie"]))
    comparisons = []
    for left, right in (("C3_training4", "C1_full105"), ("C3_training4", "C2_fixed4"),
                        ("C3_training4", "C4_random"), ("C3_training4", "C4_random_matched"),
                        ("C2_fixed4", "C1_full105"), ("C2_fixed4", "C4_random_matched")):
        row = difference("cv", left, right)
        comparisons.append([f"{LABELS[left]} meno {LABELS[right]}", decimal(row.auc_difference), interval(row)])
    comparison = markdown_table(pd.DataFrame(comparisons, columns=["Confronto appaiato CV", "Differenza AUC", "IC 95%"] ))
    stable_names = ["mattr_50", "pos_num", "pos_cconj", "pos_propn", "ttr", "punct_total_per100", "fw_and"]
    stable = markdown_table(pd.DataFrame([[name, *[frequency(p, name) for p in PROTOCOLS]] for name in stable_names],
                              columns=["Feature", "CV: 25 training", "Domini: 3", "Generatori: 7"]))
    random_rows = []
    for name in PROTOCOLS:
        for ref in ("C4_random", "C4_random_matched"):
            row = random_row(name, ref)
            random_rows.append([PROTOCOLS[name], "Uniforme" if ref == "C4_random" else "Stesse famiglie",
                                decimal(row.median_auc), f"{decimal(row.q025_auc)}–{decimal(row.q975_auc)}",
                                f"{round(100*row.fraction_ge_C2_fixed4)}/100",
                                f"{round(100*row.fraction_ge_C3_training4)}/100"])
    random_table = markdown_table(pd.DataFrame(random_rows, columns=["Protocollo", "Estrazione", "Mediana AUC", "Quantili 2,5–97,5%", "Almeno C2*", "Almeno C3"]))
    selected = selections[selections.selected.astype(str).str.lower().eq("true")]
    selected_rows = []
    for held_out, part in selected[selected.protocol == "leave_domain_out"].groupby("held_out"):
        selected_rows.append([held_out, ", ".join(f"`{x}`" for x in part.sort_values("rank").feature)])
    selected_table = markdown_table(pd.DataFrame(selected_rows, columns=["Dominio escluso", "Quattro scelte sul training degli altri due domini"]))
    lodo_delta = difference("leave_domain_out", "C3_training4", "C1_full105")
    cv_gap = difference("cv", "C3_training4", "C1_full105")
    random_cv = random_row("cv", "C4_random_matched")
    random_lgo = random_row("leave_generator_out", "C4_random_matched")
    reference_table = markdown_table(pd.DataFrame([
        [LABELS[c], *[decimal(metric(p, c).auc) for p in PROTOCOLS]]
        for c in ("C1_full105", "C1_full100", "C1_style93", "word_count_only")],
        columns=["Pannello", "CV macro", "Dominio escluso", "Generatore escluso"]))
    text = f"""# Quattro feature stilometriche: verifica predittiva e stabilità

**Esperimento eseguito l'11 novembre 2026.** Dati storici conservati; nessun nuovo testo generato, modello linguistico o chiamata API.

## Risposta alla domanda

La shortlist storica contiene segnale: AUC CV **{decimal(metric('cv','C2_fixed4').auc)}**, contro **{decimal(metric('cv','C4_random').auc)}** delle quaterne casuali uniformi e **{decimal(metric('cv','C4_random_matched').auc)}** di quelle con la stessa composizione per famiglie. Tuttavia non sostituisce il pannello completo, che raggiunge **{decimal(metric('cv','C1_full105').auc)}**.

Selezionando quattro feature sul solo training si ottiene AUC **{decimal(metric('cv','C3_training4').auc)}**. La procedura supera i due riferimenti casuali medi, ma perde **{decimal(abs(cv_gap.auc_difference))} AUC** rispetto alle 105 feature. Fra le quattro caratteristiche storiche, solo **MATTR** viene selezionata in tutti i 35 training. Questo risultato non conferma indipendentemente la shortlist e non identifica un insieme unico di quattro indicatori ottimali.

## Che cosa abbiamo confrontato

Il campione contiene **1.800 testi: 900 umani e 900 macchina**, suddivisi in 300 abbinamenti per ciascuno di Essay, Reuters e WritingPrompts. Ogni abbinamento ha lo stesso numero di parole lessicali. Non significa che umano e macchina abbiano lo stesso contenuto: sono associazioni del disegno di campionamento, non coppie di parafrasi. Ogni gruppo del campione congelato contiene un umano e una macchina; i due testi rimangono sempre insieme nelle partizioni.

- **C1 Full:** tutte le 105 feature, compresi indicatori diagnostici come TTR e conteggi generali.
- **C2 Fixed:** `pos_num`, `pos_cconj`, `mattr_50`, `pos_propn`. I loro nomi erano già stati scelti sull'intero campione: tutti i risultati C2, contrassegnati con un asterisco, restano esplorativi.
- **C3 Training:** quattro feature con maggiore valore assoluto del g di Hedges sul training. Si rimuovono candidati con correlazione assoluta di Spearman almeno 0,85 con una feature già scelta. Ranking, correlazioni e gestione dei mancanti non usano il test. È una nuova regola, distinta dal precedente filtro di coerenza fra domini e generatori.
- **C4 uniforme:** 100 quaterne casuali distinte, fissate senza consultare etichette o valori.
- **C4 famiglie:** altre 100 quaterne con tre POS e una misura lessicale, come composizione della shortlist. Controlla il fatto che 59 delle 105 colonne riguardino parole funzionali.

La regressione logistica mantiene C=1, bilanciamento delle classi e regolarizzazione L2. Imputazione con mediana e standardizzazione sono apprese sul solo training; nessuna ottimizzazione di iperparametri. Anche C2 riceve coefficienti appresi sul training: l'informazione precedente riguarda la scelta delle colonne, non il fit dei coefficienti.

La validazione usa cinque fold per gruppo, ripetuti con cinque semi e cinque partizioni realmente diverse. Si aggiungono tre test lasciando fuori un dominio e sette lasciando fuori un generatore. Tutte le configurazioni condividono le partizioni. Sono completati **{manifest['n_fits']:,} fit**, con **{manifest['n_convergence_warnings']} avvisi di convergenza**; fit finali circa {manifest['elapsed_seconds']:.0f} secondi, analisi statistica circa {stats['elapsed_seconds']:.0f} secondi su CPU.

Nella CV principale il training comprende esempi di tutti e tre i domini; ogni test contiene gruppi esclusi da quel training. L'AUC viene poi calcolata separatamente per dominio. Nel leave-one-domain-out, invece, il dominio di test è interamente assente dal training.

La misura principale è la **media delle AUC dei tre domini**, ulteriormente mediata fra le cinque ripetizioni. Nel trasferimento fra generatori si mediano le sette AUC. Per C4 si mediano le AUC delle quaterne, non le probabilità: non è un ensemble. AUC 0,5 indica un ordinamento vicino al caso e AUC 1 una separazione completa; non è una percentuale di testi classificati correttamente.

Gli intervalli usano **2.000 ricampionamenti appaiati dei gruppi**, mantenendo insieme tutte le ripetizioni dello stesso testo. Sono condizionati alle predizioni salvate: non includono tutta l'incertezza del riaddestramento o della scelta del protocollo. Sono intervalli puntuali esplorativi, senza correzione complessiva per confronti multipli.

## Risultato principale

{primary}

Nota su C2: le quattro colonne sono state scelte sul campione completo prima di questa analisi; non è una conferma su dati indipendenti. Gli intervalli C4 riguardano la media delle quaterne fissate, non la dispersione fra singole quaterne.

Le AUC CV dei singoli domini, con training che comprende tutti e tre i domini, sono:

{transfer_table('cv','domain').replace('Test escluso', 'Dominio valutato')}

{comparison}

Le quattro feature selezionate nel training migliorano rispetto alla shortlist storica nella CV, ma il pannello completo conserva un vantaggio netto. Non è stato fissato un margine di non inferiorità: non si dichiara equivalenza fra modelli.

![AUC macro e intervalli appaiati per gruppo. C2 rimane esplorativa; C4 mostra medie di quaterne.](auc_comparison.png)

## Quanto conta il riferimento casuale

{random_table}

I quantili nella tabella descrivono la **dispersione fra le 100 quaterne** e non sono intervalli di confidenza. Le ultime colonne contano quante quaterne eguagliano o superano le AUC di C2 e C3. Non sono p-value né misure di ottimalità.

In CV nessuna delle 100 quaterne uniformi raggiunge C2, ma **{round(100*random_cv.fraction_ge_C2_fixed4)} su 100** quaterne delle stesse famiglie la raggiungono o superano. Nel trasferimento a generatori esclusi il conteggio diventa **{round(100*random_lgo.fraction_ge_C2_fixed4)} su 100**. Parte del vantaggio rispetto al caso dipende quindi anche dalle famiglie considerate; la specifica shortlist non emerge come scelta unica. C3 supera la media dei due riferimenti, senza superare ogni singola quaterna.

## Quali feature vengono scelte davvero

{stable}

Le frequenze sono descrittive: i training si sovrappongono e non sono 35 prove indipendenti. **MATTR** misura la varietà lessicale su finestre di 50 parole; **TTR** misura il rapporto fra parole diverse e parole totali; `punct_total_per100` è la frequenza complessiva della punteggiatura; `fw_and` è la frequenza di “and”. TTR è ammessa perché il nuovo insieme candidato comprende tutte le 105 feature; era esclusa dal classificatore stilometrico storico principale per la sua sensibilità alla lunghezza.

Nella CV, MATTR, TTR e punteggiatura totale compaiono sempre; il quarto posto cambia. Della shortlist storica, numerali compare raramente, mentre congiunzioni coordinanti e nomi propri non vengono selezionati da C3. **Questo non dimostra che siano inutili:** possono contenere informazione già rappresentata da altre colonne e la nuova regola massimizza un effetto aggregato, mentre quella storica cercava coerenza fra domini e generatori.

![Frequenze di selezione nei 25 training CV. Le barre rosse identificano le quattro feature storiche.](selection_stability.png)

## Trasferimento a domini e generatori esclusi

{transfer}

Nel trasferimento fra domini C3 e C1 sono vicini: differenza **{decimal(lodo_delta.auc_difference)}**, IC 95% **{interval(lodo_delta)}**. Questo non prova equivalenza. Il valore apparentemente migliore di C2 non può essere interpretato come conferma su domini mai visti: il dominio era escluso dal fit della regressione, ma era già stato usato nella selezione storica delle quattro colonne.

{transfer_table('leave_domain_out','domain')}

La selezione C3 cambia usando esclusivamente gli altri due domini:

{selected_table}

Nel trasferimento fra generatori il pannello completo conserva più segnale. La media nasconde differenze importanti:

{transfer_table('leave_generator_out','generator')}

Gli intervalli per ciascun dominio e generatore sono conservati in `metrics.csv`. I sette generatori sono quelli storici del corpus; questi risultati non riguardano automaticamente modelli contemporanei o output di humanizer. In particolare, la shortlist fissa è poco separante su Claude e ChatGPT-turbo.

## Confronto con i pannelli storici

{reference_table}

Il pannello da 100 rimuove `n_words`, `n_characters`, `n_sentences`, `ttr`, `hapax_ratio`; quello da 93 rimuove anche le sette misure usate dal riferimento Wikipedia. Questi risultati sono ricalcolati sulle partizioni nuove, non copiati dalle vecchie AUC entro dominio. Il solo numero di parole rimane a **0,500 in ogni protocollo**, coerentemente con il matching esatto. Non basta però a eliminare tutti gli effetti della lunghezza o del contenuto.

## Conclusione utilizzabile nella tesi

> La rappresentazione compatta contiene informazione discriminante, ma non recupera tutta la prestazione del pannello completo. Una selezione di quattro caratteristiche limitata al training supera i riferimenti casuali medi, mentre la composizione delle famiglie influenza fortemente il confronto. MATTR è la caratteristica della shortlist storica più stabile rispetto alla nuova regola. Non è tuttavia dimostrata l'ottimalità o la validità indipendente dell'insieme fisso di quattro caratteristiche.

Le conclusioni restano sul riconoscimento umano/macchina in questo corpus. Numerali e nomi propri possono dipendere anche dall'argomento; il matching delle parole non garantisce contenuto equivalente. Una nuova conferma della shortlist richiede gruppi mai utilizzati per sceglierla. Non si sostituisce automaticamente la shortlist con le feature più frequenti dopo questa analisi e non si attribuisce alla loro AUC una misura di naturalezza o di riuscita della humanization.

## Codice, materiali e verifica

Il protocollo operativo è `configs/protocols/shortlist.json`. `python experiments/01_stylometry.py --analysis shortlist --mode compute` esegue i classificatori; `--mode reproduce` ricalcola tabelle, intervalli, grafici e questo report dalle predizioni salvate. Le implementazioni sono `humanization.evaluation.shortlist`, `humanization.evaluation.shortlist_statistics` e `humanization.evaluation.shortlist_reporting`.

Sono salvati: [metriche](metrics.csv), [differenze appaiate](paired_differences.csv), [frequenze di tutte le feature](selection_frequency.csv), [famiglie selezionate](selection_family_frequency.csv), [quaterne casuali](random_subsets.csv), [loro risultati](random_subset_metrics.csv), [distribuzioni casuali](random_reference_summary.csv), [selezioni per training](selections.csv), [coefficienti delle quattro feature](coefficients.csv), [audit delle partizioni](split_audit.csv) e predizioni numeriche compresse in `predictions.npz`.

Il controllo indipendente verifica 900 gruppi, cinque partizioni CV distinte, nessuna sovrapposizione training/test e nessun dominio o generatore escluso presente nel relativo training. Nove test sintetici controllano selezione, standardizzazione, ridondanza, partizioni e calcolo dell'AUC appaiata. Il CSV delle feature storico è invariato.

Nota di implementazione: il primo tentativo produceva partizioni identiche cambiando seme, perché ogni gruppo aveva la stessa composizione di classe. È escluso prima dell'analisi AUC. Si è corretta soltanto la randomizzazione dei codici dei gruppi; il protocollo v1.1 è fissato prima dei fit finali. La traccia è in [implementation_note.json](implementation_note.json). Nessuna scelta di feature, soglia o classificatore è stata cambiata sulla base delle prestazioni del primo tentativo.
"""
    text = text.replace(f"{manifest['n_fits']:,}", f"{manifest['n_fits']:,}".replace(",", "."))
    target = results / "report.md"
    target.write_text(text, encoding="utf-8")
    return target

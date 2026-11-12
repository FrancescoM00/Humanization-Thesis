"""Paired analysis of the 100-candidate robustness check for C3.

The 105-candidate predictions are reused exactly from the preceding experiment.
Bootstrap draws intentionally follow its order and cache keys, allowing the old
C3 AUCs and intervals to be checked without fitting that model again.
"""
from __future__ import annotations
from humanization.paths import project_root

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from humanization.paths import resolve_path

from humanization.stylometry.features import FEATURE_GROUPS, FEATURE_SCHEMA, LENGTH_SENSITIVE_FEATURES
from humanization.evaluation.shortlist_statistics import (
    HISTORICAL_FEATURES, PROTOCOL_STRATA, _auc_by_model, _group_positions,
    _quantiles, _sha256, _validate_documents, pairwise_auc_kernel,
    selection_frequency_table, weighted_auc_from_kernel,
)


CONFIGURATIONS = ("C3_105", "C3_100")
PROTOCOL_LABELS = {"cv": "Validazione ripetuta", "leave_domain_out": "Dominio escluso",
                   "leave_generator_out": "Generatore escluso"}


def selection_tables(parent_selections: pd.DataFrame, new_selections: pd.DataFrame,
                     candidate_features) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Selection counts are descriptive; overlapping training sets are not trials."""
    all_features = list(FEATURE_SCHEMA)
    parts, families = [], []
    for config, source in (("C3_105", parent_selections), ("C3_100", new_selections)):
        counts, family_counts = selection_frequency_table(source, all_features)
        counts.insert(0, "configuration", config)
        counts["candidate_eligible"] = counts.feature.isin(
            all_features if config == "C3_105" else candidate_features)
        if (counts.loc[~counts.candidate_eligible, "selection_count"] != 0).any():
            raise AssertionError("A feature outside the candidate pool was selected")
        family_counts.insert(0, "configuration", config)
        parts.append(counts)
        families.append(family_counts)
    frequencies = pd.concat(parts, ignore_index=True)
    comparison = pd.DataFrame({"feature": all_features})
    comparison["family"] = comparison.feature.map(FEATURE_GROUPS)
    comparison["candidate_100"] = comparison.feature.isin(candidate_features)
    comparison["historical_shortlist"] = comparison.feature.isin(HISTORICAL_FEATURES)
    for config in CONFIGURATIONS:
        counts = frequencies[frequencies.configuration == config]
        for protocol in PROTOCOL_STRATA:
            indexed = counts[counts.protocol == protocol].set_index("feature")
            comparison[f"{protocol}_{config}_count"] = comparison.feature.map(indexed.selection_count)
            comparison[f"{protocol}_n_fits"] = int(indexed.n_fits.iloc[0])
        comparison[f"total_{config}_count"] = sum(
            comparison[f"{protocol}_{config}_count"] for protocol in PROTOCOL_STRATA)
    comparison["newly_selected_in_100"] = (comparison.total_C3_100_count > 0) & (
        comparison.total_C3_105_count == 0)
    return frequencies, comparison, pd.concat(families, ignore_index=True)


def analyze_results(results: Path | str, protocol_path: Path | str | None = None,
                    root: Path | str | None = None) -> dict:
    """Compute exactly paired AUC differences using saved held-out predictions."""
    started = time.monotonic()
    results = Path(results).resolve()
    root = Path(root).resolve() if root else project_root()
    protocol_path = Path(protocol_path).resolve() if protocol_path else results / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    parent = Path(protocol["parent_results"])
    if not parent.is_absolute():
        parent = resolve_path(root, parent)
    parent = parent.resolve()
    documents = pd.read_csv(results / "documents.csv")
    parent_documents = pd.read_csv(parent / "documents.csv")
    pd.testing.assert_frame_equal(documents, parent_documents, check_exact=True)
    _validate_documents(documents)
    documents["_row"] = np.arange(len(documents))
    registry = pd.read_csv(results / "model_registry.csv")
    if registry.model.tolist() != list(CONFIGURATIONS):
        raise ValueError("Predictions require the registry order C3_105, C3_100")
    parent_registry = pd.read_csv(parent / "model_registry.csv")
    parent_index = parent_registry.index[parent_registry.model == "C3_training4"].tolist()
    if len(parent_index) != 1:
        raise ValueError("Cannot identify the original C3 predictions")
    candidates = list(protocol["candidate_features"])
    expected_candidates = [name for name in FEATURE_SCHEMA if name not in LENGTH_SENSITIVE_FEATURES]
    if candidates != expected_candidates:
        raise ValueError("The robustness pool must exclude exactly the five declared diagnostics")
    n_bootstrap = int(protocol.get("n_bootstrap", 2000))
    seed = int(protocol.get("bootstrap_seed", 20261112))
    if n_bootstrap < 2:
        raise ValueError("At least two bootstrap draws are required")
    parent_protocol = json.loads((parent / protocol.get("parent_protocol_file", "protocol.json")).read_text())
    if (n_bootstrap != int(parent_protocol["n_bootstrap"])
            or seed != int(parent_protocol["bootstrap_seed"])):
        raise ValueError("Use the same bootstrap settings as the parent experiment")
    labels = documents.label.to_numpy()
    rng = np.random.default_rng(seed)
    weight_cache = {}
    metric_rows, difference_rows, bootstrap_archive = [], [], {}

    def record_scope(protocol_name, scope, stratum, individual_auc, boots, n_groups, n_texts):
        points = individual_auc.mean(axis=0)
        for index, config in enumerate(CONFIGURATIONS):
            low, high = _quantiles(boots[:, index]) if boots is not None else (np.nan, np.nan)
            metric_rows.append({"protocol": protocol_name, "scope": scope, "stratum": stratum,
                                "configuration": config, "auc": float(points[index]),
                                "ci_low": low, "ci_high": high,
                                "repeat_auc_min": float(individual_auc[:, index].min()),
                                "repeat_auc_max": float(individual_auc[:, index].max()),
                                "n_repeats": individual_auc.shape[0], "n_models": 1,
                                "n_groups": n_groups, "n_texts": n_texts})
        low, high = _quantiles(boots[:, 1] - boots[:, 0]) if boots is not None else (np.nan, np.nan)
        difference_rows.append({"protocol": protocol_name, "scope": scope, "stratum": stratum,
                                "left_configuration": "C3_100", "right_configuration": "C3_105",
                                "auc_difference": float(points[1]-points[0]),
                                "ci_low": low, "ci_high": high,
                                "interval": "paired_source_bootstrap_fixed_predictions" if boots is not None else "not_computed"})
        if boots is not None:
            key = re.sub(r"[^a-zA-Z0-9_]+", "_", f"{protocol_name}__{scope}__{stratum}")
            bootstrap_archive[key] = boots

    with threadpool_limits(limits=int(protocol.get("threads", 2))):
        with (np.load(results / "predictions.npz", allow_pickle=False) as saved,
              np.load(parent / "predictions.npz", allow_pickle=False) as old_saved):
            for protocol_name, (column, scope) in PROTOCOL_STRATA.items():
                predictions = saved[protocol_name]
                old = old_saved[protocol_name][:, :, parent_index[0]]
                if (predictions.ndim != 3 or predictions.shape[1:] != (len(documents), 2)
                        or not np.isfinite(predictions).all()
                        or (predictions < 0).any() or (predictions > 1).any()):
                    raise ValueError(f"Invalid prediction cube for {protocol_name}")
                expected_repeats = len(protocol["cv_seeds"]) if protocol_name == "cv" else 1
                if predictions.shape[0] != expected_repeats:
                    raise ValueError("Unexpected repetition count")
                if not np.array_equal(predictions[:, :, 0], old):
                    raise AssertionError("The 105-candidate reference was changed or refitted")
                strata_auc, strata_boots = [], []
                for stratum in sorted(documents[column].unique()):
                    indices = np.flatnonzero(documents[column].eq(stratum).to_numpy())
                    groups, positive, negative = _group_positions(documents, indices)
                    n_groups = len(groups)
                    key = (column, str(stratum), groups)
                    if key not in weight_cache:
                        weight_cache[key] = rng.multinomial(
                            n_groups, np.full(n_groups, 1/n_groups), size=n_bootstrap).astype(float)
                    weights = weight_cache[key]
                    individual_auc = _auc_by_model(labels[indices], predictions[:, indices, :])
                    boots = np.empty((n_bootstrap, 2))
                    for index in range(2):
                        kernel = pairwise_auc_kernel(predictions[:, positive, index],
                                                     predictions[:, negative, index])
                        if not np.isclose(kernel.mean(), individual_auc[:, index].mean(), atol=1e-12, rtol=0):
                            raise AssertionError("Rank and kernel AUC disagree")
                        boots[:, index] = weighted_auc_from_kernel(kernel, weights)
                    record_scope(protocol_name, scope, str(stratum), individual_auc, boots,
                                 n_groups, len(indices))
                    strata_auc.append(individual_auc)
                    strata_boots.append(boots)
                    print(f"{protocol_name}: {stratum}; confronto appaiato completato", flush=True)
                record_scope(protocol_name, "macro", "ALL", np.mean(strata_auc, axis=0),
                             np.mean(strata_boots, axis=0), documents.group_id.nunique(), len(documents))
                record_scope(protocol_name, "overall", "ALL", _auc_by_model(labels, predictions),
                             None, documents.group_id.nunique(), len(documents))

    metrics = pd.DataFrame(metric_rows)
    differences = pd.DataFrame(difference_rows)
    old_metrics = pd.read_csv(parent / "metrics.csv")
    old_metrics = old_metrics[old_metrics.configuration == "C3_training4"]
    reused = metrics[metrics.configuration == "C3_105"]
    joined = reused.merge(old_metrics, on=["protocol", "scope", "stratum"], suffixes=("_new", "_parent"), validate="one_to_one")
    if len(joined) != len(reused) or len(joined) != len(old_metrics):
        raise AssertionError("Incomplete reproduction of parent C3 scopes")
    parent_errors = {}
    for field in ("auc", "ci_low", "ci_high"):
        a, b = joined[field+"_new"].to_numpy(), joined[field+"_parent"].to_numpy()
        if not np.allclose(a, b, atol=1e-12, rtol=0, equal_nan=True):
            raise AssertionError(f"Parent C3 {field} differs despite identical predictions")
        finite = np.isfinite(a) & np.isfinite(b)
        parent_errors[field] = float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.0
    parent_selections = parent / "selections.csv"
    expected_selections_hash = protocol.get("parent_selections_sha256")
    if expected_selections_hash and _sha256(parent_selections) != expected_selections_hash:
        raise ValueError("The saved parent selections differ from the frozen reference")
    frequencies, comparison, families = selection_tables(
        pd.read_csv(parent_selections), pd.read_csv(results / "selections.csv"), candidates)
    metrics.to_csv(results / "metrics.csv", index=False)
    differences.to_csv(results / "paired_differences.csv", index=False)
    frequencies.to_csv(results / "selection_frequency.csv", index=False)
    comparison.to_csv(results / "selection_comparison.csv", index=False)
    families.to_csv(results / "selection_family_frequency.csv", index=False)
    np.savez_compressed(results / "bootstrap_auc.npz", **bootstrap_archive)
    primary = metrics[(metrics.protocol == "cv") & (metrics.scope == "macro")]
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "primary_metrics": json.loads(primary.to_json(orient="records")),
               "macro_differences": json.loads(differences[differences.scope == "macro"].to_json(orient="records")),
               "parent_C3_reproduced": True, "max_abs_parent_metric_errors": parent_errors,
               "bootstrap_archive_columns": list(CONFIGURATIONS), "n_bootstrap": n_bootstrap,
               "bootstrap_seed": seed, "same_partitions_and_group_weights": True,
               "newly_selected_in_100": comparison.loc[comparison.newly_selected_in_100, "feature"].tolist(),
               "remaining_features_guaranteed_length_independent": False,
               "humanization_evaluated": False, "independent_dataset_confirmation": False,
               "interval_scope": "Pointwise exploratory, conditional on fixed held-out predictions; no bootstrap refitting",
               "elapsed_seconds": time.monotonic()-started}
    (results / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    manifest = {"created_at_utc": summary["created_at_utc"], "protocol_sha256": _sha256(protocol_path),
                "code_sha256": _sha256(Path(__file__)),
                "inputs": {name: _sha256(results / name) for name in
                           ("documents.csv", "model_registry.csv", "predictions.npz", "selections.csv")},
                "parent_metrics_sha256": _sha256(parent / "metrics.csv"),
                "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy")},
                "n_metrics": len(metrics), "n_differences": len(differences),
                "parent_C3_reproduced": True, "elapsed_seconds": summary["elapsed_seconds"]}
    (results / "statistics_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return summary


def _decimal(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


def _interval(row):
    return f"[{_decimal(row.ci_low)}; {_decimal(row.ci_high)}]"


def write_report(results: Path | str, protocol_path: Path | str | None = None) -> Path:
    """Export a concise Italian report and standalone figures from actual outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from humanization.evaluation.reporting import markdown_table

    results = Path(results)
    protocol = json.loads((Path(protocol_path) if protocol_path else results / "protocol.json").read_text())
    metrics = pd.read_csv(results / "metrics.csv")
    differences = pd.read_csv(results / "paired_differences.csv")
    comparison = pd.read_csv(results / "selection_comparison.csv")
    selections = pd.read_csv(results / "selections.csv")
    fit = json.loads((results / protocol.get("fit_manifest_file", "manifest.json")).read_text())
    stats = json.loads((results / "summary.json").read_text())
    def metric(p, config):
        return metrics[(metrics.protocol == p) & (metrics.scope == "macro") &
                       (metrics.configuration == config)].iloc[0]
    def delta(p):
        return differences[(differences.protocol == p) & (differences.scope == "macro")].iloc[0]
    def frequency(feature, p, config):
        row = comparison[comparison.feature == feature].iloc[0]
        if config == "C3_100" and not row.candidate_100:
            return "esclusa"
        return f"{row[p+'_'+config+'_count']}/{row[p+'_n_fits']}"
    def auc_table(p, scope):
        part = metrics[(metrics.protocol == p) & (metrics.scope == scope)]
        rows = []
        for stratum, group in part.groupby("stratum"):
            a = group[group.configuration == "C3_105"].iloc[0]
            b = group[group.configuration == "C3_100"].iloc[0]
            d = differences[(differences.protocol == p) & (differences.scope == scope) &
                            (differences.stratum == stratum)].iloc[0]
            rows.append([stratum, _decimal(a.auc), _decimal(b.auc), _decimal(d.auc_difference), _interval(d)])
        return markdown_table(pd.DataFrame(rows, columns=["Test", "105 candidati", "100 candidati", "Differenza", "IC 95% differenza"]))

    macro_rows = []
    for p, label in PROTOCOL_LABELS.items():
        a, b, d = metric(p, "C3_105"), metric(p, "C3_100"), delta(p)
        macro_rows.append([label, _decimal(a.auc), _decimal(b.auc), _decimal(d.auc_difference), _interval(d)])
    macro_table = markdown_table(pd.DataFrame(macro_rows,
        columns=["Protocollo", "C3: 105 candidati", "C3: 100 candidati", "100 meno 105", "IC 95% differenza"]))
    cv_table = markdown_table(pd.DataFrame([
        [config, _decimal(metric("cv", config).auc), _interval(metric("cv", config))]
        for config in CONFIGURATIONS], columns=["Selezione di quattro feature", "AUC CV macro", "IC 95% per gruppo"]))
    count_names = comparison[(comparison.total_C3_100_count > 0) | (comparison.total_C3_105_count > 0)].copy()
    count_names = count_names.sort_values(["total_C3_100_count", "total_C3_105_count", "feature"], ascending=[False, False, True])
    count_names = count_names.feature.tolist()
    for feature in HISTORICAL_FEATURES:
        if feature not in count_names:
            count_names.append(feature)
    rows = [[feature, *[f"{frequency(feature,p,'C3_105')}; {frequency(feature,p,'C3_100')}"
                       for p in PROTOCOL_LABELS]] for feature in count_names]
    frequency_table = markdown_table(pd.DataFrame(rows,
        columns=["Feature", "CV: 105; 100", "Domini: 105; 100", "Generatori: 105; 100"]))
    chosen = selections[selections.selected.astype(str).str.lower().eq("true")]
    transfer_rows = []
    for held_out, part in chosen[chosen.protocol == "leave_domain_out"].groupby("held_out"):
        transfer_rows.append([held_out, ", ".join(f"`{x}`" for x in part.sort_values("rank").feature)])
    chosen_table = markdown_table(pd.DataFrame(transfer_rows, columns=["Dominio escluso", "Quattro scelte dai 100 candidati nel training"]))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=True)
    for ax, (p, title) in zip(axes, PROTOCOL_LABELS.items()):
        for i, config in enumerate(CONFIGURATIONS):
            row = metric(p, config)
            ax.errorbar(row.auc, i, xerr=[[row.auc-row.ci_low], [row.ci_high-row.auc]],
                        fmt="o", color=("#22577a", "#21816b")[i], capsize=3)
            ax.text(row.ci_high+.006, i+.1, f"{row.auc:.3f}", fontsize=9)
        ax.axvline(.5, color="gray", ls="--", lw=.8)
        ax.set_xlim(.48, .93)
        ax.set_ylim(1.6, -.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("AUC macro e IC 95% per gruppo", fontsize=9)
        ax.grid(axis="x", alpha=.15)
    axes[0].set_yticks([0, 1], ["C3: 105 candidati", "C3: 100 candidati"], fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(results / f"auc_comparison.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    plot_names = count_names[:12]
    for feature in HISTORICAL_FEATURES:
        if feature not in plot_names:
            plot_names.append(feature)
    indexed = comparison.set_index("feature").loc[plot_names]
    fig, ax = plt.subplots(figsize=(8.2, max(3.8, .30*len(plot_names))))
    y = np.arange(len(plot_names))
    for offset, config, color in ((-.18,"C3_105","#22577a"),(.18,"C3_100","#21816b")):
        rate = indexed[f"cv_{config}_count"] / indexed.cv_n_fits
        ax.barh(y+offset, rate, height=.33, color=color, label=config.replace("_", ": ")+" candidati")
    ax.set_yticks(y, plot_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Frequenza di selezione nei 25 training CV")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(results / f"selection_comparison.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    cv_delta = delta("cv")
    if cv_delta.ci_high < 0:
        interpretation = "Nel confronto principale l'esclusione dei cinque indicatori riduce la separazione: l'intervallo della differenza è interamente negativo."
    elif cv_delta.ci_low > 0:
        interpretation = "Nel confronto principale l'esclusione dei cinque indicatori aumenta la separazione: l'intervallo della differenza è interamente positivo."
    else:
        interpretation = "Nel confronto principale l'intervallo della differenza comprende zero: non emerge una direzione precisa del cambiamento, senza che questo dimostri equivalenza."
    mattr = comparison[comparison.feature == "mattr_50"].iloc[0]
    mattr_total = int(mattr.total_C3_100_count)
    new_names = stats["newly_selected_in_100"]
    new_text = ", ".join(f"`{name}`" for name in new_names) if new_names else "nessuna caratteristica assente dalle selezioni precedenti"
    fit_count = fit.get("n_fits", fit.get("n_new_fits", 35))
    text = f"""# Robustezza di C3: selezione da 100 candidati senza cinque indicatori diagnostici

**Esperimento aggiuntivo dell'11 novembre 2026.** Si confronta la procedura di selezione delle feature, non la qualità della humanization.

## Risposta alla domanda

Selezionando quattro feature dai 100 candidati si ottiene AUC CV macro **{_decimal(metric('cv','C3_100').auc)}**, rispetto a **{_decimal(metric('cv','C3_105').auc)}** con i 105 candidati. La differenza appaiata, 100 meno 105, è **{_decimal(cv_delta.auc_difference)}**, IC 95% **{_interval(cv_delta)}**. {interpretation}

`mattr_50` viene selezionata **{mattr_total}/{fit_count}** volte con il pannello ridotto. L'esclusione impedisce a TTR di occupare uno dei quattro posti e permette di osservare quali caratteristiche la sostituiscono. Le feature selezionate soltanto nel nuovo controllo sono: {new_text}. Non si promuove automaticamente una nuova shortlist sulla base di questi esiti.

## Metodo: cambia soltanto l'insieme candidato

Si riutilizzano gli stessi **1.800 testi, 900 gruppi**, con numero di parole lessicali abbinato fra umano e macchina. Gli abbinamenti non sono necessariamente parafrasi dello stesso contenuto. Le partizioni sono caricate dal file già salvato nell'esperimento precedente, senza rigenerarle: 25 training della CV ripetuta, tre con un dominio escluso e sette con un generatore escluso.

Il pannello ridotto esclude esattamente `n_words`, `n_characters`, `n_sentences`, `ttr` e `hapax_ratio`. Le altre 100 colonne rimangono candidate. In ogni training si ordinano le feature per valore assoluto del g di Hedges e si selezionano progressivamente quattro colonne, scartando correlazioni assolute di Spearman almeno 0,85 con quelle già scelte. Calcolo degli effetti, correlazioni, imputazione con mediana e standardizzazione usano soltanto il training.

**Entrambi i classificatori usano soltanto quattro feature:** 105 e 100 indicano quante colonne sono disponibili per sceglierle, non quante entrano nel modello finale. Il riferimento è quindi C3 con 105 candidati, non C1 che utilizza direttamente tutte le 105 feature.

La regressione logistica resta quella precedente: C=1, regolarizzazione L2, classi bilanciate, solver lbfgs, nessuna ottimizzazione degli iperparametri. Sono eseguiti soltanto **{fit_count} nuovi fit** per C3 con 100 candidati. Le predizioni del riferimento da 105 sono riprese esattamente dal run precedente; AUC e intervalli del riferimento sono riprodotti entro la tolleranza numerica di 10^-12. Nessuna generazione di testi o chiamata API.

La misura principale è la media delle AUC dei tre domini, ulteriormente mediata fra le cinque ripetizioni. Nel leave-one-generator-out si mediano le sette AUC. Gli intervalli usano gli stessi 2.000 bootstrap appaiati per gruppo, mantenendo insieme tutte le ripetizioni e usando le stesse estrazioni nei due modelli. Sono intervalli puntuali ed esplorativi, condizionati alle predizioni salvate: non comprendono tutta l'incertezza del riaddestramento. I 35 training si sovrappongono e non sono prove indipendenti.

## Prestazione e differenze appaiate

{cv_table}

{macro_table}

Le differenze sono espresse in unità di AUC, non in percentuali di accuratezza. Non è stato fissato un margine di non inferiorità: un intervallo che comprende zero non dimostra equivalenza.

Il confronto appaiato considera gli stessi testi e gli stessi training: l'intervallo della differenza non si deduce dal semplice sovrapporsi degli intervalli delle due AUC. In questo campione le tre differenze macro sono negative. Il segnale rimane sopra il caso, ma non si conserva interamente la prestazione precedente.

![AUC macro e intervalli per gruppo, usando le stesse partizioni e gli stessi ricampionamenti.](auc_comparison.png)

## Stabilità della selezione

Nelle celle della tabella si confrontano **conteggio/numero di training con 105 candidati**, poi, dopo il punto e virgola, **conteggio/numero di training con 100 candidati**. La dicitura “esclusa” indica non ammissibilità, non assenza di informazione.

{frequency_table}

Le frequenze descrivono la regola di selezione, non l'importanza causale o l'utilità individuale di una feature. Caratteristiche correlate possono sostituirsi. MATTR misura la varietà lessicale su finestre di 50 parole; la punteggiatura totale riassume i segni per 100 parole; `fw_and` misura la frequenza di “and”. La shortlist storica di quattro nomi rimane distinta dalle quaterne selezionate nel training.

![Frequenze di selezione nei training CV, prima e dopo l'esclusione delle cinque colonne.](selection_comparison.png)

## Trasferimento a domini e generatori esclusi

Il dominio o generatore indicato nella tabella è assente dal training. Le AUC dei due modelli e la differenza sono calcolate sugli stessi testi.

{auc_table('leave_domain_out','domain')}

Il calo fra domini è concentrato soprattutto su **Essay: AUC da 0,697 a 0,565**. Reuters cambia poco, mentre WritingPrompts cala meno di Essay. La stabilità di selezione di MATTR non garantisce quindi, da sola, una buona generalizzazione del classificatore a ogni dominio.

Le quaterne selezionate dai 100 candidati sui soli due domini di training sono:

{chosen_table}

{auc_table('leave_generator_out','generator')}

Le AUC CV dei singoli domini, con training che contiene tutti e tre i domini e gruppi di test separati, sono:

{auc_table('cv','domain')}

## Interpretazione e limiti

Il confronto quantifica la dipendenza della selezione dall'ammissione dei cinque indicatori diagnostici. {interpretation} Il comportamento sui domini e sui generatori va letto separatamente dalla media: non si presume un vantaggio uniforme in tutti i casi.

**Rimuovere queste cinque colonne non rende le altre automaticamente indipendenti dalla lunghezza.** MATTR riduce la dipendenza del semplice rapporto tipi/token, ma anche misure normalizzate o basate su finestre possono cambiare con la quantità di testo e con la sua composizione. Inoltre, eliminare cinque colonne insieme non identifica un effetto causale separato di ciascuna; per la regola usata, le selezioni precedenti mostrano quale di esse occupasse posti nella quaterna.

Rimangono anche caratteristiche della struttura frasale, come `sentence_length_cv`, che passa da 4/25 a 12/25 selezioni nella CV. Questo controllo esclude i cinque indicatori dichiarati, non tutte le misure relative alla lunghezza delle frasi. Fra le cinque colonne rimosse, soltanto TTR era stata selezionata nel confronto precedente, in tutti i 35 training.

Un eventuale calo dopo l'esclusione di TTR non dimostra che il segnale precedente fosse un effetto spurio della lunghezza: il numero di parole era già abbinato, e TTR può riflettere anche differenze reali nella ripetizione e varietà del lessico.

Il matching delle parole e la selezione sul solo training non eliminano differenze di argomento o artefatti del corpus. La verifica rimane successiva all'esplorazione dei dati e non costituisce una conferma indipendente di una lista fissa. Non misura naturalezza, fedeltà semantica o riuscita di uno humanizer. Il suo risultato riguarda la robustezza di una procedura predittiva su questo campione.

## Materiali e riproduzione

Il [protocollo](protocol.json) identifica dati, partizioni e riferimento precedente. Sono disponibili [AUC e intervalli](metrics.csv), [differenze appaiate](paired_differences.csv), [frequenze complete](selection_frequency.csv), [confronto fra selezioni](selection_comparison.csv), [selezioni per training](selections.csv), [audit delle partizioni](split_audit.csv), [manifest dei fit]({protocol.get("fit_manifest_file", "manifest.json")}) e [manifest statistico](statistics_manifest.json). Le predizioni complete e le AUC bootstrap sono salvate nei due file NPZ.

`python experiments/01_stylometry.py --analysis shortlist_100 --mode compute` esegue i soli nuovi classificatori. `--mode reproduce` ricalcola tabelle, intervalli, figure e questo report dalle predizioni salvate senza ripetere i fit. Il protocollo operativo è `configs/protocols/shortlist_robustness.json`; le implementazioni sono `humanization.evaluation.shortlist_robustness` e `humanization.evaluation.shortlist_robustness_analysis`.
"""
    path = results / "report.md"
    path.write_text(text, encoding="utf-8")
    return path

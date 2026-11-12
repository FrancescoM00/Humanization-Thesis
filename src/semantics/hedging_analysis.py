"""Frozen-threshold evaluation of a dedicated hedging stress test.

The primary score is the historical bidirectional document minimum. Its
support threshold is never fitted here. Rule intent and assistant review are
kept separate from independent human labels; AUC always uses alteration risk
``-support`` and is not flipped after observing a value below one half.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from semantics.aggregator_analysis import _auc_sorted
from stylometry.report import markdown_table


HISTORICAL_THRESHOLD = 0.040749866515398
CONDITIONS = ("hedging_complete", "hedging_partial", "hedging_preserved")
DISPLAY = {"hedging_complete": "Rimozione completa", "hedging_partial": "Rafforzamento parziale",
           "hedging_preserved": "Parafrasi conservativa"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_panel(pairs_path: Path | str, scores_path: Path | str) -> pd.DataFrame:
    rows = []
    for line in Path(pairs_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        metadata = pair.get("metadata", {})
        condition = pair.get("condition", pair.get("perturbation"))
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown hedging condition {condition!r}")
        group = pair.get("source_group_id") or pair.get("source_group") or pair.get("source_id")
        if not group:
            raise ValueError("Every pair requires its original source group")
        intended = int(condition != "hedging_preserved")
        if "label" in pair and int(pair["label"]) != intended:
            raise ValueError("Pair label disagrees with the declared transformation intent")
        rows.append({"id": pair["id"], "source_group": str(group), "condition": condition,
                     "is_altered": intended, "domain": pair.get("domain", "unknown"),
                     "generator": pair.get("generator", "unknown"),
                     "rule": metadata.get("rule", ""),
                     "original_words": len(pair["original"].split()),
                     "modified_words": len(pair["modified"].split()),
                     "target_original": pair.get("target_original", ""),
                     "target_modified": pair.get("target_modified", "")})
    frame = pd.DataFrame(rows)
    if frame.empty or frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError("Pairs need nonmissing, unique IDs")
    if frame.duplicated(["source_group", "condition"]).any():
        raise ValueError("At most one pair of each condition is allowed per original")
    scores = pd.read_csv(scores_path)
    if "id" not in scores or scores.id.duplicated().any():
        raise ValueError("Score IDs must be unique")
    # A frozen numerical threshold is meaningful only for its old score.
    for column, expected in (("nli_retrieval_mode", "top_k_cosine"),
                             ("nli_retrieval_top_k", 3), ("nli_max_pair_tokens", 384)):
        if column in scores and not scores[column].eq(expected).all():
            raise ValueError(f"Historical threshold cannot be transferred to different {column}")
    # Accommodate the historical scorer names and an explicit equivalent alias.
    for prefix in ("nli_forward", "nli_reverse", "local_nli_forward", "local_nli_reverse"):
        if prefix+"_min" not in scores and prefix+"_entailment_min" in scores:
            scores[prefix+"_min"] = scores[prefix+"_entailment_min"]
    score_columns = [name for name in scores if name.startswith(("nli_", "local_nli_"))]
    frame = frame.merge(scores[["id", *score_columns]], on="id", how="left", validate="one_to_one")
    required = ["nli_forward_min", "nli_reverse_min"]
    if any(column not in frame for column in required):
        raise ValueError("Historical document forward/reverse minimum scores are required")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not frame[column].between(0, 1).all():
            raise ValueError(f"Incomplete or invalid primary scores: {column}")
    frame["document_support"] = frame[required].min(axis=1, skipna=False)
    local = ["local_nli_forward_min", "local_nli_reverse_min"]
    if all(column in frame for column in local):
        values = frame[local].apply(pd.to_numeric, errors="coerce")
        if ((values < 0) | (values > 1)).any().any():
            raise ValueError("Local support must be in [0,1] when available")
        frame["local_support"] = values.min(axis=1, skipna=False)
    return frame


def load_review(path: Path | str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["id", "validity", "reviewer_source", "notes"])
    review = pd.read_csv(path).fillna("")
    aliases = {"pair_id": "id", "review_status": "validity", "review_validity": "validity",
               "label_source": "reviewer_source", "reviewer_type": "reviewer_source"}
    review.rename(columns={old: new for old, new in aliases.items() if old in review and new not in review}, inplace=True)
    if "id" not in review or "validity" not in review:
        raise ValueError("Review requires id and validity fields")
    if "reviewer_source" not in review:
        # Unspecified provenance must not be silently promoted to human.
        review["reviewer_source"] = "unspecified"
    review["reviewer_source"] = review.reviewer_source.astype(str).str.lower()
    review["validity"] = review.validity.astype(str).str.lower()
    aliases = {"clear": "valid", "uncertain": "qualified", "ambiguous": "qualified",
               "not_valid": "invalid", "unreviewed": ""}
    review["validity"] = review.validity.replace(aliases)
    if not set(review.validity).issubset({"", "valid", "qualified", "invalid"}):
        raise ValueError("Review validity must be valid, qualified, invalid or empty")
    if review.duplicated(["id", "reviewer_source"]).any():
        raise ValueError("Consolidate review disagreements before producing one row per pair and provenance")
    return review


def _matched_cohort(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    negatives = frame[frame.is_altered == 1]
    if kind != "all":
        negatives = negatives[negatives.condition == "hedging_"+kind]
    positives = frame[frame.is_altered == 0]
    sources = set(negatives.source_group) & set(positives.source_group)
    return pd.concat([positives[positives.source_group.isin(sources)],
                      negatives[negatives.source_group.isin(sources)]], ignore_index=True)


def _source_weights(frame):
    return 1 / frame.groupby(["source_group", "is_altered"]).id.transform("count").to_numpy()


def wilson95(successes: int, total: int) -> tuple[float, float]:
    """Secondary binomial interval, used only for one observation per source."""
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("Wilson counts require 0 <= successes <= total and total > 0")
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z*z / total
    center = (rate + z*z/(2*total)) / denominator
    half = z * np.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / denominator
    low = 0.0 if successes == 0 else max(0.0, float(center-half))
    high = 1.0 if successes == total else min(1.0, float(center+half))
    return low, high


def grouped_metrics(frame: pd.DataFrame, score_column: str, threshold: float | None,
                    rng: np.random.Generator, n_bootstrap=2000, weight_cache=None):
    """Source-weighted AUC and operating point with paired cluster bootstrap."""
    part = frame[np.isfinite(pd.to_numeric(frame[score_column], errors="coerce"))].copy()
    # For the local diagnostics, missing units must not create unmatched corpora.
    part = _matched_cohort(part, "all")
    labels = part.is_altered.to_numpy(dtype=int)
    n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
    result = {"n_pairs": len(part), "n_preserved": n0, "n_altered": n1,
              "n_sources": part.source_group.nunique(), "n_input_pairs": len(frame),
              "n_missing_or_unmatched_pairs": len(frame)-len(part)}
    if not n0 or not n1:
        return {**result, "auc": np.nan, "uncertainty_note": "both matched classes required"}, {}
    support = part[score_column].to_numpy(dtype=float)
    risk = -support
    weights = _source_weights(part)
    groups = sorted(part.source_group.unique())
    group_index = {group: i for i, group in enumerate(groups)}
    codes = part.source_group.map(group_index).to_numpy(dtype=int)
    alerts = support < threshold if threshold is not None else None
    result.update(auc=_auc_sorted(labels, risk, weights), threshold_support=threshold,
                  score_orientation="alteration_risk=-bidirectional_minimum_support",
                  alert_rule="support < threshold" if threshold is not None else "not calibrated for this diagnostic")
    if alerts is not None:
        sensitivity = float(np.average(alerts[labels == 1], weights=weights[labels == 1]))
        fpr = float(np.average(alerts[labels == 0], weights=weights[labels == 0]))
        result.update(sensitivity=sensitivity, false_positive_rate=fpr, specificity=1-fpr,
                      true_positive_count=int(alerts[labels == 1].sum()), false_positive_count=int(alerts[labels == 0].sum()),
                      false_negative_count=int((~alerts[labels == 1]).sum()),
                      true_negative_count=int((~alerts[labels == 0]).sum()),
                      sensitivity_raw=float(alerts[labels == 1].mean()), false_positive_rate_raw=float(alerts[labels == 0].mean()))
        for label, name, successful in ((1, "sensitivity", result["true_positive_count"]),
                                        (0, "false_positive_rate", result["false_positive_count"])):
            class_part = part.loc[part.is_altered == label]
            eligible = class_part.groupby("source_group").size().eq(1).all()
            result[name+"_wilson95_eligible"] = bool(eligible)
            if eligible:
                low, high = wilson95(successful, len(class_part))
                result[name+"_wilson95_low"] = low
                result[name+"_wilson95_high"] = high
                result[name+"_wilson95_n_sources"] = len(class_part)
            else:
                result[name+"_wilson95_low"] = np.nan
                result[name+"_wilson95_high"] = np.nan
                result[name+"_wilson95_n_sources"] = np.nan
        result["wilson95_is_secondary"] = True
    if len(groups) < 5:
        return {**result, "uncertainty_note": "fewer than five matched original sources"}, {}
    cache = weight_cache if weight_cache is not None else {}
    key = tuple(groups)
    if key not in cache:
        cache[key] = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=n_bootstrap)
    draws = cache[key][:, codes] * weights
    order = np.argsort(risk, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(risk[order]))+1]
    positive = np.add.reduceat(draws[:, order]*labels[order], starts, axis=1)
    negative = np.add.reduceat(draws[:, order]*(1-labels[order]), starts, axis=1)
    denominator = positive.sum(axis=1)*negative.sum(axis=1)
    auc = (positive*(np.cumsum(negative, axis=1)-.5*negative)).sum(axis=1)/denominator
    values = {"auc": auc}
    if alerts is not None:
        sens = (draws[:, labels == 1]*alerts[labels == 1]).sum(axis=1)/draws[:, labels == 1].sum(axis=1)
        fpr = (draws[:, labels == 0]*alerts[labels == 0]).sum(axis=1)/draws[:, labels == 0].sum(axis=1)
        values.update(sensitivity=sens, false_positive_rate=fpr, specificity=1-fpr)
    for metric, draw in values.items():
        result[metric+"_ci_low"], result[metric+"_ci_high"] = map(float, np.quantile(draw, [.025, .975]))
    result.update(valid_bootstrap_draws=n_bootstrap, uncertainty_note="pointwise source-bootstrap conditional on frozen scores")
    return result, values


def analyze(pairs_path: Path | str, scores_path: Path | str, output: Path | str,
            protocol_path: Path | str, review_path: Path | str | None = None) -> dict:
    started = time.monotonic()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    threshold = float(protocol.get("threshold_support", protocol.get("historical_threshold_support", HISTORICAL_THRESHOLD)))
    if abs(threshold-HISTORICAL_THRESHOLD) > 1e-14:
        raise ValueError("The historical support threshold must remain unchanged")
    seed = int(protocol.get("bootstrap_seed", 20261111))
    n_bootstrap = int(protocol.get("n_bootstrap", 2000))
    if n_bootstrap < 2:
        raise ValueError("At least two bootstrap draws are required")
    frame = load_panel(pairs_path, scores_path)
    review = load_review(review_path)
    if not set(review.id).issubset(set(frame.id)):
        raise ValueError("Review contains IDs outside the frozen panel")
    frame["document_alert"] = frame.document_support < threshold
    frame.to_csv(output / "pair_scores.csv", index=False)
    panels = {"rule_intended_full": frame}
    for source, part in review[review.validity.ne("")].groupby("reviewer_source", sort=True):
        for suffix, accepted in (("valid", {"valid"}), ("valid_qualified", {"valid", "qualified"})):
            ids = set(part.loc[part.validity.isin(accepted), "id"])
            panels[f"{source}_review_{suffix}"] = frame[frame.id.isin(ids)]
    rng = np.random.default_rng(seed); weights = {}
    rows, saved_draws = [], {}
    for panel_name, panel in panels.items():
        for scope in ("all", "complete", "partial"):
            cohort = _matched_cohort(panel, scope)
            for metric, column in (("document_min", "document_support"), ("local_min", "local_support")):
                if column not in cohort or cohort.empty:
                    continue
                result, draws = grouped_metrics(cohort, column, threshold if metric == "document_min" else None,
                                                 rng, n_bootstrap, weights)
                rows.append({"panel": panel_name, "scope": scope, "metric": metric, **result})
                for name, values in draws.items():
                    key = re.sub(r"[^A-Za-z0-9_]+", "_", f"{panel_name}__{scope}__{metric}__{name}")
                    saved_draws[key] = values
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError("No matched evaluation cohort was available")
    metrics.to_csv(output / "metrics.csv", index=False)
    np.savez_compressed(output / "bootstrap_metrics.npz", **saved_draws)
    diagnostic_columns = [name for name in frame if name.startswith(("nli_", "local_nli_"))
                          and pd.api.types.is_numeric_dtype(frame[name])]
    diagnostic_columns += ["document_support"] + (["local_support"] if "local_support" in frame else [])
    diagnostics = frame.groupby("condition")[diagnostic_columns].agg(["mean", "median", "min", "max"])
    diagnostics.columns = [f"{a}__{b}" for a, b in diagnostics.columns]
    diagnostics.reset_index().to_csv(output / "directional_summary.csv", index=False)
    if not review.empty:
        review.groupby(["reviewer_source", "validity"]).size().rename("n_pairs").reset_index().to_csv(output / "review_summary.csv", index=False)
        review.to_csv(output / "review_used.csv", index=False)
    primary = metrics[(metrics.panel == "rule_intended_full") & (metrics.metric == "document_min")]
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "threshold_support": threshold,
               "threshold_recalibrated": False, "n_pairs": len(frame), "n_sources": frame.source_group.nunique(),
               "condition_counts": frame.condition.value_counts().to_dict(),
               "primary_results": json.loads(primary.to_json(orient="records", double_precision=15)),
               "review_sources": sorted(review.reviewer_source.unique().tolist()),
               "independent_human_validation_available": bool((review.reviewer_source.eq("human") & review.validity.ne("")).any()),
               "score_orientation": "risk=-support, higher means more altered",
               "local_diagnostic_uses_historical_threshold": False,
               "secondary_rate_intervals": "Wilson 95% only for classes with exactly one observation per original; never pooled sensitivity with two alterations per original or AUC",
               "n_bootstrap": n_bootstrap, "bootstrap_seed": seed,
               "confidence_intervals": "pointwise source-cluster bootstrap; all transformations and controls of an original travel together",
               "elapsed_seconds": time.monotonic()-started}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    manifest = {"created_at_utc": summary["created_at_utc"], "protocol_sha256": _sha(protocol_path),
                "pairs_sha256": _sha(pairs_path), "scores_sha256": _sha(scores_path),
                "review_sha256": _sha(review_path) if review_path else None,
                "code_sha256": _sha(__file__), "n_metric_rows": len(metrics),
                "threshold_recalibrated": False, "elapsed_seconds": summary["elapsed_seconds"]}
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return summary


def _decimal(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",") if pd.notna(value) else "non disponibile"


def _ci(row, name, digits=3):
    return f"[{_decimal(row.get(name+'_ci_low'),digits)}; {_decimal(row.get(name+'_ci_high'),digits)}]"


def write_report(output: Path | str) -> Path:
    output = Path(output)
    metrics = pd.read_csv(output / "metrics.csv")
    frame = pd.read_csv(output / "pair_scores.csv")
    summary = json.loads((output / "summary.json").read_text())
    report_date=datetime.fromisoformat(summary['created_at_utc']).astimezone(ZoneInfo('Europe/Rome')).strftime('%d/%m/%Y')
    preparation_path=output/'preparation_manifest.json'
    preparation=json.loads(preparation_path.read_text()) if preparation_path.exists() else {}
    source_table=markdown_table(pd.DataFrame(
        [[domain,count] for domain,count in preparation.get('sources_per_domain',{}).items()],
        columns=['Dominio','Originali indipendenti'])) if preparation else ''
    primary = metrics[(metrics.panel == "rule_intended_full") & (metrics.metric == "document_min")]
    labels = {"all": "Tutte le alterazioni", "complete": "Rimozione completa", "partial": "Rafforzamento parziale"}
    rows = []
    for row in primary.to_dict("records"):
        rows.append([labels[row["scope"]], f"{row['n_altered']}/{row['n_preserved']}",
                     _decimal(row["auc"]), _ci(row,"auc"),
                     _decimal(100*row.get("sensitivity",np.nan),1)+"%", _ci({k:100*v if k.startswith("sensitivity_ci_") else v for k,v in row.items()},"sensitivity",1),
                     _decimal(100*row.get("false_positive_rate",np.nan),1)+"%"])
    main_table = markdown_table(pd.DataFrame(rows, columns=["Confronto", "Alterati / controlli", "AUC", "IC 95% AUC", "Sensibilità", "IC sensibilità (%)", "Falsi allarmi"]))
    local = metrics[(metrics.panel == "rule_intended_full") & (metrics.metric == "local_min")]
    local_table = markdown_table(pd.DataFrame([[labels[row.scope],_decimal(row.auc),_ci(row._asdict(),"auc"),row.n_sources]
                        for row in local.itertuples(index=False)], columns=["Confronto locale", "AUC", "IC 95%", "Originali"])) if len(local) else "Punteggi locali non disponibili."
    operating_rows=[]
    for row in primary.to_dict("records"):
        operating_rows.append([labels[row['scope']],f"{row.get('true_positive_count',0):.0f}/{row['n_altered']}",
            f"{row.get('false_positive_count',0):.0f}/{row['n_preserved']}",
            _decimal(100*row.get('false_positive_rate',np.nan),1)+"%",
            _ci({k:100*v if k.startswith('false_positive_rate_ci_') else v for k,v in row.items()},'false_positive_rate',1),
            _decimal(100*row.get('specificity',np.nan),1)+"%"])
    operating_table=markdown_table(pd.DataFrame(operating_rows,columns=["Confronto","Allarmi alterati","Allarmi conservativi","FPR per fonte","IC FPR (%)","Specificità"]))
    wilson_rows=[]
    for row in primary.to_dict("records"):
        def wilson_cell(name):
            if not row.get(name+"_wilson95_eligible",False):
                return "non applicabile: più varianti per fonte"
            return f"[{_decimal(100*row[name+'_wilson95_low'],1)}; {_decimal(100*row[name+'_wilson95_high'],1)}]"
        wilson_rows.append([labels[row['scope']],wilson_cell('sensitivity'),wilson_cell('false_positive_rate')])
    wilson_table=markdown_table(pd.DataFrame(wilson_rows,columns=['Confronto','Wilson 95% sensibilità (%)','Wilson 95% falsi allarmi (%)']))
    review_text="Non è disponibile una revisione testuale associata al pannello. Le etichette descrivono l'intento delle regole."
    if (output/'review_summary.csv').exists():
        review=pd.read_csv(output/'review_summary.csv')
        review_text=markdown_table(review.rename(columns={'reviewer_source':'Provenienza revisione','validity':'Giudizio','n_pairs':'Coppie'}))
        review_text+="\n\nUna revisione dell'assistente non è annotazione umana indipendente. Le classi del pannello completo rimangono etichette di intento; i sottoinsiemi revisionati sono analisi secondarie, non nuove etichette umane."
        used=pd.read_csv(output/'review_used.csv')
        detail=frame[['id','condition']].merge(used[['id','validity']],on='id')
        detail=detail.groupby(['condition','validity']).size().unstack(fill_value=0)
        detail.index=detail.index.map(DISPLAY)
        review_text+='\n\n'+markdown_table(detail.reset_index().rename(columns={
            'condition':'Condizione','valid':'Chiaramente validi','qualified':'Qualificati','invalid':'Non validi'}))
        counts=used.validity.value_counts()
        review_text+=f'\n\n“Qualificato” indica un dubbio concreto sul valore epistemico, sul tempo narrativo, sulla portata della cautela o sulla ridondanza rispetto al contesto; non è sinonimo di significato certamente alterato. Nel pannello finale {counts.get("valid",0)} coppie sono chiaramente valide e {counts.get("qualified",0)} qualificate. Le {preparation.get("n_replaced_sources",0)} triplette con errori evidenti sono state sostituite prima delle inferenze, conservando originali e motivazioni in `pre_review/` e `replacements.csv`. Non è stata effettuata alcuna selezione in base agli score.'
    reviewed=metrics[(metrics.panel!='rule_intended_full') & (metrics.metric=='document_min')]
    review_names={'assistant_review_valid':'Assistente: validi',
                  'assistant_review_valid_qualified':'Assistente: validi e qualificati'}
    review_table=markdown_table(pd.DataFrame([[review_names.get(row.panel,row.panel),labels[row.scope],row.n_sources,_decimal(row.auc),_decimal(100*getattr(row,'sensitivity',np.nan),1)+"%"]
                     for row in reviewed.itertuples(index=False)],columns=['Sottoinsieme','Confronto','Originali','AUC','Sensibilità'])) if len(reviewed) else "Non ci sono controlli conservativi e alterazioni entrambi revisionati sufficienti per un confronto appaiato."
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conditions=list(CONDITIONS)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,column,title in ((axes[0],'document_support','Minimo bidirezionale sul documento'),(axes[1],'local_support','Minimo bidirezionale sulla frase modificata')):
        if column not in frame:
            ax.axis('off');continue
        values=[frame.loc[frame.condition==condition,column].dropna().to_numpy() for condition in conditions]
        ax.boxplot(values,tick_labels=['Completa','Parziale','Conservativa'],showfliers=True)
        ax.set_ylim(-.03,1.03);ax.set_ylabel('Supporto NLI');ax.set_title(title,fontsize=10)
        if column=='document_support':ax.axhline(summary['threshold_support'],color='#b6483e',ls='--',label='Soglia storica');ax.legend(fontsize=8)
    fig.tight_layout()
    for extension in ('png','pdf'):fig.savefig(output/f'support_distributions.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    n_full=int(summary['condition_counts'].get('hedging_complete',0));n_partial=int(summary['condition_counts'].get('hedging_partial',0));n_positive=int(summary['condition_counts'].get('hedging_preserved',0))
    all_row=primary[primary.scope=='all'].iloc[0]
    strict=metrics[(metrics.panel=='assistant_review_valid') & (metrics.metric=='document_min') & (metrics.scope=='all')]
    strict_intro=''
    if len(strict):
        clean=strict.iloc[0]
        strict_intro=f"Nel sottoinsieme con alterazioni e controlli entrambi chiaramente validi alla lettura dell'assistente rimangono **{int(clean.n_sources)} originali, {int(clean.n_altered)} alterazioni e {int(clean.n_preserved)} controlli**: AUC **{_decimal(clean.auc)}**, IC 95% **{_ci(clean,'auc')}**, sensibilità **{_decimal(100*clean.sensitivity,1)}%**, falsi allarmi **{_decimal(100*clean.false_positive_rate,1)}%**. Anche questo sottoinsieme resta privo di etichette umane indipendenti."
    discussion_path=output/'discussione.md'
    discussion=discussion_path.read_text(encoding='utf-8') if discussion_path.exists() else ''
    text=f"""# Cautele epistemiche: test mirato con soglia storica fissa

**Data dell'analisi: {report_date}.**

## Obiettivo e risposta del test

Il pannello comprende **{n_full+n_partial} modifiche con intento di alterazione** ({n_full} rimozioni complete e {n_partial} rafforzamenti parziali), **{n_positive} parafrasi con intento conservativo** e **{summary['n_sources']} originali distinti**. Si valuta soltanto la risposta dell'NLI alle cautele: non si ripetono gli altri esperimenti semantici e non si valuta direttamente la humanization.

Le etichette indicano l'intento delle regole, controllato mediante lettura dell'assistente prima dei punteggi. Non sono disponibili etichette umane indipendenti per questo nuovo pannello. I casi linguisticamente ambigui sono segnalati e analizzati separatamente.

Sul confronto complessivo, l'AUC documentale è **{_decimal(all_row.auc)}**, la sensibilità alla soglia storica è **{_decimal(100*all_row.sensitivity,1)}%** e i falsi allarmi sui controlli conservativi sono **{_decimal(100*all_row.false_positive_rate,1)}%**. La soglia è mantenuta, non ricalibrata. Il confronto con i due tipi di trasformazione e con i controlli positivi è necessario per interpretare questi valori.

{strict_intro}

## Che cosa viene modificato

Ogni originale mantiene tre confronti collegati quando applicabili: una rimozione della cautela, un rafforzamento parziale che mantiene una qualificazione e una parafrasi che conserva l'incertezza. L'intento delle trasformazioni non è di per sé una gold label: avverbi e modali possono avere funzioni differenti, essere ridondanti o modificare proposizioni diverse. Il tipo di errore deve quindi essere letto nel contesto.

Un esempio effettivo del pannello chiarisce la distinzione. L'originale dice: “Memes themselves **may be** byproducts of other evolved features rather than adaptations”. La rimozione sostituisce “may be” con “are”; il rafforzamento parziale usa “are probably”; il controllo fedele diventa “It is possible that memes themselves are byproducts of other evolved features rather than adaptations”. Il primo passaggio afferma ciò che prima era soltanto possibile, il secondo lo rende probabile, il terzo conserva la possibilità. Non si sta decidendo se l'affermazione sui memi sia vera: si confronta la forza delle affermazioni.

I controlli conservativi provengono dagli stessi originali delle alterazioni. Non si confrontano semplicemente identità facili con testi modificati, né si confonde la classe con la provenienza da un altro corpus. Il numero di coppie è distinto dal numero di originali indipendenti; le varianti dello stesso testo rimangono insieme nell'analisi.

## Provenienza e selezione dei testi

Si usano testi macchina nativi dei tre file MGTBench del progetto, lunghi da 80 a 1.000 parole prima della modifica. La cautela è già presente nell'originale: non viene inserita artificialmente. Si seleziona un solo testo macchina per gruppo di fonte, collegando righe con prompt o testo umano condiviso e rimuovendo duplicati. Gruppo e testo macchina non sono sinonimi: lo stesso gruppo può contenere generazioni differenti.

{source_table}

Il piano iniziale preferiva 20 originali per dominio. Lo screening non ha trovato abbastanza Reuters ammissibili; prima di qualsiasi inferenza NLI è stata accettata una distribuzione non uniforme, mantenendo 60 originali e senza allentare le regole semantiche. Sono stati riammessi testi usati soltanto per la stilometria: non avevano contribuito alla calibrazione NLI. Restano esclusi i casi delle precedenti perturbazioni semantiche indicati nel protocollo. Le modifiche del piano e gli screening falliti sono documentati, non nascosti.

Il criterio storico era calibrato su OpAI, mentre qui si usano testi MGTBench. Si verifica quindi anche un trasferimento fra distribuzioni: il risultato non isola soltanto l'effetto di ampliare il numero di esempi della vecchia categoria. I controlli conservativi e le alterazioni condividono però le stesse fonti nel nuovo test.

L'audit delle sovrapposizioni è in `source_overlap_audit.json`: cinque gruppi sono condivisi con lo study Kaggle da 126 fonti, ma con generatori differenti e nessuna coincidenza dell'originale macchina confrontato. Non si presenta dunque il pannello come un insieme di gruppi mai incontrati in tutto il progetto. Gli originali della calibrazione NLI OpAI non coincidono testualmente con quelli del nuovo test.

## Configurazione e soglia

spaCy con `en_core_web_sm` identifica le frasi e le costruzioni ammissibili. `all-MiniLM-L6-v2` recupera, per ogni unità, le tre premesse più simili nell'altro testo. Le unità lunghe sono suddivise in blocchi entro 160 token per il recupero; ogni coppia NLI rispetta il budget di 384 token. **DeBERTa-v3 base**, checkpoint `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, calcola entailment nelle due direzioni in float32. Si riutilizzano modelli locali e l'export ONNX già disponibile; non viene addestrato alcun modello.

Forward significa originale come premessa e riscrittura come ipotesi; reverse scambia i ruoli. Per ogni unità si prende il massimo entailment tra le tre premesse recuperate, poi il minimo sulle unità. Questa è la configurazione storica della soglia, distinta dalla configurazione contestuale `window2` della suite attuale.

Il punteggio documentale è **il minimo fra supporto forward e reverse**, con le rispettive unità e regole storiche. La decisione è: **segnalare se il supporto è inferiore a {summary['threshold_support']:.15f}**, visualizzato come 0,04075. Non si sceglie una soglia guardando il nuovo test e non la si trasferisce a una configurazione NLI diversa.

L'AUC usa il punteggio continuo orientato come rischio di alterazione, cioè meno supporto significa più rischio. È indipendente dalla soglia; sensibilità e falsi allarmi riguardano invece il punto operativo fissato. Non si inverte il segno se un'AUC è inferiore a 0,5.

In termini pratici, **AUC 0,5** indica un ordinamento equivalente al caso e **AUC 1** una separazione perfetta; non è una percentuale di testi classificati correttamente. La **sensibilità** è la quota delle modifiche intenzionalmente alteranti che produce un allarme. Il **tasso di falsi positivi (FPR)** è la quota delle parafrasi conservative segnalata come problematica. La specificità è uno meno questo tasso. Sensibilità e falsi allarmi devono essere letti insieme.

Rafforzare un'affermazione cauta può produrre un mancato supporto senza contraddizione: da “P potrebbe essere vero” non segue “P è vero”, mentre le due affermazioni possono essere compatibili. Per questo si conservano anche i punteggi delle due direzioni e della frase modificata.

## Risultati sul pannello completo

{main_table}

{operating_table}

Ogni originale ha uguale peso entro ciascuna classe. La sensibilità per fonte può quindi differire dal semplice rapporto degli allarmi. I conteggi grezzi sono riportati esplicitamente. Nel confronto rimozione completa o parziale vengono usati soltanto i controlli conservativi collegati agli stessi originali.

Gli intervalli al 95% sono ottenuti con **{summary['n_bootstrap']} bootstrap per originale**: tutte le varianti di una fonte viaggiano insieme. Sono intervalli puntuali, condizionati a testi, regole e modelli fissati; non rappresentano tutta l'incertezza di selezione del campione o l'affidabilità generale sui veri humanizer. Non viene ricalibrata alcuna soglia nei ricampionamenti.

Se una quota osservata è zero o uno, il bootstrap empirico può produrre un intervallo degenere: questo non dimostra che il corrispondente tasso nella popolazione sia esattamente zero o uno.

Anche un'AUC empirica di 1 può avere un bootstrap interamente pari a 1 se i casi osservati sono perfettamente separati. Questo descrive il campione e non dimostra una separazione perfetta su nuovi testi, soprattutto nei sottoinsiemi piccoli.

Come controllo secondario si riportano intervalli di Wilson soltanto quando ogni originale contribuisce un unico esempio nella classe considerata, assumendo indipendenza fra originali. Questo vale per ciascun sottotipo e per i controlli conservativi. Non si applica Wilson alla sensibilità complessiva con due alterazioni dello stesso originale, né all'AUC. Il bootstrap per fonte rimane il metodo principale.

{wilson_table}

## Frase modificata e aggregazione documentale

{local_table}

La soglia storica documentale **non è applicata al punteggio locale**. La sua AUC è una diagnostica della capacità di riconoscere il cambiamento nella frase interessata. Un minimo documentale può invece essere determinato da una frase difficile diversa da quella modificata; non va interpretato come percentuale di significato conservato.

![Distribuzioni del supporto documentale e locale. La linea della soglia è presente soltanto sulla misura documentale storica.](support_distributions.png)

I riepiloghi forward/reverse, supporto e contraddizione sono in `directional_summary.csv`. Un aumento della contraddizione non è un requisito per riconoscere un rafforzamento epistemico.

## Revisione testuale e provenienza delle etichette

{review_text}

{review_table}

I sottoinsiemi richiedono controlli conservativi e alterazioni entrambi revisionati e collegati agli stessi originali. Le coppie non revisionate non sono considerate automaticamente valide. I casi qualificati rimangono distinti dai casi chiaramente validi; il pannello completo non viene ripulito sulla base degli score NLI.

## Interpretazione e limiti

{discussion}

Il test misura il trasferimento di un criterio già fissato a nuovi esempi di cautela. Una sensibilità alta accompagnata da molti falsi allarmi non dimostra una buona capacità di distinguere cambiamenti reali da parafrasi fedeli. Un'AUC buona con bassa sensibilità alla soglia fissa indica invece che il ranking e la decisione operativa rispondono a domande diverse.

Le regole coprono un sottoinsieme della cautela epistemica: non rappresentano tutte le forme indirette, le citazioni, le condizioni ipotetiche o le differenze di probabilità. Eliminare una parola non equivale necessariamente a eliminare l'incertezza. La revisione testuale e l'eventuale annotazione umana indipendente sono perciò distinte dalle probabilità emesse dal modello.

## Materiali

Sono disponibili [metriche e intervalli](metrics.csv), [punteggi per coppia](pair_scores.csv), [riepiloghi direzionali](directional_summary.csv), [riepilogo](summary.json) e [manifest dell'analisi](analysis_manifest.json). Le AUC dei ricampionamenti sono in `bootstrap_metrics.npz`; eventuali revisioni usate sono conservate separatamente.

`scripts/analyze_hedging_controls.py` ricalcola tabelle, intervalli, figura e questo report senza eseguire nuove inferenze NLI. Il calcolo della soglia storica non viene ripetuto.
"""
    path=output/'report.md';path.write_text(text,encoding='utf-8');return path

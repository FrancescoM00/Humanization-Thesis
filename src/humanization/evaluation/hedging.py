"""Valutazione a soglia congelata di un test dedicato alle espressioni di cautela.

"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from humanization.evaluation.semantic import _auc_sorted


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
    # Una soglia numerica congelata ha senso solo per lo score sul quale era stata definita.
    for column, expected in (("nli_retrieval_mode", "top_k_cosine"),
                             ("nli_retrieval_top_k", 3), ("nli_max_pair_tokens", 384)):
        if column in scores and not scores[column].eq(expected).all():
            raise ValueError(f"Historical threshold cannot be transferred to different {column}")
    # Ammette i nomi precedenti degli scorer e un alias equivalente esplicito.
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
        # Una provenienza non specificata non deve diventare implicitamente umana.
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
    """Intervallo binomiale secondario, usato solo con una osservazione per fonte."""
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
    """AUC ponderata per fonte e punto operativo con bootstrap appaiato per gruppi."""
    part = frame[np.isfinite(pd.to_numeric(frame[score_column], errors="coerce"))].copy()
    # Nella diagnostica locale, le unità mancanti non devono creare corpora non abbinati.
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


def plot_support_distributions(output: Path | str) -> Path:
    """Salva le distribuzioni documentali e locali come immagine PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    frame = pd.read_csv(output / "pair_scores.csv")
    summary = json.loads((output / "summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    panels = ((axes[0], "document_support", "Minimo bidirezionale sul documento"),
              (axes[1], "local_support", "Minimo bidirezionale sulla frase modificata"))
    for ax, column, title in panels:
        if column not in frame:
            ax.axis("off")
            continue
        values = [frame.loc[frame.condition == condition, column].dropna().to_numpy()
                  for condition in CONDITIONS]
        ax.boxplot(values, tick_labels=["Completa", "Parziale", "Conservativa"],
                   showfliers=True)
        ax.set_ylim(-.03, 1.03)
        ax.set_ylabel("Supporto NLI")
        ax.set_title(title, fontsize=10)
        if column == "document_support":
            ax.axhline(summary["threshold_support"], color="#b6483e", ls="--",
                       label="Soglia storica")
            ax.legend(fontsize=8)
    fig.tight_layout()
    destination = output / "support_distributions.png"
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination

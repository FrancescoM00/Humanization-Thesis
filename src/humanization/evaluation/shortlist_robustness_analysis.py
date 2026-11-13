"""Analisi con 100 feature candidate. 
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
    """I conteggi di selezione sono descrittivi; training sovrapposti non sono prove indipendenti."""
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
    """Calcola differenze di AUC esattamente appaiate dalle predizioni salvate sui test esclusi."""
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

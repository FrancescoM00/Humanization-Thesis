"""Paired, source-group uncertainty for the compact-feature experiment.

The input predictions are already out of fold. Repeated predictions of a text
stay together in the bootstrap; neither folds nor CV seeds are independent
observations. Random-feature references average *AUCs*, not probabilities.

For this experiment each source group contains one human and one machine text.
That permits an exact comparison kernel: resampling a group gives the same
weight to its two texts. The resulting intervals are conditional on the saved
fits and do not include uncertainty from refitting the training procedure.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from .features import FEATURE_GROUPS, FEATURE_SCHEMA


CORE_MODELS = (
    "C1_full105", "C2_fixed4", "C3_training4", "C1_full100",
    "C1_style93", "word_count_only",
)
RANDOM_CONFIGURATIONS = ("C4_random", "C4_random_matched")
CONFIGURATION_ORDER = (*CORE_MODELS, *RANDOM_CONFIGURATIONS)
HISTORICAL_FEATURES = ("pos_num", "pos_cconj", "mattr_50", "pos_propn")
PROTOCOL_STRATA = {
    "cv": ("domain", "domain"),
    "leave_domain_out": ("domain", "domain"),
    "leave_generator_out": ("source_generator", "generator"),
}


def pairwise_auc_kernel(positive_scores, negative_scores) -> np.ndarray:
    """Average comparison credit over views; ties receive one half.

    A view is one fitted model/repetition. Inputs have shape ``(groups,)`` or
    ``(views, groups)``; positive and negative positions use the same group
    order. Averaging kernels therefore averages the individual model AUCs.
    """
    positive = np.asarray(positive_scores, dtype=float)
    negative = np.asarray(negative_scores, dtype=float)
    if positive.ndim == 1:
        positive = positive[None, :]
    if negative.ndim == 1:
        negative = negative[None, :]
    if positive.ndim != 2 or positive.shape != negative.shape:
        raise ValueError("Positive and negative scores need equal (views, groups) shapes")
    if not positive.size or not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("Comparison scores must be finite and nonempty")
    n_views, n_groups = positive.shape
    kernel = np.zeros((n_groups, n_groups), dtype=float)
    # Limit transient comparison arrays, including for the random references.
    for start in range(0, n_views, 16):
        a = positive[start:start + 16, :, None]
        b = negative[start:start + 16, None, :]
        kernel += np.greater(a, b).sum(axis=0)
        kernel += 0.5 * np.equal(a, b).sum(axis=0)
    return kernel / n_views


def weighted_auc_from_kernel(kernel, weights):
    """Return exact grouped AUC for one or several group-weight vectors.

    Weights are shared by the positive and negative document of every group.
    For bootstrap counts ``w``, the AUC is ``w.T @ K @ w / sum(w)**2``.
    """
    kernel = np.asarray(kernel, dtype=float)
    values = np.asarray(weights, dtype=float)
    scalar = values.ndim == 1
    if scalar:
        values = values[None, :]
    if (kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]
            or values.ndim != 2 or values.shape[1] != kernel.shape[0]):
        raise ValueError("A square kernel and compatible group weights are required")
    if (not np.isfinite(kernel).all() or not np.isfinite(values).all()
            or (values < 0).any() or (values.sum(axis=1) <= 0).any()):
        raise ValueError("Kernel and weights must be finite; weights need positive total mass")
    answer = ((values @ kernel) * values).sum(axis=1) / values.sum(axis=1)**2
    return float(answer[0]) if scalar else answer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _auc_by_model(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Vectorized AUC for every repetition and individual model."""
    positive = labels == 1
    n_positive, n_negative = int(positive.sum()), int((labels == 0).sum())
    if not n_positive or not n_negative:
        raise ValueError("AUC needs both classes")
    # rankdata averages ties, exactly as the comparison-credit definition.
    values = predictions.transpose(0, 2, 1)
    ranks = rankdata(values, method="average", axis=-1)
    return (ranks[..., positive].sum(axis=-1)
            - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def _configuration_indices(registry: pd.DataFrame) -> dict[str, list[int]]:
    if registry.model.duplicated().any():
        raise ValueError("Model registry contains duplicate names")
    names = dict(zip(registry.model.astype(str), range(len(registry))))
    missing = set(CORE_MODELS) - set(names)
    if missing:
        raise ValueError(f"Missing core models: {sorted(missing)}")
    result = {name: [names[name]] for name in CORE_MODELS}
    for configuration in RANDOM_CONFIGURATIONS:
        indices = np.flatnonzero(registry.configuration.eq(configuration)).tolist()
        if not indices:
            raise ValueError(f"Missing random reference {configuration}")
        result[configuration] = indices
    return result


def _validate_documents(documents: pd.DataFrame) -> None:
    required = {"id", "group_id", "pair_id", "label", "domain", "source_generator"}
    if required - set(documents.columns):
        raise ValueError(f"Missing document metadata: {sorted(required - set(documents.columns))}")
    if documents.id.duplicated().any() or documents[list(required)].isna().any().any():
        raise ValueError("Document IDs must be unique and group metadata complete")
    if set(documents.label.unique()) != {0, 1}:
        raise ValueError("Exactly labels zero and one are required")
    counts = documents.groupby(["group_id", "label"]).size().unstack(fill_value=0)
    if counts.shape[1] != 2 or not counts.eq(1).all().all():
        raise ValueError("Kernel bootstrap requires one human and one machine in each group")
    for column in ("domain", "source_generator"):
        if documents.groupby("group_id")[column].nunique().gt(1).any():
            raise ValueError(f"A source group crosses {column} strata")


def _group_positions(documents: pd.DataFrame, indices: np.ndarray):
    part = documents.iloc[indices]
    groups = pd.unique(part.group_id)
    human = part[part.label == 0].set_index("group_id")
    machine = part[part.label == 1].set_index("group_id")
    negative = human.loc[groups, "_row"].to_numpy(dtype=int)
    positive = machine.loc[groups, "_row"].to_numpy(dtype=int)
    return tuple(map(str, groups)), positive, negative


def _quantiles(values: np.ndarray) -> tuple[float, float]:
    return tuple(map(float, np.quantile(values, [0.025, 0.975])))


def selection_frequency_table(selections: pd.DataFrame,
                              feature_names=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive selection rates; overlapping fits are not independent trials."""
    if feature_names is None:
        feature_names = list(FEATURE_SCHEMA)
    required = {"protocol", "seed", "fold", "feature", "selected"}
    if required - set(selections.columns):
        raise ValueError(f"Missing selection fields: {sorted(required - set(selections.columns))}")
    selections = selections.copy()
    selections["_selected"] = selections.selected.astype(str).str.lower().isin(("true", "1"))
    split_columns = [name for name in ("seed", "fold", "held_out") if name in selections]
    feature_rows, family_rows = [], []
    for protocol, part in selections.groupby("protocol", sort=True):
        n_fits = len(part[split_columns].drop_duplicates())
        chosen = part.loc[part._selected]
        if chosen.duplicated([*split_columns, "feature"]).any():
            raise ValueError("A feature occurs twice in a selection fit")
        frequencies = chosen.feature.value_counts()
        for feature in feature_names:
            count = int(frequencies.get(feature, 0))
            feature_rows.append({
                "protocol": protocol, "feature": feature,
                "family": FEATURE_GROUPS.get(feature, "unknown"),
                "selection_count": count, "selection_rate": count / n_fits,
                "n_fits": n_fits, "historical_shortlist": feature in HISTORICAL_FEATURES,
            })
        selected_slots = len(chosen)
        chosen = chosen.assign(family=chosen.feature.map(FEATURE_GROUPS))
        for family in sorted(set(FEATURE_GROUPS.values())):
            family_part = chosen[chosen.family == family]
            slots = len(family_part)
            family_rows.append({
                "protocol": protocol, "family": family, "n_fits": n_fits,
                "n_selected_slots": slots,
                "fraction_of_selected_slots": slots / selected_slots if selected_slots else np.nan,
                "n_fits_with_family": len(family_part[split_columns].drop_duplicates()),
            })
    return pd.DataFrame(feature_rows), pd.DataFrame(family_rows)


def analyze_results(results: Path | str, protocol_path: Path | str | None = None) -> dict[str, Any]:
    """Analyze saved fits without model training or new feature selection."""
    started = time.monotonic()
    results = Path(results)
    protocol_path = Path(protocol_path) if protocol_path else results / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    documents = pd.read_csv(results / "documents.csv")
    registry = pd.read_csv(results / "model_registry.csv")
    _validate_documents(documents)
    documents["_row"] = np.arange(len(documents))
    configurations = _configuration_indices(registry)
    expected_random = int(protocol.get("n_random_subsets", 100))
    if any(len(configurations[name]) != expected_random for name in RANDOM_CONFIGURATIONS):
        raise ValueError("Random-reference model count differs from the frozen protocol")
    n_bootstrap = int(protocol.get("n_bootstrap", 2000))
    seed = int(protocol.get("bootstrap_seed", 20261112))
    if n_bootstrap < 2:
        raise ValueError("At least two bootstrap draws are required")
    rng = np.random.default_rng(seed)
    weight_cache: dict[tuple, np.ndarray] = {}
    metric_rows, random_rows, difference_rows, reference_rows = [], [], [], []
    bootstrap_archive = {}
    comparisons = (
        ("C3_training4", "C1_full105"), ("C3_training4", "C2_fixed4"),
        ("C3_training4", "C4_random"), ("C3_training4", "C4_random_matched"),
        ("C2_fixed4", "C1_full105"), ("C2_fixed4", "C4_random"),
        ("C2_fixed4", "C4_random_matched"),
        ("C1_full100", "C1_full105"), ("C1_style93", "C1_full105"),
    )
    labels = documents.label.to_numpy()

    def record_scope(protocol_name, scope, stratum, individual_auc, boots, n_groups, n_texts):
        points, repeat_points = {}, {}
        for name, model_indices in configurations.items():
            repeats = individual_auc[:, model_indices].mean(axis=1)
            repeat_points[name] = repeats
            point = float(repeats.mean())
            points[name] = point
            low, high = _quantiles(boots[name]) if boots is not None else (np.nan, np.nan)
            metric_rows.append({
                "protocol": protocol_name, "scope": scope, "stratum": stratum,
                "configuration": name, "auc": point, "ci_low": low, "ci_high": high,
                "repeat_auc_min": float(repeats.min()), "repeat_auc_max": float(repeats.max()),
                "n_repeats": len(repeats), "n_models": len(model_indices),
                "n_groups": n_groups, "n_texts": n_texts,
                "estimand": "mean_individual_auc" if name in RANDOM_CONFIGURATIONS else "mean_repeat_auc",
            })
        for name in RANDOM_CONFIGURATIONS:
            indices = configurations[name]
            values = individual_auc[:, indices].mean(axis=0)
            for model_index, point in zip(indices, values):
                repeats = individual_auc[:, model_index]
                random_rows.append({
                    "protocol": protocol_name, "scope": scope, "stratum": stratum,
                    "model": registry.iloc[model_index].model, "reference": name,
                    "auc": float(point), "repeat_auc_min": float(repeats.min()),
                    "repeat_auc_max": float(repeats.max()), "n_repeats": len(repeats),
                })
            low, high = _quantiles(values)
            reference_rows.append({
                "protocol": protocol_name, "scope": scope, "stratum": stratum,
                "reference": name, "n_subsets": len(values),
                "mean_auc": float(values.mean()), "median_auc": float(np.median(values)),
                "q025_auc": low, "q975_auc": high,
                "sd_across_subsets": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "fraction_ge_C2_fixed4": float(np.mean(values >= points["C2_fixed4"])),
                "fraction_ge_C3_training4": float(np.mean(values >= points["C3_training4"])),
                "quantiles_are_confidence_intervals": False,
            })
        for left, right in comparisons:
            low, high = (_quantiles(boots[left] - boots[right])
                         if boots is not None else (np.nan, np.nan))
            difference_rows.append({
                "protocol": protocol_name, "scope": scope, "stratum": stratum,
                "left_configuration": left, "right_configuration": right,
                "auc_difference": points[left] - points[right], "ci_low": low, "ci_high": high,
                "historical_fixed_shortlist_involved": "C2_fixed4" in (left, right),
                "interval": "pointwise_source_bootstrap_fixed_predictions" if boots is not None else "not_computed",
            })
        if boots is not None:
            key = re.sub(r"[^a-zA-Z0-9_]+", "_", f"{protocol_name}__{scope}__{stratum}")
            bootstrap_archive[key] = np.column_stack([boots[name] for name in CONFIGURATION_ORDER])

    with threadpool_limits(limits=int(protocol.get("threads", 2))):
        with np.load(results / "predictions.npz", allow_pickle=False) as saved:
            for protocol_name, (column, scope) in PROTOCOL_STRATA.items():
                predictions = saved[protocol_name]
                if (predictions.ndim != 3 or predictions.shape[1:] != (len(documents), len(registry))
                        or not np.isfinite(predictions).all()
                        or (predictions < 0).any() or (predictions > 1).any()):
                    raise ValueError(f"Invalid prediction cube for {protocol_name}")
                expected_repeats = len(protocol.get("cv_seeds", [42, 137, 271, 811, 2026])) if protocol_name == "cv" else 1
                if predictions.shape[0] != expected_repeats:
                    raise ValueError(f"Unexpected repetition count for {protocol_name}")
                per_stratum_auc, per_stratum_boots = [], []
                for stratum in sorted(documents[column].unique()):
                    indices = np.flatnonzero(documents[column].eq(stratum).to_numpy())
                    groups, positive, negative = _group_positions(documents, indices)
                    n_groups = len(groups)
                    weight_key = (column, str(stratum), groups)
                    if weight_key not in weight_cache:
                        weight_cache[weight_key] = rng.multinomial(
                            n_groups, np.full(n_groups, 1 / n_groups), size=n_bootstrap).astype(float)
                    weights = weight_cache[weight_key]
                    individual_auc = _auc_by_model(labels[indices], predictions[:, indices, :])
                    boots = {}
                    for name, model_indices in configurations.items():
                        positive_scores = predictions[:, positive, :][:, :, model_indices].transpose(0, 2, 1)
                        negative_scores = predictions[:, negative, :][:, :, model_indices].transpose(0, 2, 1)
                        kernel = pairwise_auc_kernel(
                            positive_scores.reshape(-1, n_groups), negative_scores.reshape(-1, n_groups))
                        expected = float(individual_auc[:, model_indices].mean())
                        if not np.isclose(kernel.mean(), expected, atol=1e-12, rtol=0):
                            raise AssertionError("Kernel and rank-based AUC disagree")
                        boots[name] = weighted_auc_from_kernel(kernel, weights)
                    record_scope(protocol_name, scope, str(stratum), individual_auc, boots,
                                 n_groups, len(indices))
                    per_stratum_auc.append(individual_auc)
                    per_stratum_boots.append(boots)
                    print(f"{protocol_name}: {stratum}; {n_groups} gruppi; bootstrap completato", flush=True)
                macro_auc = np.mean(per_stratum_auc, axis=0)
                macro_boots = {name: np.mean([part[name] for part in per_stratum_boots], axis=0)
                               for name in CONFIGURATION_ORDER}
                record_scope(protocol_name, "macro", "ALL", macro_auc, macro_boots,
                             documents.group_id.nunique(), len(documents))
                overall_auc = _auc_by_model(labels, predictions)
                record_scope(protocol_name, "overall", "ALL", overall_auc, None,
                             documents.group_id.nunique(), len(documents))

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(results / "metrics.csv", index=False)
    pd.DataFrame(random_rows).to_csv(results / "random_subset_metrics.csv", index=False)
    pd.DataFrame(reference_rows).to_csv(results / "random_reference_summary.csv", index=False)
    pd.DataFrame(difference_rows).to_csv(results / "paired_differences.csv", index=False)
    np.savez_compressed(results / "bootstrap_auc.npz", **bootstrap_archive)
    selections = pd.read_csv(results / "selections.csv")
    feature_frequency, family_frequency = selection_frequency_table(selections)
    feature_frequency.to_csv(results / "selection_frequency.csv", index=False)
    family_frequency.to_csv(results / "selection_family_frequency.csv", index=False)
    primary = metrics[(metrics.protocol == "cv") & (metrics.scope == "macro")]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_estimand": "CV mean of the three domain AUCs, averaged across fixed CV seeds",
        "primary_metrics": json.loads(primary.to_json(orient="records")),
        "configuration_order_in_bootstrap_archive": list(CONFIGURATION_ORDER),
        "n_bootstrap": n_bootstrap, "bootstrap_seed": seed,
        "bootstrap_strata": {name: column for name, (column, _) in PROTOCOL_STRATA.items()},
        "same_group_weights_for_all_configurations_and_repeats": True,
        "random_reference_is_probability_ensemble": False,
        "fixed_shortlist_independently_validated": False,
        "interval_scope": "Conditional on saved fits; no bootstrap refitting, fixed candidate pools and random subsets",
        "multiple_comparison_adjustment": "None; pointwise exploratory confidence intervals, no equivalence claim",
        "selection_frequency_is_independent_binomial_sample": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "created_at": summary["created_at"], "protocol_sha256": _sha256(protocol_path),
        "inputs": {name: _sha256(results / name) for name in
                   ("documents.csv", "model_registry.csv", "predictions.npz", "selections.csv")},
        "code_sha256": _sha256(Path(__file__)),
        "versions": {package: importlib.metadata.version(package) for package in ("numpy", "pandas", "scipy", "threadpoolctl")},
        "n_metrics": len(metrics), "n_paired_differences": len(difference_rows),
        "bootstrap_archive_columns": list(CONFIGURATION_ORDER),
        "elapsed_seconds": summary["elapsed_seconds"],
    }
    (results / "statistics_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary

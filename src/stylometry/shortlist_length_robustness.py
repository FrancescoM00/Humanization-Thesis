"""Refit C3 after excluding five diagnostics, reusing the saved source splits.

The previous C3 probabilities are copied, not retrained. Only the candidate
pool changes; supervised selection and all preprocessing remain training-only.
Excluding five diagnostics is not a claim of eliminating every length effect.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from humanization_paths import resolve_legacy_path

from .features import FEATURE_SCHEMA, LENGTH_SENSITIVE_FEATURES
from .shortlist_validation import prepare_documents, select_training_features, sha256


SPLIT_KEYS = ["protocol", "repeat", "seed", "fold", "held_out"]
PREDICTION_PROTOCOLS = ("cv", "leave_domain_out", "leave_generator_out")


def load_saved_splits(documents: pd.DataFrame, membership: pd.DataFrame):
    """Recover exact parent train/test indices without generating new folds.

    Every saved split must assign every source group exactly once. Whole-group
    membership and held-out domain/generator protection are checked on the
    original document order returned by ``prepare_documents``.
    """
    required = {*SPLIT_KEYS, "group_id", "role"}
    if required - set(membership):
        raise ValueError(f"Missing saved split fields: {sorted(required - set(membership))}")
    if membership[list(required)].isna().any().any():
        raise ValueError("Missing split membership metadata")
    if not set(membership.role).issubset({"train", "test"}):
        raise ValueError("Saved split roles must be train or test")
    expected_groups = set(documents.group_id.astype(str))
    for key, part in membership.groupby(SPLIT_KEYS, sort=False, dropna=False):
        info = dict(zip(SPLIT_KEYS, key))
        for name in ("repeat", "seed", "fold"):
            info[name] = int(info[name])
        if info["protocol"] not in PREDICTION_PROTOCOLS:
            raise ValueError("Unknown saved prediction protocol")
        if part.group_id.astype(str).duplicated().any():
            raise ValueError("A group has more than one role in a saved split")
        if set(part.group_id.astype(str)) != expected_groups:
            raise ValueError("Saved split does not cover the frozen source groups")
        train_groups = set(part.loc[part.role == "train", "group_id"].astype(str))
        test_groups = set(part.loc[part.role == "test", "group_id"].astype(str))
        if not train_groups or not test_groups or train_groups & test_groups:
            raise ValueError("Empty or overlapping saved train/test groups")
        train = np.flatnonzero(documents.group_id.astype(str).isin(train_groups).to_numpy())
        test = np.flatnonzero(documents.group_id.astype(str).isin(test_groups).to_numpy())
        if documents.iloc[train].label.nunique() != 2 or documents.iloc[test].label.nunique() != 2:
            raise ValueError("Each saved training and test set needs both classes")
        if info["protocol"] == "leave_domain_out":
            if documents.iloc[train].domain.eq(info["held_out"]).any():
                raise ValueError("Held-out domain occurs in training")
            if not documents.iloc[test].domain.eq(info["held_out"]).all():
                raise ValueError("Domain test does not match the saved holdout")
        if info["protocol"] == "leave_generator_out":
            if documents.iloc[train].source_generator.eq(info["held_out"]).any():
                raise ValueError("Held-out generator occurs in training")
            if not documents.iloc[test].source_generator.eq(info["held_out"]).all():
                raise ValueError("Generator test does not match the saved holdout")
        yield info, train, test


def _resolve(root: Path, path) -> Path:
    value = Path(path)
    return resolve_legacy_path(root, value).resolve()


def _verified_hash(path: Path, expected: str | None, description: str) -> str:
    if not expected:
        raise ValueError(f"Frozen protocol needs {description} SHA-256")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{description} differs from its frozen SHA-256")
    return actual


def run_robustness(protocol_path: Path, output: Path, root: Path) -> dict:
    """Fit only C3_100, retaining parent C3_105 and exact parent memberships."""
    started = time.monotonic()
    protocol_path, output, root = protocol_path.resolve(), output.resolve(), root.resolve()
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    parent = _resolve(root, protocol.get("parent_results", "results/02_stylometry/shortlist"))
    source = _resolve(root, protocol.get("input_csv", "results/02_stylometry/length_matched/features.csv"))
    if output in {parent, source.parent}:
        raise ValueError("Use a new results directory; parent results are immutable")
    checks = {
        source: _verified_hash(source, protocol.get("input_sha256"), "input features"),
        parent / "protocol.json": _verified_hash(parent / "protocol.json", protocol.get("parent_protocol_sha256"), "parent protocol"),
        parent / "predictions.npz": _verified_hash(parent / "predictions.npz", protocol.get("parent_predictions_sha256"), "parent predictions"),
        parent / "split_membership.csv": _verified_hash(parent / "split_membership.csv", protocol.get("parent_splits_sha256"), "parent splits"),
    }
    old_protocol = json.loads((parent / "protocol.json").read_text())
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    if parent_manifest.get("status") != "completed":
        raise ValueError("Parent validation has not completed")
    if parent_manifest.get("input_sha256") != checks[source]:
        raise ValueError("The parent used a different input feature table")
    if parent_manifest.get("protocol_sha256") != checks[parent / "protocol.json"]:
        raise ValueError("Parent protocol differs from its run manifest")
    settings = protocol.get("selection", {})
    for field in ("n_features", "max_abs_spearman"):
        if settings.get(field) != old_protocol.get("selection", {}).get(field):
            raise ValueError(f"Selection setting {field} differs from the parent")
    if settings.get("n_features") != 4:
        raise ValueError("This robustness comparison selects four features")
    logistic = protocol.get("logistic", {})
    if logistic != old_protocol.get("logistic", {}):
        raise ValueError("Logistic settings differ from the parent")
    seeds = [int(s) for s in protocol.get("cv_seeds", [])]
    if seeds != old_protocol.get("cv_seeds"):
        raise ValueError("CV seeds differ from the parent")
    candidates = list(protocol.get("candidate_features", []))
    expected_candidates = [f for f in FEATURE_SCHEMA if f not in LENGTH_SENSITIVE_FEATURES]
    if candidates != expected_candidates or len(candidates) != 100:
        raise ValueError("Candidate pool must be the 100 schema features excluding the five diagnostics")
    if set(protocol.get("excluded_features", [])) != set(LENGTH_SENSITIVE_FEATURES):
        raise ValueError("The excluded feature inventory differs from the five diagnostics")
    frame = pd.read_csv(source).reset_index(drop=True)
    documents = prepare_documents(frame)
    parent_documents = pd.read_csv(parent / "documents.csv")
    pd.testing.assert_frame_equal(documents[parent_documents.columns], parent_documents,
                                  check_dtype=False, check_exact=True)
    membership = pd.read_csv(parent / "split_membership.csv")
    splits = list(load_saved_splits(documents, membership))
    expected_counts = {"cv": len(seeds) * int(old_protocol["n_splits"]),
                       "leave_domain_out": documents.domain.nunique(),
                       "leave_generator_out": documents.source_generator.nunique()}
    actual_counts = pd.Series([info["protocol"] for info, _, _ in splits]).value_counts().to_dict()
    if actual_counts != expected_counts:
        raise ValueError("Saved split counts differ from the parent design")
    split_audit = pd.read_csv(parent / "split_audit.csv")
    actual_keys = pd.DataFrame([info for info, _, _ in splits])[SPLIT_KEYS].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_keys, split_audit[SPLIT_KEYS].reset_index(drop=True),
                                  check_dtype=False, check_exact=True)
    registry = pd.read_csv(parent / "model_registry.csv")
    old_positions = np.flatnonzero(registry.model.eq("C3_training4"))
    if len(old_positions) != 1:
        raise ValueError("Parent model registry must contain exactly one C3_training4")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("manifest.json", "predictions.npz")):
        raise FileExistsError("A completed robustness run already exists; choose another output")
    destination_protocol = output / "protocol.json"
    if destination_protocol != protocol_path:
        if destination_protocol.exists() and destination_protocol.read_bytes() != protocol_bytes:
            raise ValueError("Output contains another protocol")
        destination_protocol.write_bytes(protocol_bytes)
    documents.to_csv(output / "documents.csv", index=False)
    pd.DataFrame([
        {"model": "C3_105", "configuration": "C3_105", "candidate_features": 105, "n_features": 4},
        {"model": "C3_100", "configuration": "C3_100", "candidate_features": 100, "n_features": 4},
    ]).to_csv(output / "model_registry.csv", index=False)
    predictions, old_predictions = {}, {}
    with np.load(parent / "predictions.npz", allow_pickle=False) as saved:
        for name in PREDICTION_PROTOCOLS:
            cube = saved[name]
            repeats = len(seeds) if name == "cv" else 1
            if cube.shape != (repeats, len(documents), len(registry)):
                raise ValueError(f"Unexpected parent prediction shape for {name}")
            old_predictions[name] = cube[:, :, int(old_positions[0])].copy()
            if not np.isfinite(old_predictions[name]).all():
                raise ValueError("Parent C3 probabilities are incomplete")
            predictions[name] = np.full((repeats, len(documents), 2), np.nan, dtype=np.float64)
            predictions[name][:, :, 0] = old_predictions[name]
    assignments = {name: np.zeros(values.shape[:2], dtype=np.int16) for name, values in predictions.items()}
    audit_rows, selected_rows, coefficient_rows, convergence = [], [], [], []
    with threadpool_limits(limits=1):
        for info, train_indices, test_indices in splits:
            if info["protocol"] == "cv":
                if info["repeat"] >= len(seeds) or seeds[info["repeat"]] != info["seed"]:
                    raise ValueError("Saved repeat index disagrees with parent CV seeds")
            elif info["repeat"] != 0:
                raise ValueError("A single holdout protocol must use repeat zero")
            train, test = frame.iloc[train_indices], frame.iloc[test_indices]
            selected, diagnostic = select_training_features(
                train, candidates, n_features=4, max_abs_spearman=float(settings["max_abs_spearman"]))
            selected_rows.extend({**info, **row} for row in diagnostic.to_dict("records"))
            model = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                LogisticRegression(C=float(logistic["C"]), class_weight=logistic["class_weight"],
                                   max_iter=int(logistic["max_iter"]), solver=logistic["solver"],
                                   tol=float(logistic["tol"]), random_state=info["seed"]),
            )
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(train[selected].replace([np.inf, -np.inf], np.nan), train.label)
            for warning in recorded:
                if issubclass(warning.category, ConvergenceWarning):
                    convergence.append({**info, "message": str(warning.message)})
            values = model.predict_proba(test[selected].replace([np.inf, -np.inf], np.nan))[:, 1]
            if not np.isfinite(values).all():
                raise AssertionError("New held-out probabilities are nonfinite")
            predictions[info["protocol"]][info["repeat"], test_indices, 1] = values
            assignments[info["protocol"]][info["repeat"], test_indices] += 1
            imputer, scaler, classifier = (step[1] for step in model.steps)
            for index, feature in enumerate(selected):
                coefficient_rows.append({**info, "model": "C3_100", "feature": feature,
                                         "coefficient_standardized": classifier.coef_[0, index],
                                         "intercept": classifier.intercept_[0],
                                         "imputation_median": imputer.statistics_[index],
                                         "scaler_mean": scaler.mean_[index], "scaler_scale": scaler.scale_[index]})
            train_docs, test_docs = documents.iloc[train_indices], documents.iloc[test_indices]
            audit_rows.append({**info, "n_train_documents": len(train_indices),
                               "n_test_documents": len(test_indices),
                               "n_train_groups": train_docs.group_id.nunique(),
                               "n_test_groups": test_docs.group_id.nunique(),
                               "group_overlap": len(set(train_docs.group_id) & set(test_docs.group_id)),
                               "saved_parent_membership_reused": True,
                               "heldout_domain_in_training": int(train_docs.domain.eq(info["held_out"]).sum()) if info["protocol"] == "leave_domain_out" else 0,
                               "heldout_generator_in_training": int(train_docs.source_generator.eq(info["held_out"]).sum()) if info["protocol"] == "leave_generator_out" else 0})
            print(f"{info['protocol']} seed={info['seed']} fold={info['fold']} held_out={info['held_out']}: "
                  f"selected={','.join(selected)}; elapsed={time.monotonic()-started:.2f}s", flush=True)
    for name in predictions:
        if not np.all(assignments[name] == 1) or not np.isfinite(predictions[name]).all():
            raise AssertionError("Each document needs exactly one new prediction per repetition")
        if not np.array_equal(predictions[name][:, :, 0], old_predictions[name]):
            raise AssertionError("Copied parent C3_105 probabilities changed")
    for path, before in checks.items():
        if sha256(path) != before:
            raise AssertionError("Frozen input or parent artifact changed during fitting")
    if protocol_path.read_bytes() != protocol_bytes:
        raise AssertionError("Robustness protocol changed during fitting")
    with (output / "predictions.npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, **predictions)
    (output / "predictions.npz.tmp").replace(output / "predictions.npz")
    pd.DataFrame(selected_rows).to_csv(output / "selections.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(output / "coefficients.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(output / "split_audit.csv", index=False)
    manifest = {
        "status": "completed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "input_sha256": checks[source], "parent_protocol_sha256": checks[parent / "protocol.json"],
        "parent_predictions_sha256": checks[parent / "predictions.npz"],
        "parent_splits_sha256": checks[parent / "split_membership.csv"],
        "parent_results": str(parent.relative_to(root)) if parent.is_relative_to(root) else str(parent),
        "n_documents": len(documents), "n_groups": int(documents.group_id.nunique()),
        "n_candidate_features": len(candidates), "n_selected_features": 4,
        "excluded_features": list(LENGTH_SENSITIVE_FEATURES), "n_fits": len(splits),
        "n_new_fits": len(splits), "n_refitted_parent_models": 0,
        "copied_C3_105_predictions_identical": True, "saved_parent_splits_reused": True,
        "prediction_shapes": {name: list(values.shape) for name, values in predictions.items()},
        "n_convergence_warnings": len(convergence), "convergence_warnings": convergence,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
        "code_sha256": sha256(Path(__file__)),
        "reused_selection_code_sha256": sha256(Path(__file__).with_name("shortlist_validation.py")),
        "elapsed_seconds": time.monotonic()-started,
        "all_length_dependence_eliminated": False,
        "humanization_quality_evaluated": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest

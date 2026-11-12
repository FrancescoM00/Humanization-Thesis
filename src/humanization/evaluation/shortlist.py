"""Selezione delle feature sul solo training e validazione per gruppi di un insieme compatto.

L'insieme fisso precedente è esplicitamente esplorativo. Questo modulo valida
una nuova procedura di selezione; non annulla la precedente selezione sul
campione completo. Le righe umano/macchina sono associazioni del disegno
sperimentale, non parafrasi semantiche presunte. Tutto il preprocessing e la
selezione supervisionata usano esclusivamente le righe di training.
"""
from __future__ import annotations

from collections import Counter
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from humanization.paths import resolve_path

from humanization.stylometry.features import FEATURE_GROUPS, FEATURE_SCHEMA, LENGTH_SENSITIVE_FEATURES, WIKIPEDIA_FEATURES


FIXED_FEATURES = ("pos_num", "pos_cconj", "mattr_50", "pos_propn")
DEFAULT_CV_SEEDS = (42, 137, 271, 811, 2026)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_documents(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida il disegno congelato e associa agli umani il generatore macchina abbinato.

    Questo esperimento ammette esattamente un umano e un testo macchina per gruppo
    di fonte. Se l'input cambia, si interrompe: non vengono modificati implicitamente
    né i pesi né il successivo calcolo dell'incertezza per gruppi.
    """
    required = {"id", "pair_id", "group_id", "domain", "generator", "label", "n_words"}
    missing = required.union(FEATURE_SCHEMA).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError("Every document needs a unique, nonmissing id")
    if frame[["pair_id", "group_id", "domain", "generator"]].isna().any().any():
        raise ValueError("Missing design metadata")
    if set(frame.label.unique()) != {0, 1}:
        raise ValueError("Both human=0 and machine=1 labels are required")
    if (frame.groupby("pair_id").group_id.nunique() != 1).any():
        raise ValueError("An association may not span source groups")
    for group, part in frame.groupby("group_id", sort=False):
        if len(part) != 2 or Counter(part.label) != {0: 1, 1: 1}:
            raise ValueError(f"Expected one human and one machine in group {group}")
        if part.domain.nunique() != 1 or part.pair_id.nunique() != 1:
            raise ValueError("This design requires one domain and one association per group")
        if part.n_words.nunique(dropna=False) != 1 or not np.isfinite(part.n_words).all():
            raise ValueError(f"Exact lexical word matching failed in group {group}")
    machines = frame.loc[frame.label == 1, ["pair_id", "generator"]].set_index("pair_id").generator
    result = frame[["id", "pair_id", "group_id", "domain", "label", "n_words"]].copy()
    result["source_generator"] = result.pair_id.map(machines)
    if result.source_generator.isna().any():
        raise ValueError("Missing machine generator for a design association")
    return result.reset_index(drop=True)


def select_training_features(
    train: pd.DataFrame,
    features: list[str],
    n_features: int = 4,
    max_abs_spearman: float = 0.85,
) -> tuple[list[str], pd.DataFrame]:
    """Ordina i valori osservati di |g di Hedges| sul solo training e riduce la ridondanza.

    Le correlazioni usano mediane stimate sulle stesse righe di training. Gli effetti
    non definiti, anche per varianza aggregata entro classe nulla, non entrano nella
    graduatoria. Le parità esatte sono risolte per nome della feature. Né le righe
    di validazione né l'appartenenza alla shortlist precedente influenzano la selezione.
    """
    if n_features < 1 or not 0 < max_abs_spearman <= 1:
        raise ValueError("Invalid selection settings")
    x = train[features].replace([np.inf, -np.inf], np.nan).astype(float)
    y = train.label.to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training selection needs both classes")
    moments = []
    for label in (0, 1):
        z = x.loc[y == label]
        n = z.count().to_numpy(dtype=float)
        means = z.mean().to_numpy()
        variances = z.var(ddof=1).to_numpy()
        moments.append((n, means, variances))
    n0, mean0, var0 = moments[0]
    n1, mean1, var1 = moments[1]
    dof = n0 + n1 - 2
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / dof)
        g = (1 - 3 / (4 * dof - 1)) * (mean1 - mean0) / pooled
    valid = (n0 >= 2) & (n1 >= 2) & (pooled > 0) & np.isfinite(g)
    report = pd.DataFrame({
        "feature": features,
        "feature_group": [FEATURE_GROUPS.get(name, "synthetic") for name in features],
        "n_human": n0.astype(int), "n_machine": n1.astype(int),
        "hedges_g": np.where(valid, g, np.nan),
        "abs_g": np.where(valid, np.abs(g), np.nan),
        "eligible": valid, "rank": np.nan, "selected": False,
        "reason": np.where((n0 < 2) | (n1 < 2), "insufficient_finite_class_values", "undefined_effect"),
        "correlated_with": "",
    }).set_index("feature", drop=False)
    candidates = report.loc[report.eligible].reset_index(drop=True).sort_values(
        ["abs_g", "feature"], ascending=[False, True])
    # Imputa solo dal training. Una colonna interamente mancante è comunque esclusa.
    corr = x.fillna(x.median()).fillna(0.0).corr(method="spearman").abs()
    selected: list[str] = []
    for rank, name in enumerate(candidates.feature, start=1):
        report.loc[name, "rank"] = rank
        if len(selected) >= n_features:
            report.loc[name, "reason"] = "maximum_size_reached"
            continue
        redundant = [other for other in selected if corr.loc[name, other] >= max_abs_spearman]
        if redundant:
            report.loc[name, "reason"] = "correlated_with_selected"
            report.loc[name, "correlated_with"] = redundant[0]
        else:
            selected.append(name)
            report.loc[name, ["selected", "reason"]] = [True, "selected"]
    if len(selected) != n_features:
        raise ValueError(f"Only {len(selected)} admissible features; need {n_features}; no threshold relaxation")
    return selected, report.reset_index(drop=True)


def build_model_registry(
    features: list[str], n_random_subsets: int = 100, random_seed: int = 20261111,
    fixed_features: tuple[str, ...] = FIXED_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estrae insiemi casuali distinti senza esaminare valori dei testi o etichette."""
    if not set(fixed_features).issubset(features):
        raise ValueError("The historical fixed panel is absent from the candidate pool")
    models = []
    def add(name, configuration, columns):
        models.append({"model": name, "configuration": configuration,
                       "features_json": json.dumps([] if columns is None else columns),
                       "n_features": 4 if columns is None else len(columns)})
    add("C1_full105", "C1_full105", features)
    add("C2_fixed4", "C2_fixed4", list(fixed_features))
    add("C3_training4", "C3_training4", None)
    add("C1_full100", "C1_full100", [f for f in features if f not in LENGTH_SENSITIVE_FEATURES])
    add("C1_style93", "C1_style93", [f for f in features if f not in LENGTH_SENSITIVE_FEATURES and f not in WIKIPEDIA_FEATURES])
    add("word_count_only", "word_count_only", ["n_words"])
    rng = np.random.default_rng(random_seed)
    random_rows = []
    pos = [f for f in features if FEATURE_GROUPS[f] == "pos"]
    lexical = [f for f in features if FEATURE_GROUPS[f] == "lexical"]
    for kind in ("uniform", "matched"):
        seen: set[tuple[str, ...]] = set()
        attempts = 0
        while len(seen) < n_random_subsets:
            attempts += 1
            if attempts > max(10000, n_random_subsets * 1000):
                raise ValueError("Requested too many distinct random panels")
            if kind == "uniform":
                selected = tuple(sorted(rng.choice(features, 4, replace=False).tolist()))
            else:
                selected = tuple(sorted(rng.choice(pos, 3, replace=False).tolist() + [str(rng.choice(lexical))]))
            if selected in seen:
                continue
            index = len(seen)
            seen.add(selected)
            name = f"C4_{kind}_{index:03d}"
            configuration = "C4_random" if kind == "uniform" else "C4_random_matched"
            add(name, configuration, list(selected))
            random_rows.append({"model": name, "configuration": configuration,
                                "draw_index": index, "random_seed": random_seed,
                                "features_json": json.dumps(selected)})
    return pd.DataFrame(models), pd.DataFrame(random_rows)


def iter_validation_splits(documents: pd.DataFrame, cv_seeds=DEFAULT_CV_SEEDS, n_splits: int = 5):
    """Produce fold comuni per gruppi e un test escluso effettivo per dominio/generatore."""
    all_indices = np.arange(len(documents))
    groups = documents.group_id.astype(str).to_numpy()
    for repeat, seed in enumerate(cv_seeds):
        # Quando ogni gruppo contiene [un umano, un testo macchina], sklearn 1.7.2
        # può produrre partizioni identiche con semi diversi: mescolare conteggi di
        # classe identici per gruppo non cambia il loro ordine. Si rietichettano quindi
        # prima i gruppi casualmente, senza usare feature o modificare gli abbinamenti.
        unique_groups = np.unique(groups)
        permutation = np.random.default_rng(int(seed)).permutation(len(unique_groups))
        randomized_codes = dict(zip(unique_groups, permutation))
        split_groups = np.asarray([randomized_codes[group] for group in groups])
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        for fold, (train, test) in enumerate(splitter.split(all_indices, documents.label, split_groups)):
            yield {"protocol": "cv", "repeat": repeat, "seed": int(seed), "fold": fold, "held_out": "ALL"}, train, test
    seed = int(cv_seeds[0])
    for protocol, column in (("leave_domain_out", "domain"), ("leave_generator_out", "source_generator")):
        for fold, value in enumerate(sorted(documents[column].unique())):
            test = all_indices[documents[column].eq(value).to_numpy()]
            test_groups = set(groups[test])
            train = all_indices[~documents.group_id.astype(str).isin(test_groups).to_numpy()]
            if protocol == "leave_generator_out":
                train = train[~documents.iloc[train].source_generator.eq(value).to_numpy()]
            yield {"protocol": protocol, "repeat": 0, "seed": seed, "fold": fold, "held_out": str(value)}, train, test


def run_validation(protocol_path: Path, output: Path, root: Path) -> dict:
    """Esegue una volta le impostazioni fisse, conservando gli esperimenti precedenti."""
    started = time.monotonic()
    protocol_path, output, root = protocol_path.resolve(), output.resolve(), root.resolve()
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    source = Path(protocol.get("input_csv", "results/published/02_stylometry/length_matched/features.csv"))
    source = resolve_path(root, source)
    source = source.resolve()
    input_hash = sha256(source)
    if not protocol.get("input_sha256"):
        raise ValueError("The frozen protocol must specify the input SHA-256")
    if protocol["input_sha256"] != input_hash:
        raise ValueError("Input features differ from the frozen protocol hash")
    if output == source.parent:
        raise ValueError("Use a new results directory; historical feature results are immutable")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("predictions.npz", "manifest.json")):
        raise FileExistsError("A completed run already exists; choose a new output directory")
    frame = pd.read_csv(source).reset_index(drop=True)
    documents = prepare_documents(frame)
    features = list(protocol.get("features", protocol.get("feature_pool", FEATURE_SCHEMA)))
    if set(features) != set(FEATURE_SCHEMA) or len(features) != len(FEATURE_SCHEMA):
        raise ValueError("The primary full candidate pool must contain the 105 fixed feature columns")
    cv_seeds = tuple(int(s) for s in protocol.get("cv_seeds", DEFAULT_CV_SEEDS))
    if not cv_seeds or len(set(cv_seeds)) != len(cv_seeds):
        raise ValueError("CV seeds must be nonempty and distinct")
    n_splits = int(protocol.get("n_splits", 5))
    selection = protocol.get("selection", {})
    n_selected = int(selection.get("n_features", 4))
    correlation_limit = float(selection.get("max_abs_spearman", 0.85))
    if n_selected != 4:
        raise ValueError("This comparison fixes the compact panel size to four")
    registry, random_panels = build_model_registry(
        features, int(protocol.get("n_random_subsets", 100)), int(protocol.get("random_seed", 20261111)),
        tuple(protocol.get("fixed_features", FIXED_FEATURES)))
    documents.to_csv(output / "documents.csv", index=False)
    registry.to_csv(output / "model_registry.csv", index=False)
    random_panels.to_csv(output / "random_subsets.csv", index=False)
    model_specs = registry.to_dict("records")
    predictions = {
        name: np.full((repeats, len(documents), len(registry)), np.nan, dtype=np.float64)
        for name, repeats in (("cv", len(cv_seeds)), ("leave_domain_out", 1), ("leave_generator_out", 1))
    }
    assignments = {name: np.zeros(values.shape[:2], dtype=np.int16) for name, values in predictions.items()}
    selection_rows, coefficient_rows, audit_rows, membership_rows, convergence = [], [], [], [], []
    logistic = protocol.get("logistic", {})
    fits = 0
    with threadpool_limits(limits=1):
        for info, train_indices, test_indices in iter_validation_splits(documents, cv_seeds, n_splits):
            train_docs, test_docs = documents.iloc[train_indices], documents.iloc[test_indices]
            train_groups, test_groups = set(train_docs.group_id), set(test_docs.group_id)
            overlap = len(train_groups.intersection(test_groups))
            if overlap or train_docs.label.nunique() != 2 or test_docs.label.nunique() != 2:
                raise AssertionError("Invalid train/test split or source group leakage")
            held_gen = int(train_docs.source_generator.eq(info["held_out"]).sum()) if info["protocol"] == "leave_generator_out" else 0
            held_domain = int(train_docs.domain.eq(info["held_out"]).sum()) if info["protocol"] == "leave_domain_out" else 0
            if held_gen or held_domain:
                raise AssertionError("Held-out generator/domain appeared in training")
            audit_rows.append({**info, "n_train_documents": len(train_docs), "n_test_documents": len(test_docs),
                               "n_train_groups": len(train_groups), "n_test_groups": len(test_groups),
                               "group_overlap": overlap, "heldout_generator_in_training": held_gen,
                               "heldout_domain_in_training": held_domain,
                               "test_domains_json": json.dumps(test_docs.domain.value_counts().to_dict()),
                               "test_generators_json": json.dumps(test_docs.source_generator.value_counts().to_dict())})
            for role, values in (("train", train_groups), ("test", test_groups)):
                membership_rows.extend({**info, "group_id": value, "role": role} for value in sorted(values))
            train, test = frame.iloc[train_indices], frame.iloc[test_indices]
            selected, diagnostic = select_training_features(train, features, n_selected, correlation_limit)
            selection_rows.extend({**info, **row} for row in diagnostic.to_dict("records"))
            for model_index, spec in enumerate(model_specs):
                columns = selected if spec["model"] == "C3_training4" else json.loads(spec["features_json"])
                model = make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                    LogisticRegression(C=float(logistic.get("C", 1.0)),
                                       class_weight=logistic.get("class_weight", "balanced"),
                                       max_iter=int(logistic.get("max_iter", 2000)),
                                       solver=logistic.get("solver", "lbfgs"),
                                       tol=float(logistic.get("tol", 1e-4)),
                                       random_state=info["seed"]),
                )
                x_train = train[columns].replace([np.inf, -np.inf], np.nan)
                x_test = test[columns].replace([np.inf, -np.inf], np.nan)
                with warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(x_train, train.label)
                for warning in recorded:
                    if issubclass(warning.category, ConvergenceWarning):
                        convergence.append({**info, "model": spec["model"], "message": str(warning.message)})
                values = model.predict_proba(x_test)[:, 1]
                if not np.isfinite(values).all():
                    raise AssertionError("Nonfinite held-out predictions")
                predictions[info["protocol"]][info["repeat"], test_indices, model_index] = values
                fits += 1
                if spec["model"] in {"C2_fixed4", "C3_training4"}:
                    imputer, scaler, classifier = model.steps[0][1], model.steps[1][1], model.steps[2][1]
                    for j, name in enumerate(columns):
                        coefficient_rows.append({**info, "model": spec["model"], "feature": name,
                                                 "coefficient_standardized": classifier.coef_[0, j],
                                                 "intercept": classifier.intercept_[0],
                                                 "imputation_median": imputer.statistics_[j],
                                                 "scaler_mean": scaler.mean_[j], "scaler_scale": scaler.scale_[j]})
            assignments[info["protocol"]][info["repeat"], test_indices] += 1
            print(f"{info['protocol']} seed={info['seed']} fold={info['fold']} held_out={info['held_out']}: "
                  f"{len(registry)} models; selected={','.join(selected)}; elapsed={time.monotonic()-started:.1f}s", flush=True)
    if any(not np.all(count == 1) for count in assignments.values()):
        raise AssertionError("Every document must have exactly one held-out prediction per protocol/repetition")
    if any(not np.isfinite(array).all() for array in predictions.values()):
        raise AssertionError("Incomplete prediction coverage")
    if sha256(source) != input_hash or protocol_path.read_bytes() != protocol_bytes:
        raise AssertionError("Frozen data or protocol changed during model fitting")
    temporary = output / "predictions.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **predictions)
    temporary.replace(output / "predictions.npz")
    pd.DataFrame(audit_rows).to_csv(output / "split_audit.csv", index=False)
    pd.DataFrame(membership_rows).to_csv(output / "split_membership.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "selections.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(output / "coefficients.csv", index=False)
    manifest = {
        "status": "completed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "input_csv": str(source.relative_to(root)) if source.is_relative_to(root) else str(source),
        "input_sha256": input_hash, "n_documents": len(documents),
        "n_groups": documents.group_id.nunique(), "n_associations": documents.pair_id.nunique(),
        "n_features": len(features), "n_models": len(registry), "n_fits": fits,
        "cv_seeds": list(cv_seeds), "prediction_shapes": {key: list(value.shape) for key, value in predictions.items()},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
        "code_sha256": sha256(Path(__file__)), "n_convergence_warnings": len(convergence),
        "convergence_warnings": convergence, "elapsed_seconds": time.monotonic()-started,
        "fitting_and_preprocessing": "training rows only; no tuning; source groups disjoint",
        "historical_fixed_panel_is_independently_validated": False,
        "semantic_alignment_of_human_machine_rows_is_assumed": False,
    }
    if protocol.get("fit_manifest_file"):
        (output / protocol["fit_manifest_file"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

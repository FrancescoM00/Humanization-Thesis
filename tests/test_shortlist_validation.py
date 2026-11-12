"""Checks of training-only selection, source-disjoint splits and paired AUC.

These fixtures exercise the failure modes of the new experiment. They do not
use the historical corpus or rerun any language models.
"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from stylometry.features import FEATURE_GROUPS, FEATURE_SCHEMA
from stylometry.shortlist_validation import (
    build_model_registry,
    iter_validation_splits,
    prepare_documents,
    run_validation,
    select_training_features,
    sha256,
)
from stylometry.shortlist_statistics import (
    pairwise_auc_kernel,
    weighted_auc_from_kernel,
)


def toy_frame(pairs_per_domain=8):
    rng = np.random.default_rng(117)
    rows = []
    for domain in ("Essay", "Reuters", "WritingPrompts"):
        for pair in range(pairs_per_domain):
            generator = "A" if pair % 2 else "B"
            for label in (0, 1):
                features = {name: float(rng.normal()) for name in FEATURE_SCHEMA}
                # All four fixed features remain numeric. The first is a
                # deliberately spurious effect confined to the held-out domain.
                features["pos_num"] = (1000 * label + rng.normal()) if domain == "WritingPrompts" else 0.0
                for name in ("pos_cconj", "mattr_50", "pos_propn"):
                    features[name] += label
                features["n_words"] = float(120 + pair)
                rows.append({"id": f"{domain}:{pair}:{label}", "pair_id": f"{domain}:{pair}",
                             "group_id": f"{domain}:{pair}", "domain": domain,
                             "generator": generator if label else "human", "label": label,
                             **features})
    return pd.DataFrame(rows)


class TrainingSelectionTests(unittest.TestCase):
    def test_effect_ranking_and_redundancy_use_absolute_training_correlation(self):
        rng = np.random.default_rng(8)
        y = np.repeat([0, 1], 40)
        signal = 4 * y + rng.normal(size=len(y))
        frame = pd.DataFrame({"label": y, "alpha": signal, "beta": signal.copy(),
                              "opposite": -signal, "other": y + rng.normal(size=len(y)),
                              "noise": rng.normal(size=len(y)), "constant": 0.0,
                              "missing": np.nan})
        selected, report = select_training_features(frame, list(frame.columns[1:]), n_features=3)
        # Exact tie-breaking picks alpha, and both +1 and -1 redundant copies
        # must be removed before filling the remaining two slots.
        self.assertEqual(selected, ["alpha", "other", "noise"])
        indexed = report.set_index("feature")
        self.assertEqual(indexed.loc["beta", "reason"], "correlated_with_selected")
        self.assertEqual(indexed.loc["opposite", "reason"], "correlated_with_selected")
        self.assertFalse(indexed.loc["constant", "eligible"])
        self.assertFalse(indexed.loc["missing", "eligible"])
        self.assertEqual(indexed.loc["missing", "n_human"], 0)

    def test_effect_ranking_is_invariant_to_positive_affine_units(self):
        rng = np.random.default_rng(9)
        y = np.repeat([0, 1], 30)
        frame = pd.DataFrame({"label": y, "f": 2 * y + rng.normal(size=len(y)),
                              "noise": rng.normal(size=len(y))})
        selected, before = select_training_features(frame, ["f", "noise"], n_features=1)
        changed = frame.copy()
        changed["f"] = changed.f * 1000 + 500
        again, after = select_training_features(changed, ["f", "noise"], n_features=1)
        self.assertEqual(selected, again)
        np.testing.assert_allclose(before.hedges_g, after.hedges_g, rtol=1e-12, atol=1e-12)

    def test_insufficient_admissible_features_do_not_relax_the_rule(self):
        frame = pd.DataFrame({"label": [0, 0, 1, 1], "a": [0, 1, 2, 3],
                              "b": [0, 2, 4, 6], "constant": 0})
        with self.assertRaisesRegex(ValueError, "no threshold relaxation"):
            select_training_features(frame, ["a", "b", "constant"], n_features=2)


class GroupedDesignTests(unittest.TestCase):
    def test_split_coverage_and_source_generator_protection(self):
        documents = prepare_documents(toy_frame())
        partitions = {}
        counts = {"cv": np.zeros((2, len(documents)), dtype=int),
                  "leave_domain_out": np.zeros((1, len(documents)), dtype=int),
                  "leave_generator_out": np.zeros((1, len(documents)), dtype=int)}
        for info, train, test in iter_validation_splits(documents, cv_seeds=(42, 137), n_splits=3):
            self.assertFalse(set(documents.iloc[train].group_id) & set(documents.iloc[test].group_id))
            counts[info["protocol"]][info["repeat"], test] += 1
            if info["protocol"] == "cv":
                partitions.setdefault(info["seed"], []).append(tuple(sorted(documents.iloc[test].group_id)))
            if info["protocol"] == "leave_generator_out":
                self.assertNotIn(info["held_out"], set(documents.iloc[train].source_generator))
                self.assertEqual(set(documents.iloc[test].source_generator), {info["held_out"]})
            elif info["protocol"] == "leave_domain_out":
                self.assertNotIn(info["held_out"], set(documents.iloc[train].domain))
        for values in counts.values():
            self.assertTrue(np.all(values == 1))
        # Identical class composition in every group must not make repeated
        # StratifiedGroupKFold seeds silently reproduce the same partition.
        self.assertNotEqual(sorted(partitions[42]), sorted(partitions[137]))

    def test_unsupported_duplicate_group_or_unmatched_length_stops(self):
        frame = toy_frame()
        frame.loc[frame.pair_id.isin(["Essay:0", "Essay:1"]), "group_id"] = "connected-four-rows"
        with self.assertRaisesRegex(ValueError, "one human and one machine"):
            prepare_documents(frame)
        frame = toy_frame()
        frame.loc[0, "n_words"] += 1
        with self.assertRaisesRegex(ValueError, "matching failed"):
            prepare_documents(frame)

    def test_random_panels_are_reproducible_unique_and_family_matched(self):
        a, panels = build_model_registry(list(FEATURE_SCHEMA), n_random_subsets=12, random_seed=55)
        b, again = build_model_registry(list(FEATURE_SCHEMA), n_random_subsets=12, random_seed=55)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(panels, again)
        for _, part in panels.groupby("configuration"):
            self.assertFalse(part.features_json.duplicated().any())
        for row in panels.itertuples():
            names = json.loads(row.features_json)
            self.assertEqual(len(set(names)), 4)
            if row.configuration == "C4_random_matched":
                families = [FEATURE_GROUPS[name] for name in names]
                self.assertEqual(families.count("pos"), 3)
                self.assertEqual(families.count("lexical"), 1)

    def test_end_to_end_domain_holdout_does_not_influence_selection_or_scaling(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "features.csv", root / "output"
            frame = toy_frame()
            frame.to_csv(source, index=False)
            protocol = {"input_csv": str(source), "input_sha256": sha256(source),
                        "features": list(FEATURE_SCHEMA), "cv_seeds": [42], "n_splits": 3,
                        "n_random_subsets": 1, "random_seed": 55,
                        "selection": {"n_features": 4, "max_abs_spearman": 0.85}}
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol))
            original = source.read_bytes()
            with redirect_stdout(io.StringIO()):
                manifest = run_validation(protocol_path, output, root)
            self.assertEqual(manifest["n_convergence_warnings"], 0)
            self.assertEqual(source.read_bytes(), original)
            selections = pd.read_csv(output / "selections.csv")
            target = selections[(selections.protocol == "leave_domain_out") &
                                (selections.held_out == "WritingPrompts") &
                                (selections.feature == "pos_num")].iloc[0]
            self.assertFalse(target.eligible)
            self.assertFalse(target.selected)
            coefficients = pd.read_csv(output / "coefficients.csv")
            fixed = coefficients[(coefficients.protocol == "leave_domain_out") &
                                 (coefficients.held_out == "WritingPrompts") &
                                 (coefficients.model == "C2_fixed4") &
                                 (coefficients.feature == "pos_num")].iloc[0]
            self.assertEqual(fixed.imputation_median, 0)
            self.assertEqual(fixed.scaler_mean, 0)
            self.assertEqual(fixed.scaler_scale, 1)
            with np.load(output / "predictions.npz") as predictions:
                for name in ("cv", "leave_domain_out", "leave_generator_out"):
                    self.assertTrue(np.isfinite(predictions[name]).all())


class PairedAUCTests(unittest.TestCase):
    def test_kernel_matches_weighted_sklearn_auc_with_ties_and_zero_weight_groups(self):
        positive = np.array([0.9, 0.5, 0.2, 0.5])
        negative = np.array([0.5, 0.5, 0.1, 0.8])
        kernel = pairwise_auc_kernel(positive, negative)
        draws = np.array([[1, 1, 1, 1], [0, 3, 1, 0], [2, 0, 1, 2]], dtype=float)
        actual = weighted_auc_from_kernel(kernel, draws)
        labels = np.r_[np.ones(4), np.zeros(4)]
        scores = np.r_[positive, negative]
        expected = [roc_auc_score(labels, scores, sample_weight=np.r_[weights, weights]) for weights in draws]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        self.assertAlmostEqual(weighted_auc_from_kernel(kernel, draws[0]), expected[0])

    def test_averaging_views_averages_auc_instead_of_ensembling_predictions(self):
        positive = np.array([[0.9, 0.8, 0.7], [0.1, 0.2, 0.3]])
        negative = np.array([[0.6, 0.5, 0.4], [0.2, 0.3, 0.4]])
        labels = np.r_[np.ones(3), np.zeros(3)]
        expected = np.mean([roc_auc_score(labels, np.r_[a, b]) for a, b in zip(positive, negative)])
        actual = weighted_auc_from_kernel(pairwise_auc_kernel(positive, negative), np.ones(3))
        ensemble_auc = roc_auc_score(labels, np.r_[positive.mean(axis=0), negative.mean(axis=0)])
        self.assertAlmostEqual(actual, expected)
        self.assertGreater(abs(actual - ensemble_auc), 0.1)


if __name__ == "__main__":
    unittest.main()

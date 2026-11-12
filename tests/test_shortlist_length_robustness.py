"""Failure-mode checks for reusing fixed splits while excluding diagnostics.

All fixtures are synthetic. The historical corpus and its predictions are not
fitted or modified by these tests.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from humanization.stylometry.features import FEATURE_SCHEMA, LENGTH_SENSITIVE_FEATURES
from humanization.evaluation.shortlist import run_validation, sha256
from humanization.evaluation.shortlist_statistics import analyze_results
from humanization.evaluation.shortlist_robustness import load_saved_splits, run_robustness


def synthetic_features():
    rng = np.random.default_rng(903)
    rows = []
    for domain in ("Essay", "Reuters", "WritingPrompts"):
        for pair in range(8):
            for label in (0, 1):
                values = {name: float(rng.normal()) for name in FEATURE_SCHEMA}
                values["n_words"] = float(120 + pair)
                # The excluded feature is deliberately the easiest predictor.
                values["ttr"] = 1000 * label + float(rng.normal(scale=.01))
                # A second strong effect exists only in the held-out domain.
                values["pos_num"] = (500 * label + float(rng.normal())) if domain == "WritingPrompts" else 0.0
                for name in ("mattr_50", "pos_cconj", "pos_propn", "punct_total_per100"):
                    values[name] += label
                group = f"{domain}:{pair}"
                rows.append({"id": f"{group}:{label}", "pair_id": group,
                             "group_id": group, "domain": domain,
                             "generator": ("A" if pair % 2 else "B") if label else "human",
                             "label": label, **values})
    return pd.DataFrame(rows)


class LengthRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.parent, cls.output = cls.root / "parent", cls.root / "robustness"
        cls.parent.mkdir()
        cls.output.mkdir()
        cls.source = cls.root / "features.csv"
        synthetic_features().to_csv(cls.source, index=False)
        logistic = {"C": 1.0, "class_weight": "balanced", "max_iter": 2000,
                    "solver": "lbfgs", "tol": 1e-4}
        selection = {"n_features": 4, "max_abs_spearman": .85}
        old_protocol = {"input_csv": "features.csv", "input_sha256": sha256(cls.source),
                        "features": list(FEATURE_SCHEMA), "cv_seeds": [42], "n_splits": 3,
                        "n_random_subsets": 1, "random_seed": 55,
                        "selection": selection, "logistic": logistic,
                        "n_bootstrap": 37, "bootstrap_seed": 20261112}
        (cls.parent / "protocol.json").write_text(json.dumps(old_protocol))
        with redirect_stdout(io.StringIO()):
            run_validation(cls.parent / "protocol.json", cls.parent, cls.root)
            analyze_results(cls.parent)
        cls.documents = pd.read_csv(cls.parent / "documents.csv")
        cls.membership = pd.read_csv(cls.parent / "split_membership.csv")
        cls.protocol = {
            "parent_results": "parent", "input_csv": "features.csv",
            "input_sha256": sha256(cls.source),
            "parent_protocol_sha256": sha256(cls.parent / "protocol.json"),
            "parent_predictions_sha256": sha256(cls.parent / "predictions.npz"),
            "parent_splits_sha256": sha256(cls.parent / "split_membership.csv"),
            "candidate_features": [name for name in FEATURE_SCHEMA if name not in LENGTH_SENSITIVE_FEATURES],
            "excluded_features": list(LENGTH_SENSITIVE_FEATURES),
            "selection": selection, "logistic": logistic, "cv_seeds": [42],
            "n_bootstrap": 37, "bootstrap_seed": 20261112,
        }
        cls.protocol_path = cls.output / "protocol.json"
        cls.protocol_path.write_text(json.dumps(cls.protocol))
        cls.parent_hashes = {p.name: sha256(p) for p in cls.parent.iterdir() if p.is_file()}
        from humanization.evaluation import shortlist_robustness as module
        original_pipeline = module.make_pipeline
        with patch.object(module, "make_pipeline", wraps=original_pipeline) as construction:
            with redirect_stdout(io.StringIO()):
                cls.manifest = run_robustness(cls.protocol_path, cls.output, cls.root)
            cls.pipeline_constructions = construction.call_count

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_saved_membership_is_used_even_if_it_differs_from_a_regenerated_fold(self):
        membership = self.membership[(self.membership.protocol == "cv") &
                                     (self.membership.fold == 0)].copy()
        test_group = membership.loc[membership.role == "test", "group_id"].iloc[0]
        train_group = membership.loc[membership.role == "train", "group_id"].iloc[0]
        membership.loc[membership.group_id == test_group, "role"] = "train"
        membership.loc[membership.group_id == train_group, "role"] = "test"
        splits = list(load_saved_splits(self.documents, membership))
        self.assertEqual(len(splits), 1)
        _, train, test = splits[0]
        actual = set(self.documents.iloc[test].group_id)
        expected = set(membership.loc[membership.role == "test", "group_id"])
        self.assertEqual(actual, expected)
        self.assertIn(train_group, actual)
        self.assertNotIn(test_group, actual)
        self.assertFalse(set(self.documents.iloc[train].group_id) & actual)

    def test_inconsistent_saved_group_roles_are_rejected(self):
        membership = self.membership[(self.membership.protocol == "cv") &
                                     (self.membership.fold == 0)].copy()
        duplicate = membership.iloc[[0]].copy()
        duplicate["role"] = "test" if duplicate.role.iloc[0] == "train" else "train"
        with self.assertRaisesRegex(ValueError, "more than one role"):
            list(load_saved_splits(self.documents, pd.concat([membership, duplicate])))

    def test_old_predictions_are_copied_and_only_new_models_are_fitted(self):
        registry = pd.read_csv(self.parent / "model_registry.csv")
        parent_position = int(np.flatnonzero(registry.model.eq("C3_training4"))[0])
        with np.load(self.parent / "predictions.npz") as parent:
            with np.load(self.output / "predictions.npz") as child:
                for protocol in ("cv", "leave_domain_out", "leave_generator_out"):
                    np.testing.assert_array_equal(child[protocol][:, :, 0], parent[protocol][:, :, parent_position])
                    self.assertTrue(np.isfinite(child[protocol][:, :, 1]).all())
        self.assertEqual(self.manifest["n_new_fits"], 8)
        self.assertEqual(self.pipeline_constructions, 8)
        self.assertEqual(self.manifest["n_refitted_parent_models"], 0)
        self.assertEqual(self.manifest["n_convergence_warnings"], 0)
        for name, old_hash in self.parent_hashes.items():
            self.assertEqual(sha256(self.parent / name), old_hash)
        self.assertEqual(sha256(self.source), self.protocol["input_sha256"])

    def test_excluded_strong_signal_cannot_enter_and_holdout_does_not_select_features(self):
        old = pd.read_csv(self.parent / "selections.csv")
        self.assertTrue(old.loc[old.feature.eq("ttr"), "selected"].all())
        new = pd.read_csv(self.output / "selections.csv")
        self.assertEqual(new.feature.nunique(), 100)
        self.assertFalse(set(new.feature) & set(LENGTH_SENSITIVE_FEATURES))
        self.assertTrue(new.groupby(["protocol", "seed", "fold"]).selected.sum().eq(4).all())
        target = new[(new.protocol == "leave_domain_out") &
                     (new.held_out == "WritingPrompts") & (new.feature == "pos_num")].iloc[0]
        self.assertFalse(target.eligible)
        self.assertFalse(target.selected)

    def test_forbidden_candidate_or_parent_hash_change_stops_before_fitting(self):
        bad = dict(self.protocol)
        bad["candidate_features"] = [*bad["candidate_features"][:-1], "ttr"]
        path = self.root / "bad_candidates.json"
        path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "Candidate pool"):
            run_robustness(path, self.root / "bad_output", self.root)
        bad = dict(self.protocol)
        bad["parent_predictions_sha256"] = "0" * 64
        path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "parent predictions differs"):
            run_robustness(path, self.root / "bad_output", self.root)

    def test_saved_reference_analysis_reproduces_parent_auc_and_intervals(self):
        from humanization.evaluation.shortlist_robustness_analysis import analyze_results as analyze_robustness
        with redirect_stdout(io.StringIO()):
            summary = analyze_robustness(self.output, root=self.root)
        self.assertTrue(summary["parent_C3_reproduced"])
        self.assertTrue(all(value < 1e-12 for value in summary["max_abs_parent_metric_errors"].values()))
        old_metrics = pd.read_csv(self.parent / "metrics.csv")
        old_metrics = old_metrics[old_metrics.configuration.eq("C3_training4")]
        new_metrics = pd.read_csv(self.output / "metrics.csv")
        reused = new_metrics[new_metrics.configuration.eq("C3_105")]
        joined = reused.merge(old_metrics, on=["protocol", "scope", "stratum"], suffixes=("_new", "_old"))
        for field in ("auc", "ci_low", "ci_high"):
            np.testing.assert_allclose(joined[field+"_new"], joined[field+"_old"],
                                       rtol=0, atol=1e-12, equal_nan=True)


if __name__ == "__main__":
    unittest.main()

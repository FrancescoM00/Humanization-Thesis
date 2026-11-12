"""Checks of pairing, inference direction, and held-out-source protection."""

import unittest

import numpy as np
import pandas as pd

from humanization.evaluation.stylometric import _evaluation_splits, effect_tables, evaluate, shortlist_features


def sample_frame(n_pairs=18):
    rng = np.random.default_rng(101)
    records = []
    for domain in ("Essay", "Reuters", "WritingPrompts"):
        for pair in range(n_pairs):
            generator = "A" if pair % 2 else "B"
            base = rng.normal()
            for label in (0, 1):
                records.append({"id": f"{domain}:{pair}:{label}", "pair_id": f"{domain}:{pair}",
                                "group_id": f"{domain}:{pair}", "domain": domain,
                                "generator": generator if label else "human", "label": label,
                                "signal": float(3 * label + base), "noise": rng.normal()})
    return pd.DataFrame(records)


class EffectTests(unittest.TestCase):
    def test_paired_standardized_effect_and_bootstrap_are_reproducible(self):
        records = []
        for i in range(4):
            for y in (0, 1):
                records.append({"id": f"{i}:{y}", "pair_id": str(i), "domain": "Essay",
                                "generator": "A" if y else "human", "label": y, "f": i + 2*y})
        df = pd.DataFrame(records)
        a = effect_tables(df, ["f"], seed=8, n_bootstrap=80)
        b = effect_tables(df, ["f"], seed=8, n_bootstrap=80)
        pd.testing.assert_frame_equal(a, b)
        row = a[a.scope == "pooled"].iloc[0]
        expected = (1 - 3/23) * 2 / np.std([0, 1, 2, 3], ddof=1)
        self.assertAlmostEqual(row.hedges_g, expected)
        self.assertEqual(row.test_type, "paired_t")
        self.assertEqual(row.n_paired, 4)
        self.assertEqual(row.p_value, 0)
        self.assertTrue(np.isfinite(row.ci_low))
        self.assertGreater(row.ci_high, row.ci_low)

    def test_generator_effect_uses_its_matched_humans(self):
        df = sample_frame(12)
        # Put A-source humans and machines into a deliberately different range.
        sources = set(df.loc[df.generator == "A", "pair_id"])
        df.loc[df.pair_id.isin(sources), "signal"] += 100
        effects = effect_tables(df, ["signal"], n_bootstrap=0)
        row = effects[(effects.scope == "generator") & (effects.generator == "A")].iloc[0]
        self.assertEqual(row.n_human, row.n_machine)
        self.assertEqual(row.n_paired, row.n_machine)
        self.assertGreater(row.human_mean, 99)
        self.assertAlmostEqual(row.machine_mean-row.human_mean, 3)

    def test_shortlist_requires_effect_in_each_domain(self):
        df = sample_frame()
        # Add an association which reverses in one domain; pooled significance
        # or magnitude should not make it into the stable shortlist.
        df["unstable"] = df.signal
        df.loc[df.domain == "Reuters", "unstable"] = -df.loc[df.domain == "Reuters", "signal"]
        effects = effect_tables(df, ["signal", "unstable"], n_bootstrap=0)
        selected, robust = shortlist_features(df, effects, ["signal", "unstable"])
        self.assertIn("signal", selected)
        self.assertNotIn("unstable", selected)
        self.assertFalse(robust.set_index("feature").loc["unstable", "eligible"])

    def test_duplicate_sources_do_not_inflate_paired_test_units(self):
        df = sample_frame(12)
        df.loc[df.pair_id.isin(["Essay:0", "Reuters:0"]), "group_id"] = "duplicate-component"
        effects = effect_tables(df, ["signal"], n_bootstrap=0)
        row = effects[effects.scope == "pooled"].iloc[0]
        self.assertEqual(row.n_paired, 36)
        self.assertEqual(row.n_test_groups, 35)
        self.assertEqual(row.test_type, "paired_cluster_t")


class SplitTests(unittest.TestCase):
    def test_cross_generator_and_cross_domain_remove_duplicate_components(self):
        df = sample_frame(12)
        # Different source IDs connected by an exact-duplicate component,
        # deliberately spanning domains. Every protocol must respect it.
        duplicate = df.pair_id.isin(["Essay:0", "Reuters:0"])
        df.loc[duplicate, "group_id"] = "duplicate-component"
        for protocol, domain, generator, fold, train, test in _evaluation_splits(df, 42, 3):
            self.assertFalse(set(df.loc[train, "group_id"]) & set(df.loc[test, "group_id"]))
            if protocol == "leave_generator_out":
                self.assertNotIn(generator, set(df.loc[train, "generator"]))
                self.assertEqual(set(df.loc[test].loc[lambda z: z.label == 1, "generator"]), {generator})
                counts = df.loc[test].label.value_counts()
                self.assertEqual(counts[0], counts[1])
            if protocol == "leave_domain_out":
                self.assertNotIn(domain, set(df.loc[train, "domain"]))

    def test_reported_metrics_use_complete_disjoint_oof_predictions(self):
        df = sample_frame()
        metrics, predictions = evaluate(df, {"all": ["signal", "noise"]}, n_splits=3, n_bootstrap=20)
        self.assertEqual(len(metrics), 8)  # 3 within + 3 held-domain + 2 held-generator
        self.assertTrue((predictions.train_group_overlap == 0).all())
        for _, part in predictions.groupby(["protocol", "domain", "generator", "feature_set"]):
            self.assertFalse(part.id.duplicated().any())
        within = predictions[predictions.protocol == "within_domain"]
        self.assertEqual(set(within.id), set(df.id))
        held_gen = predictions[predictions.protocol == "leave_generator_out"]
        self.assertEqual(set(held_gen.id), set(df.id))
        self.assertTrue((metrics.roc_auc > .9).all())
        self.assertTrue((metrics.roc_auc_ci_low <= metrics.roc_auc_ci_high).all())


if __name__ == "__main__":
    unittest.main()

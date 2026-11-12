"""Controlli mirati per il nuovo pannello delle cautele, senza inferenza di modelli neurali."""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from humanization.evaluation.hedging import (
    CONDITIONS, HISTORICAL_THRESHOLD, analyze, grouped_metrics, load_panel, load_review, wilson95,
)
from humanization.semantics.hedging_controls import (
    collect_exclusions, discover_candidates, load_machine_records, select_sources,
    sentence_triplet, source_pairs, text_digest,
)
from humanization.semantics.perturbations import _stable_hex, load_perturbation_nlp
from humanization.datasets.mgtbench import DOMAINS, GENERATORS


def scored_frame(n_sources=5):
    rows = []
    for i in range(n_sources):
        for condition in CONDITIONS:
            rows.append({"id": f"source-{i}:{condition}", "source_group": f"source-{i}",
                         "condition": condition, "is_altered": int(condition != "hedging_preserved"),
                         "document_support": .8 if condition == "hedging_preserved" else .02,
                         "local_support": .8 if condition == "hedging_preserved" else .2})
    return pd.DataFrame(rows)


def write_panel(directory):
    directory = Path(directory)
    frame = scored_frame()
    pairs, scores = [], []
    for row in frame.itertuples(index=False):
        pairs.append({"id": row.id, "source_group_id": row.source_group,
                      "condition": row.condition, "label": row.is_altered,
                      "domain": "Essay", "generator": "toy", "original": "The effect may occur.",
                      "modified": "The effect occurs." if row.is_altered else "The effect might occur.",
                      "target_original": "The effect may occur.",
                      "target_modified": "The effect occurs." if row.is_altered else "The effect might occur."})
        scores.append({"id": row.id, "nli_forward_min": row.document_support, "nli_reverse_min": 1.0,
                       "local_nli_forward_min": row.local_support, "local_nli_reverse_min": 1.0,
                       "nli_retrieval_mode": "top_k_cosine", "nli_retrieval_top_k": 3,
                       "nli_max_pair_tokens": 384})
    pairs_path, scores_path = directory / "pairs.jsonl", directory / "nli.csv"
    pairs_path.write_text("".join(json.dumps(row) + "\n" for row in pairs))
    pd.DataFrame(scores).to_csv(scores_path, index=False)
    return pairs_path, scores_path


class FrozenThresholdTests(unittest.TestCase):
    def test_wilson_is_secondary_and_not_applied_to_two_variants_per_original(self):
        self.assertEqual(wilson95(0, 60)[0], 0)
        self.assertGreater(wilson95(0, 60)[1], 0)
        self.assertEqual(wilson95(60, 60)[1], 1)
        self.assertLess(wilson95(60, 60)[0], 1)
        frame = scored_frame()
        combined, _ = grouped_metrics(frame, "document_support", HISTORICAL_THRESHOLD,
                                       np.random.default_rng(1), 32)
        self.assertFalse(combined["sensitivity_wilson95_eligible"])
        self.assertTrue(combined["false_positive_rate_wilson95_eligible"])
        self.assertEqual(combined["false_positive_rate_wilson95_n_sources"], 5)
        subtype = frame[frame.condition.ne("hedging_partial")]
        one_per_source, _ = grouped_metrics(subtype, "document_support", HISTORICAL_THRESHOLD,
                                            np.random.default_rng(1), 32)
        self.assertTrue(one_per_source["sensitivity_wilson95_eligible"])
        self.assertEqual(one_per_source["sensitivity_wilson95_n_sources"], 5)

    def test_exact_boundary_and_risk_orientation_are_frozen(self):
        frame = scored_frame()
        tau = HISTORICAL_THRESHOLD
        for i in range(5):
            frame.loc[frame.id.eq(f"source-{i}:hedging_preserved"), "document_support"] = tau + .2
            frame.loc[frame.id.eq(f"source-{i}:hedging_complete"), "document_support"] = tau - 1e-6
            frame.loc[frame.id.eq(f"source-{i}:hedging_partial"), "document_support"] = tau + .1
        # Un valore uguale alla soglia NON è un allarme. La perturbazione al di sotto della soglia lo è.
        frame.loc[frame.id.eq("source-0:hedging_preserved"), "document_support"] = tau
        frame.loc[frame.id.eq("source-0:hedging_complete"), "document_support"] = tau
        frame.loc[frame.id.eq("source-0:hedging_partial"), "document_support"] = tau - 1e-6
        result, draws = grouped_metrics(frame, "document_support", tau, np.random.default_rng(1), 32)
        self.assertEqual(result["true_positive_count"], 5)
        self.assertEqual(result["false_negative_count"], 5)
        self.assertEqual(result["false_positive_count"], 0)
        self.assertEqual(result["true_negative_count"], 5)
        self.assertEqual(result["sensitivity"], .5)
        self.assertEqual(result["false_positive_rate"], 0)
        expected = roc_auc_score(frame.is_altered, -frame.document_support,
                                 sample_weight=np.where(frame.is_altered.eq(1), .5, 1.0))
        self.assertAlmostEqual(result["auc"], expected)
        self.assertEqual(draws["auc"].shape, (32,))
        # Un ordinamento errato deve restare tale; non invertire mai il punteggio dopo aver visto AUC<.5.
        reversed_frame = scored_frame()
        reversed_frame["document_support"] = np.where(reversed_frame.is_altered.eq(1), .9, .1)
        reversed_result, _ = grouped_metrics(reversed_frame, "document_support", tau,
                                            np.random.default_rng(1), 32)
        self.assertEqual(reversed_result["auc"], 0.0)

    def test_source_weighting_differs_from_raw_count_when_reviews_leave_unequal_variants(self):
        frame = scored_frame()
        frame = frame[~(frame.condition.eq("hedging_partial") & frame.source_group.ne("source-0"))].copy()
        frame["document_support"] = .8
        frame.loc[frame.source_group.eq("source-0"), "document_support"] = .02
        result, _ = grouped_metrics(frame, "document_support", HISTORICAL_THRESHOLD,
                                    np.random.default_rng(2), 32)
        self.assertAlmostEqual(result["sensitivity"], 1/5)
        self.assertAlmostEqual(result["sensitivity_raw"], 2/6)
        self.assertAlmostEqual(result["false_positive_rate"], 1/5)
        self.assertEqual(result["true_positive_count"], 2)
        self.assertEqual(result["false_negative_count"], 4)
        self.assertEqual(result["false_positive_count"], 1)
        self.assertEqual(result["true_negative_count"], 4)
        weights = 1/frame.groupby(["source_group", "is_altered"]).id.transform("count")
        self.assertAlmostEqual(result["auc"], roc_auc_score(frame.is_altered, -frame.document_support,
                                                           sample_weight=weights))

    def test_unmatched_source_and_missing_local_unit_are_not_used_as_matched_controls(self):
        frame = scored_frame()
        extra = frame.iloc[0].copy()
        extra["id"], extra["source_group"] = "unmatched", "unmatched"
        frame = pd.concat([frame, extra.to_frame().T], ignore_index=True)
        result, _ = grouped_metrics(frame, "document_support", HISTORICAL_THRESHOLD,
                                    np.random.default_rng(3), 32)
        self.assertEqual(result["n_sources"], 5)
        self.assertEqual(result["n_missing_or_unmatched_pairs"], 1)
        local_frame = scored_frame()
        local_frame.loc[(local_frame.source_group == "source-0") &
                        local_frame.condition.eq("hedging_preserved"), "local_support"] = np.nan
        local_result, _ = grouped_metrics(local_frame, "local_support", None,
                                          np.random.default_rng(3), 32)
        self.assertEqual(local_result["n_sources"], 4)
        self.assertEqual(local_result["n_missing_or_unmatched_pairs"], 3)
        self.assertIsNone(local_result["threshold_support"])
        self.assertNotIn("sensitivity", local_result)
        self.assertNotIn("false_positive_rate", local_result)

    def test_analyzer_refuses_a_recalibrated_threshold_and_changed_retrieval(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pairs, scores = write_panel(root)
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"threshold_support": .5, "n_bootstrap": 32}))
            with self.assertRaisesRegex(ValueError, "must remain unchanged"):
                analyze(pairs, scores, root / "out", protocol)
            changed = pd.read_csv(scores)
            changed["nli_retrieval_top_k"] = 4
            changed.to_csv(scores, index=False)
            with self.assertRaisesRegex(ValueError, "different nli_retrieval_top_k"):
                load_panel(pairs, scores)

    def test_unreviewed_qualified_and_blank_human_forms_are_not_promoted(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pairs, scores = write_panel(root)
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"threshold_support": HISTORICAL_THRESHOLD,
                                            "n_bootstrap": 32, "bootstrap_seed": 5}))
            review_rows = []
            for i in range(5):
                review_rows.append({"id": f"source-{i}:hedging_preserved", "validity": "valid",
                                    "reviewer_source": "assistant", "notes": "toy review"})
                review_rows.append({"id": f"source-{i}:hedging_complete",
                                    "validity": "valid" if i < 2 else "qualified" if i < 4 else "invalid",
                                    "reviewer_source": "assistant", "notes": "toy review"})
            # Un modulo umano vuoto non costituisce un'etichetta semantica umana indipendente.
            review_rows.append({"id": "source-0:hedging_partial", "validity": "",
                                "reviewer_source": "human", "notes": ""})
            review = root / "review.csv"
            pd.DataFrame(review_rows).to_csv(review, index=False)
            summary = analyze(pairs, scores, root / "out", protocol, review)
            self.assertFalse(summary["independent_human_validation_available"])
            self.assertEqual(summary["n_pairs"], 15)
            metrics = pd.read_csv(root / "out/metrics.csv")
            doc = metrics[metrics.metric.eq("document_min")]
            primary = doc[doc.panel.eq("rule_intended_full") & doc.scope.eq("all")].iloc[0]
            valid = doc[doc.panel.eq("assistant_review_valid") & doc.scope.eq("all")].iloc[0]
            qualified = doc[doc.panel.eq("assistant_review_valid_qualified") & doc.scope.eq("all")].iloc[0]
            self.assertEqual(primary.n_sources, 5)
            self.assertEqual(valid.n_sources, 2)
            self.assertEqual(qualified.n_sources, 4)
            self.assertFalse(doc[doc.panel.ne("rule_intended_full")].scope.eq("partial").any())
            self.assertFalse(metrics[metrics.metric.eq("local_min")].sensitivity.notna().any())
            # Una provenienza mancante resta non specificata, senza dedurre che sia umana.
            pd.DataFrame([{"id": "x", "validity": "valid"}]).to_csv(root / "anonymous.csv", index=False)
            self.assertEqual(load_review(root / "anonymous.csv").iloc[0].reviewer_source, "unspecified")


class NativeHedgeTransformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Su queste frasi sintetiche viene eseguito soltanto il parser piccolo di spaCy già disponibile;
        # nessuna NLI, generazione linguistica, download o preparazione del corpus reale.
        cls.nlp = load_perturbation_nlp()

    def triple(self, text):
        return sentence_triplet(next(self.nlp(text).sents))

    def test_modal_state_and_perfect_preserve_finite_agreement_and_caution_control(self):
        for text, expected in (
            ("The treatment may be effective in this setting.", "The treatment is effective in this setting."),
            ("The treatments may be effective in these settings.", "The treatments are effective in these settings."),
            ("The treatment may have caused these minor effects.", "The treatment has caused these minor effects."),
            ("The treatments may have caused these minor effects.", "The treatments have caused these minor effects."),
            ("The treatment may improve the final outcome.", "The treatment improves the final outcome."),
            ("The measure may vary across different population groups.", "The measure varies across different population groups."),
            ("A treatment may improve the final outcome.", "A treatment improves the final outcome."),
        ):
            with self.subTest(text=text):
                triple = self.triple(text)
                self.assertIsNotNone(triple)
                self.assertEqual(triple["complete"], expected)
                self.assertIn("probably", triple["partial"].lower())
                self.assertTrue(triple["preserved"].startswith("It is possible that "))
                self.assertFalse(triple["preserved"].startswith("It is possible that A "))
                self.assertEqual(len({text, triple["complete"], triple["partial"], triple["preserved"]}), 4)

    def test_proposition_wrapper_and_probability_adverb_have_distinct_partial_condition(self):
        possible = self.triple("It is possible that the intervention reduces the risk.")
        self.assertIsNotNone(possible)
        self.assertEqual(possible["complete"], "The intervention reduces the risk.")
        self.assertEqual(possible["partial"], "It is likely that the intervention reduces the risk.")
        self.assertEqual(possible["preserved"], "Possibly, the intervention reduces the risk.")
        probability = self.triple("The treatment probably reduces the risk in adults.")
        self.assertIsNotNone(probability)
        self.assertEqual(probability["complete"], "The treatment reduces the risk in adults.")
        self.assertIn("almost certainly", probability["partial"])
        self.assertTrue(probability["preserved"].startswith("It is probable that "))

    def test_permission_negation_quotes_conditional_and_redundant_hedges_are_rejected(self):
        for text in (
            "The participants may enter the room after the break.",
            "The participants may have refreshments during the break.",
            "You may have a cookie after the examination.",
            "The treatment may not be effective in this setting.",
            '"The treatment may be effective in this setting."',
            "If effective, the treatment may improve the final outcome.",
            "The treatment could possibly reduce the risk in adults.",
            "The treatment may possibly be effective in this setting.",
            "The treatment may reduce the risk and improve outcomes.",
            "The study began in May and ended in June.",
            "The essay length (or word count) may vary.",
            "- A brief comment about cocoa prices in other markets may be included.",
            "The Reaper sensed Drakov might prove a worthy substitute.",
            "Those countries may be able to benefit from cheap labor.",
        ):
            with self.subTest(text=text):
                self.assertIsNone(self.triple(text))

    def test_document_edit_is_local_and_controls_are_rewrites_not_identities_or_gold(self):
        original = "The treatment may be effective in this setting. The report describes the procedure in detail."
        sentence = next(self.nlp(original).sents)
        triple = sentence_triplet(sentence)
        self.assertIsNotNone(triple)
        source = {"source_id": "toy", "source_group_id": "toy-group", "source_group": "toy-group",
                  "domain": "Essay", "generator": "toy", "source_row": 0,
                  "original": original, "target_original": sentence.text,
                  "target_start_char": sentence.start_char, "target_end_char": sentence.end_char,
                  "sentence_index": 0, "family": triple["family"], "marker": triple["marker"],
                  "canonical_original_sha256": text_digest(original),
                  "targets": {name: triple[name] for name in ("complete", "partial", "preserved")}}
        pairs = source_pairs(source)
        self.assertEqual(len(pairs), 3)
        self.assertEqual({pair["source_group_id"] for pair in pairs}, {"toy-group"})
        self.assertEqual(len({original, *[pair["modified"] for pair in pairs]}), 4)
        for pair in pairs:
            self.assertEqual(pair["original"], original)
            self.assertTrue(pair["modified"].endswith("The report describes the procedure in detail."))
            self.assertFalse(pair["metadata"]["semantic_label_verified"])
            self.assertEqual(pair["label_source"], "rule_intent_not_human_gold")

    def test_prefetched_duplicate_group_does_not_consume_generator_cap_or_domain_quota(self):
        # Forza la fonte B duplicata a precedere la fonte B indipendente nella
        # coda deterministica reale; A accetta per prima quel gruppo condiviso.
        duplicate_id, fresh_id = sorted(("B-source-0", "B-source-1"),
                                        key=lambda value: _stable_hex(7, value))
        records = [
            {"source_id": "A-source", "source_group_id": "shared", "domain": "Essay",
             "generator": "A", "source_row": 0,
             "original": "The treatment may be effective in this setting."},
            {"source_id": duplicate_id, "source_group_id": "shared", "domain": "Essay",
             "generator": "B", "source_row": 0,
             "original": "The treatment may be effective in another setting."},
            {"source_id": fresh_id, "source_group_id": "independent", "domain": "Essay",
             "generator": "B", "source_row": 1,
             "original": "The method may be effective in this setting."},
        ]
        parser = self.nlp

        class PrefetchingParser:
            def pipe(self, inputs, *, as_tuples, batch_size):
                # Simula un parser che legge un batch prima di restituirne i documenti.
                queued = list(inputs)
                for text, record in queued:
                    yield parser(text), record

        candidates, diagnostics = discover_candidates(records, PrefetchingParser(), seed=7,
                                                       max_per_cell=1, max_per_domain=2)
        self.assertEqual(len(candidates), 2)
        self.assertEqual({row["source_group_id"] for row in candidates}, {"shared", "independent"})
        self.assertEqual({row["generator"] for row in candidates}, {"A", "B"})
        self.assertTrue(any(row["reason"] == "source_group_already_accepted_in_prefetched_batch"
                            for row in diagnostics))
        limited, _ = discover_candidates(records, PrefetchingParser(), seed=7,
                                         max_per_cell=1, max_per_domain=1)
        self.assertEqual(len(limited), 1)


class IndependentSourceTests(unittest.TestCase):
    def test_previous_rows_connected_prompts_humans_and_canonical_texts_are_all_excluded(self):
        import csv
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            previous = "The treatment may be effective in the previous context."
            fixtures = {
                "Essay": [
                    ("Shared human text.", "shared prompt", "The method may improve an excluded essay outcome."),
                    ("Fresh essay human.", "fresh essay prompt", "The method may improve a fresh essay outcome."),
                ],
                "Reuters": [
                    ("Reuters human text.", "shared prompt", "The method may improve an excluded Reuters outcome."),
                    ("Fresh Reuters human.", "fresh Reuters prompt", "  THE treatment may be effective in the previous context.  "),
                ],
                "WritingPrompts": [
                    ("Shared human text.", "narrative prompt", "The method may improve an excluded narrative outcome."),
                    ("Fresh narrative human.", "fresh narrative prompt", "The method may improve a fresh narrative outcome."),
                ],
            }
            for domain, prefix in DOMAINS.items():
                with (raw / f"{prefix}_LLMs.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["id", "human", "prompt", *GENERATORS])
                    writer.writeheader()
                    for index, (human, prompt, machine) in enumerate(fixtures[domain]):
                        writer.writerow({"id": index+1, "human": human, "prompt": prompt,
                                         **{name: machine if name == "ChatGPT" else "" for name in GENERATORS}})
            pd.DataFrame([{"domain": "Essay", "source_row": 0}]).to_csv(root / "old.csv", index=False)
            (root / "old.jsonl").write_text(json.dumps({"source_id": "mgtbench:Reuters:99:old",
                                                          "original": previous}) + "\n")
            excluded_rows, excluded_hashes, _ = collect_exclusions(["old.csv", "old.jsonl"], root)
            self.assertIn(("Essay", 0), excluded_rows)
            self.assertIn(("Reuters", 99), excluded_rows)
            self.assertIn(text_digest(previous), excluded_hashes)
            records, manifest = load_machine_records(raw, ["old.csv", "old.jsonl"], root,
                                                     minimum_words=1, maximum_words=1000)
            self.assertEqual({(row["domain"], row["source_row"]) for row in records},
                             {("Essay", 1), ("WritingPrompts", 1)})
            self.assertTrue(all(row["canonical_original_sha256"] not in excluded_hashes for row in records))
            self.assertEqual(manifest["n_excluded_connected_components"], 1)

    def test_source_selection_is_deterministic_balanced_and_never_duplicates_components(self):
        candidates = []
        for domain in ("Essay", "Reuters", "WritingPrompts"):
            for i in range(4):
                candidates.append({"domain": domain, "generator": "A" if i % 2 else "B",
                                   "family": "modal_be" if i % 2 else "explicit_possible",
                                   "source_id": f"{domain}:{i}", "source_group_id": f"{domain}:{i//2}"})
        selected = select_sources(candidates, 6, seed=3)
        again = select_sources(list(reversed(candidates)), 6, seed=3)
        self.assertEqual(selected, again)
        self.assertEqual(len({row["source_group_id"] for row in selected}), 6)
        self.assertEqual(pd.Series([row["domain"] for row in selected]).value_counts().to_dict(),
                         {"Essay": 2, "Reuters": 2, "WritingPrompts": 2})


if __name__ == "__main__":
    unittest.main()

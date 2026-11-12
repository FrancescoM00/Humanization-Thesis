import unittest

from semantics.opai import align_opai_records


def trajectory(doc="source-a", generator="editor-a", domain="news"):
    return [
        {"document_hash_id": doc, "domain": domain, "generator": generator,
         "version": f"v{v}", "record_id": f"{doc}:{generator}:{v}",
         "edit_operation": op, "text": text,
         "sentence_ids": ["s0", "s1"], "sentences": sentences}
        for v, op, text, sentences in [
            (0, "none", "Human source. Some detail.", ["Human source.", "Some detail."]),
            (3, "style", "The long claim is true. An extra detail is included.", ["The long claim is true.", "An extra detail is included."]),
            (4, "compress", "The claim is true. A detail.", ["The claim is true.", "A detail."]),
            (5, "expand", "The claim is demonstrably true. A detail with explanation.", ["The claim is demonstrably true.", "A detail with explanation."]),
        ]
    ]


class OpAITests(unittest.TestCase):
    def test_adjacent_stages_and_secondary_v0_are_distinguished(self):
        pairs, manifest = align_opai_records(trajectory(), n_per_operation=1)
        self.assertEqual(len(pairs), 4)
        compress = [p for p in pairs if p["perturbation"] == "opai_compress" and p["metadata"]["comparison"] == "adjacent_operation"][0]
        self.assertEqual(compress["metadata"]["from_version"], "v3")
        self.assertEqual(compress["metadata"]["to_version"], "v4")
        self.assertEqual(len(compress["metadata"]["aligned_changed_sentences"]), 2)
        self.assertTrue(compress["metadata"]["expected_length_direction_observed"])
        expand = [p for p in pairs if p["perturbation"] == "opai_expand" and p["metadata"]["comparison"] == "adjacent_operation"][0]
        self.assertEqual(expand["metadata"]["from_version"], "v4")
        self.assertTrue(expand["metadata"]["expected_length_direction_observed"])
        self.assertEqual(manifest["n_trajectories"], 1)

    def test_generators_never_cross_align_and_sampling_is_reproducible(self):
        records = sum([trajectory(str(d), g, dom) for d in range(4) for g in ["A", "B"] for dom in ["news", "essay"]], [])
        a, ma = align_opai_records(records, n_per_operation=7, seed=9)
        b, mb = align_opai_records(records, n_per_operation=7, seed=9)
        self.assertEqual(a, b)
        self.assertEqual(ma, mb)
        self.assertEqual(sum(p["metadata"]["comparison"] == "adjacent_operation" for p in a), 14)
        self.assertEqual(len({p["id"] for p in a}), len(a))

    def test_ambiguous_version_or_missing_stable_source_fails(self):
        records = trajectory()
        records.append({**records[1], "text": "Conflicting alternate version."})
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            align_opai_records(records)
        record = trajectory()[0]
        del record["document_hash_id"]
        with self.assertRaisesRegex(ValueError, "Missing required"):
            align_opai_records([record])

    def test_incomplete_trajectory_does_not_invent_pairs(self):
        records = [r for r in trajectory() if r["version"] == "v0"]
        with self.assertRaisesRegex(ValueError, "No complete"):
            align_opai_records(records)

    def test_equal_sentence_counts_do_not_imply_alignment(self):
        records = trajectory()
        for record in records:
            del record["sentence_ids"]
        pairs, _ = align_opai_records(records)
        self.assertTrue(all(p["metadata"]["aligned_changed_sentences"] == [] for p in pairs))

    def test_blank_optional_csv_arrays_are_absent(self):
        records = trajectory()
        for record in records:
            record["sentences"] = ""
            record["sentence_ids"] = ""
        pairs, _ = align_opai_records(records)
        self.assertEqual(len(pairs), 4)
        self.assertTrue(all(p["metadata"]["aligned_changed_sentences"] == [] for p in pairs))


if __name__ == "__main__":
    unittest.main()

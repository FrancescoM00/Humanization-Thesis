"""Unit fixtures exercise data handling; they are never benchmark results."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from humanization.datasets.mgtbench import DOMAINS, canonical, prepare_dataset, process_spaces, trim_incomplete_tail


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.gens = ["Claude", "ChatGPT"]
        for domain, short in DOMAINS.items():
            rows = []
            for i in range(12):
                rows.append({"id": str(i), "prompt": f"{domain} prompt {i}",
                             "human": f"Human {domain} {i}. " + "word " * 50 + ".",
                             "Claude": f"Claude {domain} {i}. " + "another " * 50 + ".",
                             "ChatGPT": f"ChatGPT {domain} {i}. " + "more " * 50 + "."})
            pd.DataFrame(rows).to_csv(self.raw / f"{short}_LLMs.csv", index=False)

    def tearDown(self):
        self.temp.cleanup()

    def run_prepare(self, out="prepared", **kwargs):
        return prepare_dataset(self.raw, self.root / out, n_pairs=8, generators=self.gens, **kwargs)

    def test_balanced_reproducible_and_paired(self):
        first, manifest = self.run_prepare(seed=42)
        second, second_manifest = self.run_prepare("second", seed=42)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(manifest["samples_sha256"], second_manifest["samples_sha256"])
        self.assertEqual(len(first), 48)
        self.assertTrue(first.groupby("pair_id")["label"].apply(lambda x: sorted(x) == [0, 1]).all())
        for _, subset in first.groupby("domain"):
            self.assertEqual(subset["label"].value_counts().to_dict(), {0: 8, 1: 8})
            self.assertEqual(subset[subset.label == 1].generator.value_counts().to_dict(), {"Claude": 4, "ChatGPT": 4})
        self.assertEqual(first.text.map(canonical).nunique(), len(first))
        self.assertTrue((self.root / "prepared" / "samples.jsonl").is_file())
        saved = json.loads((self.root / "prepared" / "manifest.json").read_text())
        self.assertFalse(saved["download_provenance"]["official_bytes_verified"])

    def test_shared_prompt_stays_in_same_group(self):
        for path in self.raw.glob("*.csv"):
            frame = pd.read_csv(path)
            frame["prompt"] = "One common prompt across all domains"
            frame.to_csv(path, index=False)
        data, _ = self.run_prepare()
        self.assertEqual(data.group_id.nunique(), 1)
        self.assertEqual(data.pair_id.nunique(), 24)

    def test_symmetric_short_filter_and_insufficient_rows_raise(self):
        path = self.raw / "Essay_LLMs.csv"
        frame = pd.read_csv(path)
        frame["human"] = "Too short."
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "unique eligible humans"):
            self.run_prepare()

    def test_generator_cannot_be_silently_dropped(self):
        path = self.raw / "WP_LLMs.csv"
        frame = pd.read_csv(path).drop(columns="Claude")
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            self.run_prepare()

    def test_missing_domain_is_actionable(self):
        (self.raw / "Reuters_LLMs.csv").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Reuters_LLMs.csv"):
            self.run_prepare()

    def test_duplicate_human_removal(self):
        path = self.raw / "Reuters_LLMs.csv"
        frame = pd.read_csv(path)
        frame.loc[1, "human"] = frame.loc[0, "human"].upper()
        frame.to_csv(path, index=False)
        data, manifest = self.run_prepare()
        self.assertEqual(manifest["diagnostics"]["Reuters"]["removed"]["duplicate_human"], 1)
        self.assertEqual(data[data.label == 0].text.map(canonical).nunique(), 24)

    def test_spacing_repair_preserves_measurable_style(self):
        self.assertEqual(process_spaces("Hello , world .<newline> New sentence !"), "Hello, world.\nNew sentence!")
        self.assertIn("delve", process_spaces("I delve into this topic."))

    def test_trims_only_incomplete_trailing_fragment(self):
        self.assertEqual(trim_incomplete_tail("A complete sentence. Trailing fragment without end"), "A complete sentence.")
        self.assertEqual(trim_incomplete_tail("A complete sentence. Another complete sentence!"),
                         "A complete sentence. Another complete sentence!")

    def test_no_boundary_is_empty_and_decimals_not_boundary(self):
        self.assertEqual(trim_incomplete_tail("Only an unfinished fragment"), "")
        self.assertEqual(trim_incomplete_tail("The value is 3.14 and rising"), "")
        self.assertEqual(trim_incomplete_tail("Dr. Smith enters the U.S. embassy without"), "")

    def test_quoted_ending_is_preserved(self):
        self.assertEqual(trim_incomplete_tail('He said, "It is done." Next unfinished'), 'He said, "It is done."')
        self.assertEqual(trim_incomplete_tail('He said, “It is done.”'), 'He said, “It is done.”')

    def test_minimum_length_applies_after_tail_removal(self):
        path = self.raw / "Essay_LLMs.csv"
        frame = pd.read_csv(path)
        frame["human"] = [f"A short complete sentence {i}. " + "unfinished " * 60 for i in range(len(frame))]
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "unique eligible humans"):
            self.run_prepare()
        # Explicit sensitivity mode preserves the fragment; no silent fallback.
        data, manifest = self.run_prepare("sensitivity", clean_sentence_ends=False)
        self.assertEqual(len(data), 48)
        self.assertFalse(manifest["clean_sentence_ends"])

    def test_trim_counters_are_saved(self):
        path = self.raw / "Essay_LLMs.csv"
        frame = pd.read_csv(path)
        frame["human"] = frame["human"] + " Unfinished fragment"
        frame.to_csv(path, index=False)
        data, manifest = self.run_prepare()
        self.assertTrue(manifest["clean_sentence_ends"])
        self.assertEqual(manifest["cleanup_diagnostics_all_source_rows"]["Essay"]["human"]["n_tail_trimmed"], 12)
        self.assertEqual(manifest["cleanup_diagnostics_selected"]["Essay"]["human"]["n_tail_trimmed"], 8)
        self.assertFalse(data[data.domain == "Essay"].text.str.endswith("fragment").any())


if __name__ == "__main__":
    unittest.main()

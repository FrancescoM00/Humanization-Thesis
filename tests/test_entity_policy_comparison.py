"""Checks that unrelated changes cannot pass the policy comparison."""
import pandas as pd
import pytest
import hashlib
import json
from types import SimpleNamespace

from humanization.evaluation.workflows.compare_entity_policy import ROOT, freeze_unchanged_alignment, unchanged_pairs, validate_scores


def test_only_entity_rows_and_their_local_controls_may_change():
    baseline = {
        "fixed": {"perturbation": "negation", "modified": "It is not true."},
        "entity": {"dataset": "controlled", "perturbation": "entity_substitution", "modified": "In Japan."},
        "control": {"dataset": "local_control", "perturbation": "identity_for_entity_substitution", "modified": "In Canada."},
        "fixture": {"dataset": "manual_sanity", "perturbation": "entity_substitution", "modified": "Paris."},
    }
    clean = {
        "fixed": baseline["fixed"].copy(),
        "new-entity": {"dataset": "controlled", "perturbation": "entity_substitution", "modified": "Alex spoke."},
        "new-control": {"dataset": "local_control", "perturbation": "identity_for_entity_substitution", "modified": "Sam spoke."},
        "fixture": baseline["fixture"].copy(),
    }
    assert unchanged_pairs(baseline, clean) == ["fixed", "fixture"]
    clean["fixed"]["modified"] = "It is true."
    with pytest.raises(ValueError, match="Non-entity inputs changed"):
        unchanged_pairs(baseline, clean)


def test_missing_opai_rows_are_not_silently_ignored():
    row = {"perturbation": "compress", "dataset": "opai"}
    with pytest.raises(ValueError, match="Non-entity IDs differ"):
        unchanged_pairs({"opai": row}, {})


def test_runtime_counters_may_change_but_semantic_scores_may_not():
    a = pd.DataFrame({"id": ["a", "b"], "nli_forward_min": [.1, .9], "bert_idf": [False, False],
                      "local_bert_f1": [float("nan"), .95], "nli_new_pairs_scored": [50, 25]})
    b = a.copy()
    b["nli_new_pairs_scored"] = 0
    b.loc[0, "nli_forward_min"] += 1e-8
    checks = validate_scores(a, b, ["a", "b"], 1e-6, 1e-6)
    assert set(checks.metric) == {"nli_forward_min", "local_bert_f1", "bert_idf"}
    b.loc[1, "nli_forward_min"] = .6
    with pytest.raises(ValueError, match="Unrelated metric changed"):
        validate_scores(a, b, ["a", "b"], 1e-6, 1e-6)


def freeze_fixture(tmp_path):
    paths = {name: tmp_path/name for name in ["baseline_data", "baseline_output", "clean_data", "clean_output"]}
    for path in paths.values():
        path.mkdir()
    args = SimpleNamespace(**paths, atol=1e-6, rtol=1e-6)
    code = hashlib.sha256((ROOT/"src/humanization/semantics/metrics.py").read_bytes()).hexdigest()
    old_run = {"model_directory": str(tmp_path/"model"), "metrics_code_sha256": "unarchived-embedding-module"}
    new_run = dict(old_run, metrics_code_sha256=code)
    (args.baseline_output/"embedding_run.json").write_text(json.dumps(old_run))
    (args.clean_output/"embedding_run.json").write_text(json.dumps(new_run))
    (args.baseline_output/"nli_run.json").write_text(json.dumps({"metrics_code_sha256": code}))
    for path in [args.baseline_output, args.clean_output]:
        pd.DataFrame({"id": ["fixed"], "embedding_cosine": [1.0]}).to_csv(path/"embedding.csv", index=False)
    fixed = {"id": "fixed", "dataset": "controlled", "perturbation": "identity",
             "original": "A. B.", "modified": "A. B."}
    old = {"fixed": fixed, "old_entity": {"dataset": "controlled", "perturbation": "entity_substitution"}}
    new = {"fixed": fixed.copy(), "new_entity": {"dataset": "controlled", "perturbation": "entity_substitution"}}
    sha = hashlib.sha256(fixed["original"].encode()).hexdigest()
    historical = {"original_sha256": sha, "modified_sha256": sha,
                  "original_chunks": ["A.", "B."], "modified_chunks": ["A.", "B."],
                  "chunk_token_budget": 160, "retrieval_top_k": 1,
                  "forward_candidates": [[0], [1]], "reverse_candidates": [[0], [1]]}
    recomputed = dict(historical, forward_candidates=[[1], [1]])
    (args.baseline_data/"alignment.json").write_text(json.dumps({"fixed": historical, "old_entity": {"historical": True}}))
    (args.clean_data/"alignment.json").write_text(json.dumps({"fixed": recomputed, "new_entity": {"fresh": True}}))
    return args, old, new, historical, code


def test_freeze_rejects_unknown_code_and_existing_nli_plan_without_mutation(tmp_path):
    args, old, new, _, code = freeze_fixture(tmp_path)
    before = (args.clean_data/"alignment.json").read_text()
    (args.baseline_output/"nli_run.json").write_text(json.dumps({"metrics_code_sha256": "different"}))
    with pytest.raises(ValueError, match="historical NLI code does not match"):
        freeze_unchanged_alignment(args, old, new, ["fixed"])
    assert (args.clean_data/"alignment.json").read_text() == before
    assert not (args.clean_data/"alignment_recomputed.json").exists()
    (args.baseline_output/"nli_run.json").write_text(json.dumps({"metrics_code_sha256": code}))
    (args.clean_data/"nli_jobs.jsonl").write_text("")
    with pytest.raises(ValueError, match="precede NLI planning"):
        freeze_unchanged_alignment(args, old, new, ["fixed"])
    assert (args.clean_data/"alignment.json").read_text() == before


def test_freeze_preserves_fresh_entity_retrieval_and_discloses_hash_fallback(tmp_path):
    args, old, new, historical, _ = freeze_fixture(tmp_path)
    before = (args.clean_data/"alignment.json").read_text()
    manifest = freeze_unchanged_alignment(args, old, new, ["fixed"])
    after = json.loads((args.clean_data/"alignment.json").read_text())
    assert after["fixed"] == historical
    assert after["new_entity"] == {"fresh": True}
    assert (args.clean_data/"alignment_recomputed.json").read_text() == before
    assert not manifest["historical_embedding_code_hash_matches"]
    assert manifest["candidate_index_arrays_differ_before_freeze"] == 1
    assert manifest["candidate_text_sets_differ_before_freeze"] == 1
    assert not manifest["nli_scores_used_for_evidence_selection"]

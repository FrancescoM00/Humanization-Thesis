"""Calibration split isolation and deterministic context/aggregator selection."""
import csv
import hashlib
import json
import math

import pytest

from scripts.calibrate_contextual_nli import (
    AGGREGATORS, MODES, analyze_scores, combine_directions, csv_text,
    frozen_policy, prepare_panel, select_candidate,
)


def fixture_inputs(tmp_path):
    def row(key, source, original, modified, dataset, split="calibration", positive=None):
        return dict(id=key, source_group_id=source, source_id=source,
            original=original, modified=modified, dataset=dataset,
            evaluation_split=split, status="applied", perturbation="matched_negation" if positive else "opai_polish",
            metadata={} if positive is None else {"matched_positive_id": positive})
    rows = [row("p1", "cal1", "Original one.", "Rewrite one.", "opai_positive"),
            row("p2", "cal1", "Original one.", "Rewrite two.", "opai_positive"),
            row("n1", "cal1", "Original one.", "Not one.", "opai_matched_negative", positive="p1"),
            row("n2", "cal1", "Original one.", "Not two.", "opai_matched_negative", positive="p2"),
            row("unused", "cal2", "Unmatched.", "Unused.", "opai_positive"),
            row("testp", "test1", "Test original.", "Test rewrite.", "opai_positive", split="test"),
            row("testn", "test1", "Test original.", "Not test.", "opai_matched_negative", split="test", positive="testp")]
    align = {r["id"]: dict(original_sha256=hashlib.sha256(r["original"].encode()).hexdigest(),
                          modified_sha256=hashlib.sha256(r["modified"].encode()).hexdigest(),
                          original_chunks=[r["original"]], modified_chunks=[r["modified"]]) for r in rows}
    pairs_path, alignment_path = tmp_path/"pairs.jsonl", tmp_path/"alignment.json"
    pairs_path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    alignment_path.write_text(json.dumps(align))
    return pairs_path, alignment_path, rows


def score_fixture(prepared, path, *, overrides=None):
    labels = list(csv.DictReader((prepared/"labels.csv").open()))
    result = []
    for label in labels:
        value = .99 if label["kind"] == "identity" else .9 if label["preserved"] == "1" else .3
        for mode in MODES:
            for direction in ("forward", "reverse"):
                row = dict(id=label["id"], direction=direction, mode=mode, complete="True",
                    mean=value, q10=value, min=value, low_support_fraction=1-value)
                if overrides:
                    overrides(row, label)
                result.append(row)
    path.write_text(csv_text(result, list(result[0])))


def test_prepare_uses_only_linked_calibration_pairs_and_deduplicates_original_identity(tmp_path):
    pairs, alignment, _ = fixture_inputs(tmp_path)
    out = tmp_path/"prepared"
    manifest = prepare_panel(pairs, alignment, out)
    assert manifest["n_positive"] == manifest["n_negative"] == 2
    assert manifest["n_identity"] == 1 and manifest["n_source_groups"] == 1
    prepared = [json.loads(line) for line in (out/"pairs.jsonl").read_text().splitlines()]
    assert {r["evaluation_split"] for r in prepared} == {"calibration"}
    assert {r["source_group_id"] for r in prepared} == {"cal1"}
    assert not {"testp", "testn", "unused"} & {r["id"] for r in prepared}
    identity = next(r for r in prepared if r["perturbation"] == "identity")
    assert identity["original"] == identity["modified"] == "Original one."
    again = prepare_panel(pairs, alignment, out)
    assert again == manifest


def test_prepare_rejects_test_positive_link_and_source_leakage(tmp_path):
    pairs, alignment, rows = fixture_inputs(tmp_path)
    rows[2]["metadata"]["matched_positive_id"] = "testp"
    pairs.write_text("".join(json.dumps(r)+"\n" for r in rows))
    with pytest.raises(ValueError, match="positive.*calibration"):
        prepare_panel(pairs, alignment, tmp_path/"one")
    rows[2]["metadata"]["matched_positive_id"] = "p1"
    rows[-1]["source_group_id"] = "cal1"
    pairs.write_text("".join(json.dumps(r)+"\n" for r in rows))
    with pytest.raises(ValueError, match="source groups overlap"):
        prepare_panel(pairs, alignment, tmp_path/"two")


def test_bidirectional_score_and_missing_coverage_are_not_averaged():
    a, b = dict(complete=True, mean=.9, low_support_fraction=.1), dict(complete=True, mean=.4, low_support_fraction=.6)
    assert combine_directions(a, b, "mean") == .4
    assert combine_directions(a, b, "low_support_fraction") == .4
    b["complete"] = False
    assert math.isnan(combine_directions(a, b, "mean"))
    b.update(complete=True, coverage_fraction=.5)
    with pytest.raises(ValueError, match="Coverage fraction"):
        combine_directions(a, b, "mean")


def test_analyze_ties_and_identity_gate_before_primary_selection(tmp_path):
    pairs, alignment, _ = fixture_inputs(tmp_path)
    prepared = tmp_path/"prepared"
    prepare_panel(pairs, alignment, prepared)
    scores = tmp_path/"scores.csv"
    score_fixture(prepared, scores)
    result = analyze_scores(prepared, scores, tmp_path/"result", bootstrap=0)
    assert result["selected"]["mode"] == "single_all"
    assert result["selected"]["aggregator"] == "mean"
    assert result["selected"]["calibration_auc"] == 1
    assert result["test_used_for_selection"] is False
    assert result["selected"]["calibration_auc_ci95_low"] is None

    def low_identity(row, label):
        if label["kind"] == "identity":
            row.update(mean=.7, q10=.7, min=.7, low_support_fraction=.3)
    score_fixture(prepared, scores, overrides=low_identity)
    with pytest.raises(ValueError, match="Analysis inputs changed"):
        analyze_scores(prepared, scores, tmp_path/"result", bootstrap=0)
    result = analyze_scores(prepared, scores, tmp_path/"different", bootstrap=0)
    assert result["status"] == "no_eligible_candidate" and result["selected"] is None


def test_incomplete_best_mode_is_disqualified_and_more_accurate_valid_mode_wins(tmp_path):
    pairs, alignment, _ = fixture_inputs(tmp_path)
    prepared, scores = tmp_path/"prepared", tmp_path/"scores.csv"
    prepare_panel(pairs, alignment, prepared)
    def change(row, label):
        if row["mode"] == "document":
            row["complete"] = "False"
        elif row["mode"] == "single_all" and label["kind"] != "identity":
            value = .2 if label["preserved"] == "1" else .8
            row.update(mean=value, q10=value, min=value, low_support_fraction=1-value)
    score_fixture(prepared, scores, overrides=change)
    result = analyze_scores(prepared, scores, tmp_path/"result", bootstrap=10)
    assert result["selected"]["mode"] == "window2"
    assert result["selected"]["aggregator"] == "mean"
    summary = list(csv.DictReader((tmp_path/"result/candidate_summary.csv").open()))
    doc = next(r for r in summary if r["mode"] == "document" and r["aggregator"] == "mean")
    assert doc["eligible_primary"] == "False" and doc["calibration_auc"] == "nan"
    assert "incomplete_labelled_coverage" in doc["exclusion_reasons"]


def test_analyze_rejects_unfrozen_policy_and_incomplete_score_grid(tmp_path):
    pairs, alignment, _ = fixture_inputs(tmp_path)
    prepared, scores = tmp_path/"prepared", tmp_path/"scores.csv"
    prepare_panel(pairs, alignment, prepared)
    score_fixture(prepared, scores)
    body = scores.read_text().splitlines()
    scores.write_text("\n".join(body[:-1])+"\n")
    with pytest.raises(ValueError, match="both directions.*exactly once"):
        analyze_scores(prepared, scores, tmp_path/"result")
    policy = json.loads((prepared/"policy.json").read_text())
    policy["primary_requirements"]["identity_mean_min"] = .1
    (prepared/"policy.json").write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="changed after freezing"):
        analyze_scores(prepared, scores, tmp_path/"result")

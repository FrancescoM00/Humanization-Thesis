import csv
import io
import json
from pathlib import Path
import zipfile

import pytest

from scripts.prepare_semantic_audit_v2 import (
    FORM_COLUMNS, candidates_from_inputs, prepare_audit, read_pairs,
    select_candidates,
)


def make_frame(per_cell=6):
    pairs, scores = [], {}
    for domain in ['Essay', 'Reuters', 'WritingPrompts']:
        for generator in [f'g{i}' for i in range(7)]:
            for index in range(per_cell):
                source = f'{domain}:{generator}:{index}'
                for system in ['baseline_3b', 'dipper_configured']:
                    pair_id = f'{source}:{system}'
                    pair = dict(id=pair_id, source_id=source, source_group_id=source,
                                domain=domain, generator=generator, humanizer=system,
                                original=f'Original {source}.', modified=f'Rewrite {pair_id}.',
                                status='ok')
                    pairs.append(pair)
                    # Includes contradiction-like, low-NLI-only and other strata.
                    scores[pair_id] = dict(id=pair_id, embedding_cosine=.95 if index % 3 == 0 else .7,
                                           nli_worst_direction_min=.02 if index % 3 != 2 else .8)
    candidates, _ = candidates_from_inputs(pairs, scores)
    return pairs, scores, candidates


def source_set(rows, arm):
    return {r['source_group_id'] for r in rows if r['selection_arm'] == arm}


def test_study_uses_probability_reference_and_disjoint_paired_diagnostics():
    _, _, candidates = make_frame()
    selected, manifest = select_candidates(candidates)
    reference, diagnostic = source_set(selected, 'reference'), source_set(selected, 'diagnostic')
    assert len(reference) == 21
    assert len(diagnostic) == 50
    assert not reference & diagnostic
    assert manifest['reference_pairs'] == 42
    assert manifest['selected_diagnostic_pairs'] == 100
    assert manifest['diagnostic_target_met']
    assert len(selected) == 142
    for source in reference | diagnostic:
        members = [r for r in selected if r['source_group_id'] == source]
        assert {r['humanizer'] for r in members} == {'baseline_3b', 'dipper_configured'}
        if source in reference:
            assert {r['reference_source_inclusion_probability'] for r in members} == {1/6}
            assert {r['reference_source_weight'] for r in members} == {6.0}
    assert manifest['diagnostic_pair_strata']['high_cosine_low_nli'] > 0
    assert manifest['diagnostic_pair_strata']['low_nli'] > 0


def test_reference_ignores_scores_and_input_order():
    _, _, candidates = make_frame()
    original, _ = select_candidates(candidates)
    changed = [dict(r, nli_min=1-r['nli_min'], embedding_cosine=-r['embedding_cosine'])
               for r in reversed(candidates)]
    other, _ = select_candidates(changed)
    assert source_set(original, 'reference') == source_set(other, 'reference')
    repeated, _ = select_candidates(list(reversed(candidates)))
    assert {r['id'] for r in original} == {r['id'] for r in repeated}


def write_inputs(tmp_path, per_cell=2):
    pairs, scores, _ = make_frame(per_cell)
    pair_path, score_path = tmp_path/'pairs.jsonl', tmp_path/'scores.csv'
    pair_path.write_text(''.join(json.dumps(r)+'\n' for r in pairs))
    with score_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(scores.values()))))
        writer.writeheader()
        writer.writerows(scores.values())
    return pair_path, score_path


def test_pilot_capacity_is_explicit_and_forms_are_blind_independent_copies(tmp_path):
    pair_path, score_path = write_inputs(tmp_path)
    output = tmp_path/'out'
    manifest = prepare_audit([pair_path], [score_path], output)
    assert manifest['available_pairs'] == 84
    assert manifest['reference_pairs'] == 42
    assert manifest['available_diagnostic_pairs'] == 42
    assert manifest['selected_diagnostic_pairs'] == 42
    assert manifest['total_selected_unique_sources'] == 42
    assert manifest['selected_is_full_census']
    assert not manifest['diagnostic_target_met']
    assert not manifest['labels_present']
    forms = []
    for annotator in ['A', 'B']:
        with (output/f'semantic_audit_annotator_{annotator}.csv').open() as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == FORM_COLUMNS
            form = list(reader)
        assert len(form) == 84
        assert all(not r['semantic_preserved_yes_partial_no'] and not r['notes'] for r in form)
        assert all(set(r) == set(FORM_COLUMNS) for r in form)
        forms.append(form)
    assert {r['review_id'] for r in forms[0]} == {r['review_id'] for r in forms[1]}
    assert [r['review_id'] for r in forms[0]] != [r['review_id'] for r in forms[1]]
    with zipfile.ZipFile(output/'annotation_forms.zip') as archive:
        assert set(archive.namelist()) == {
            'semantic_audit_annotator_A.csv', 'semantic_audit_annotator_B.csv', 'ISTRUZIONI.md'}
    # Identical inputs can resume; changing the sampling decision cannot overwrite.
    prepare_audit([pair_path], [score_path], output)
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        prepare_audit([pair_path], [score_path], output, seed=43)


def test_strict_capacity_fails_before_writing_and_sources_never_split(tmp_path):
    pair_path, score_path = write_inputs(tmp_path)
    output = tmp_path/'out'
    with pytest.raises(ValueError, match='Diagnostic target not reached'):
        prepare_audit([pair_path], [score_path], output, require_full_diagnostic=True)
    assert not output.exists()
    _, _, candidates = make_frame(2)
    selected, info = select_candidates(candidates, diagnostic_size=5)
    assert info['selected_diagnostic_pairs'] == 4
    assert not info['diagnostic_target_met']
    assert len(source_set(selected, 'diagnostic')) == 2


def test_missing_nonfinite_or_conflicting_scores_cannot_silently_bias_frame():
    pairs, scores, _ = make_frame(2)
    with pytest.raises(ValueError, match='Missing scores'):
        candidates_from_inputs(pairs, {})
    scores[pairs[0]['id']]['embedding_cosine'] = float('nan')
    with pytest.raises(ValueError, match='Invalid embedding_cosine'):
        candidates_from_inputs(pairs, scores)
    pairs, scores, _ = make_frame(2)
    duplicate = dict(pairs[0], modified='A different output.')
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        candidates_from_inputs([*pairs, duplicate], scores)


def test_zip_prefers_semantic_actual_input_over_stale_rewrites(tmp_path):
    archive = tmp_path/'legacy.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('rewriting/rewrites.jsonl', '{"id":"stale"}\n')
        z.writestr('rewriting_cuda0/rewrites.jsonl', '{"id":"complete"}\n')
        z.writestr('semantic/pairs.jsonl', '{"id":"scored"}\n')
    assert read_pairs(archive) == [{'id': 'scored'}]

import csv
import hashlib
import json
import math

import pytest
import spacy

from humanization.detection.pipeline import validate_inputs
from humanization.rewriting.controls import (
    PROTOCOL, generate_null_controls, lexical_count, minimal_lexical_noise,
    prefix_at_word_count, prepare, summarize_null_controls,
)


@pytest.fixture
def nlp():
    pipeline = spacy.blank('en')
    pipeline.add_pipe('sentencizer')
    ruler = pipeline.add_pipe('entity_ruler')
    ruler.add_patterns([{'label': 'ORG', 'pattern': 'large corporation'},
                        {'label': 'ORG', 'pattern': 'Acme'}])
    return pipeline


def fixture_data():
    cohort = [dict(id='cal', source_group_id='cal-g', split='calibration',
                   domain='Essay', generator='g', human='Calibration human.')]
    texts = ['They use several tools. They also help patients.',
             'They use tools.', 'They also use several tools.']
    rewrites = []
    for i, text in enumerate(texts):
        source = dict(id=f't{i}', source_group_id=f'g{i}', split='test',
                      domain='Essay', generator='g', human=f'Human {i}.', machine=text)
        cohort.append(source)
        rewrites.append(dict(id=f'r{i}', source_id=source['id'], source_group_id=source['source_group_id'],
                             domain='Essay', generator='g', humanizer='writer', original=text,
                             modified=['They use tools.', 'They use many more tools than before.', ''][i],
                             status='ok' if i < 2 else 'generation_error'))
    return cohort, rewrites


def test_exact_word_prefix_retains_contraction_suffix_and_declares_sentence_cut(nlp):
    text = "I can't use 50 tools. Later they help."
    prefix, metadata = prefix_at_word_count(text, 2, nlp)
    assert prefix == "I can't"
    assert lexical_count(nlp.make_doc(prefix)) == 2
    assert metadata['hard_cut'] is True
    assert metadata['cut_at_sentence_boundary'] is False
    # Si può conservare un confine corretto abbinando esattamente cinque parole
    # lessicali: I/ca/use/tools (50 non è una parola lessicale alfabetica).
    prefix, metadata = prefix_at_word_count(text, 4, nlp)
    assert prefix == "I can't use 50 tools."
    assert lexical_count(nlp.make_doc(prefix)) == 4
    assert metadata['hard_cut'] is False
    assert metadata['cut_at_sentence_boundary'] is True
    original, metadata = prefix_at_word_count(text, 100, nlp)
    assert original == text
    assert metadata['length_match_eligible'] is False
    assert metadata['length_match_status'] == 'target_longer_than_original'


def test_minimal_noise_preserves_count_protected_entities_numbers_modals_and_negation(nlp):
    original = 'Acme may not use 50 tools; the large corporation also helps.'
    modified, metadata = minimal_lexical_noise(original, nlp, seed=11)
    assert modified != original
    assert 'Acme may not' in modified and '50' in modified and 'large corporation' in modified
    assert lexical_count(nlp.make_doc(modified)) == lexical_count(nlp.make_doc(original))
    assert len(metadata['noise_edits']) == 1
    assert metadata['noise_is_semantic_gold'] is False
    assert minimal_lexical_noise(original, nlp, seed=11) == (modified, metadata)
    untouched, metadata = minimal_lexical_noise('Acme may not visit the large corporation in 2020.', nlp)
    assert untouched == 'Acme may not visit the large corporation in 2020.'
    assert metadata['noise_eligible'] is False
    # L'abbinamento in minuscolo non modifica accidentalmente nomi propri o sottostringhe.
    untouched, _ = minimal_lexical_noise('Use Useless Large.', nlp)
    assert untouched == 'Use Useless Large.'


def test_prefix_does_not_attach_a_dangling_hyphen_that_changes_token_count(nlp):
    prefix, metadata = prefix_at_word_count('A diet high in plant-based foods.', 5, nlp)
    assert prefix == 'A diet high in plant'
    assert lexical_count(nlp.make_doc(prefix)) == 5
    assert metadata['length_match_eligible'] and metadata['hard_cut']


def test_generation_keeps_longer_target_ineligible_and_never_imputes_failed_reference(nlp):
    cohort, references = fixture_data()
    controls = generate_null_controls(cohort, references, nlp, include_rewrites=True)
    validate_inputs(cohort, controls)
    lookup = {(r['humanizer'], r['source_id']): r for r in controls}
    matched = lookup['null_length__writer', 't0']
    assert matched['modified'] == 'They use several'
    assert matched['metadata']['target_n_words'] == 3
    assert matched['metadata']['modified_n_words'] == 3
    assert matched['metadata']['length_match_eligible'] is True
    longer = lookup['null_length__writer', 't1']
    assert longer['modified'] == cohort[2]['machine']
    assert longer['status'] == 'ok'
    assert longer['metadata']['comparison_eligible'] is False
    assert longer['metadata']['length_match_status'] == 'target_longer_than_original'
    unavailable = lookup['null_length__writer', 't2']
    assert unavailable['status'] == 'reference_unavailable'
    assert unavailable['metadata']['target_n_words'] is None
    assert unavailable['modified'] == ''
    # Le condizioni sono complete per ogni fonte di test, anche quando il controllo
    # di lunghezza non è applicabile; ogni identità resta distinta dal riscrittore.
    assert {r['humanizer'] for r in controls} == {
        'writer', 'null_identity', 'null_lexical_noise',
        'null_length__writer', 'null_length_noise__writer'}
    assert len(controls) == 15
    assert all(lookup['null_identity', f't{i}']['modified'] == cohort[i+1]['machine'] for i in range(3))


def test_reference_order_duplicates_and_frozen_outputs_do_not_change_controls(nlp, tmp_path):
    cohort, references = fixture_data()
    left = generate_null_controls(cohort, references, nlp)
    right = generate_null_controls(cohort, list(reversed(references)) + [dict(references[0])], nlp)
    assert left == right
    conflicting = dict(references[0], modified='A different rewrite.')
    with pytest.raises(ValueError, match='Conflicting reference'):
        generate_null_controls(cohort, references + [conflicting], nlp)
    cohort_path, reference_path = tmp_path/'cohort.jsonl', tmp_path/'references.jsonl'
    cohort_path.write_text(''.join(json.dumps(r)+'\n' for r in cohort))
    reference_path.write_text(''.join(json.dumps(r)+'\n' for r in references))
    out = tmp_path/'controls'
    manifest = prepare(cohort_path, [reference_path], out, nlp=nlp)
    assert manifest['n_rows'] == 12
    assert prepare(cohort_path, [reference_path], out, nlp=nlp) == manifest
    original = (out/'rewrites.jsonl').read_bytes()
    with pytest.raises(ValueError, match='Refusing to replace'):
        prepare(cohort_path, [reference_path], out, nlp=nlp, noise_max_edits=2)
    assert (out/'rewrites.jsonl').read_bytes() == original
    with (out/'eligibility.csv').open() as f:
        eligibility = list(csv.DictReader(f))
    assert len(eligibility) == 12


def test_control_comparison_uses_eligible_common_baseline_not_noop_or_failed_rows(nlp):
    cohort, references = fixture_data()
    controls = generate_null_controls(cohort, references, nlp, include_rewrites=True)
    scores = []
    for source in cohort[1:]:
        scores.append(dict(detector='d', source_id=source['id'], role='machine',
                           humanizer='', score=.9, text_sha256=hashlib.sha256(source['machine'].encode()).hexdigest()))
    for row in controls:
        if row['status'] != 'ok':
            continue
        # La riscrittura breve valida evade; il prefisso originale della stessa lunghezza
        # non evade. L'output t1, più lungo, non è ammissibile al confronto di lunghezza.
        score = .1 if row['humanizer'] == 'writer' else .9
        scores.append(dict(detector='d', source_id=row['source_id'], role='rewrite',
                           humanizer=row['humanizer'], score=score,
                           text_sha256=hashlib.sha256(row['modified'].encode()).hexdigest()))
    summary = summarize_null_controls(cohort, references, controls, scores, {'d': {'threshold': .5}})
    length = next(r for r in summary if r['condition'] == 'null_length__writer')
    assert length['total_sources'] == 3
    assert length['eligible_sources'] == 1
    assert length['ineligible_sources'] == 2
    assert length['eligible_original_detected'] == 1
    assert length['control_asr'] == 0
    assert length['reference_asr'] == 1
    assert length['delta_asr_reference_minus_control'] == 1
    no_reference = next(r for r in summary if r['condition'] == 'null_identity')
    assert math.isnan(no_reference['reference_asr'])
    with pytest.raises(ValueError, match='Missing detector score'):
        summarize_null_controls(cohort, references, controls,
            [r for r in scores if not (r['humanizer']=='writer' and r['source_id']=='t0')],
            {'d': {'threshold': .5}})


def test_raw_lower_direction_and_missing_baseline_are_explicit(nlp):
    cohort, references = fixture_data()
    controls = generate_null_controls(cohort, references, nlp, include_rewrites=True)
    scores=[]
    for source in cohort[1:]:
        scores.append(dict(detector='ratio',source_id=source['id'],role='machine',humanizer='',score=.4))
    for row in controls:
        if row['status']=='ok':
            scores.append(dict(detector='ratio',source_id=row['source_id'],role='rewrite',
                               humanizer=row['humanizer'],score=.7 if row['humanizer']=='writer' else .4))
    summary=summarize_null_controls(cohort,references,controls,scores,{'ratio':{'threshold':.5}},lower_is_ai={'ratio'})
    length=next(r for r in summary if r['condition']=='null_length__writer')
    assert length['ai_direction']=='lower' and length['reference_asr']==1
    with pytest.raises(ValueError,match='Missing detector score'):
        summarize_null_controls(cohort,references,controls,
            [r for r in scores if r['role']!='machine'],{'ratio':{'threshold':.5}},lower_is_ai={'ratio'})

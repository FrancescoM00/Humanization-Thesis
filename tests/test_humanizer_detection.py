from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from humanizers.detection import (calibrate_threshold, summarize_detection,
                                  text_records, validate_ai_label, validate_inputs)


def fixture_data():
    cohort = [{'id': 'cal', 'source_group_id': 'cal-g', 'split': 'calibration',
               'domain': 'Essay', 'generator': 'g', 'human': 'A human calibration sample.'}]
    rewrites = []
    for i in range(3):
        cohort.append({'id': f't{i}', 'source_group_id': f'g{i}', 'split': 'test',
                       'domain': 'Essay', 'generator': 'g', 'human': f'Human sample {i}.',
                       'machine': f'Machine sample {i}.'})
        rewrites.append({'id': f'r{i}', 'source_id': f't{i}', 'source_group_id': f'g{i}',
                         'humanizer': 'rewrite', 'original': f'Machine sample {i}.',
                         'modified': 'Rewritten sample.' if i != 1 else '',
                         'status': 'ok' if i != 1 else 'generation_error'})
    rows = text_records(cohort, rewrites)
    # Original positives t0,t1; t0 evades, t1 generation fails. t2 was already
    # undetected and must never enlarge the ASR denominator or numerator.
    score = {('cal', 'calibration_human'): .2,
             ('t0', 'human'): .1, ('t1', 'human'): .1, ('t2', 'human'): .3,
             ('t0', 'machine'): .9, ('t1', 'machine'): .8, ('t2', 'machine'): .1,
             ('t0', 'rewrite'): .1, ('t2', 'rewrite'): .05}
    scores = pd.DataFrame([{key: value for key, value in row.items() if key != 'text'} |
                           {'detector': 'd1', 'score': score[row['source_id'], row['role']],
                            'text_sha256': hashlib.sha256(row['text'].encode()).hexdigest()}
                           for row in rows])
    return cohort, rewrites, scores


def test_label_direction_is_explicit_and_checked_against_config():
    model = SimpleNamespace(num_labels=2, id2label={0: 'Fake', 1: 'Real'})
    assert validate_ai_label(model, {'ai_label_index': 0, 'ai_label': 'Fake'})[0] == 0
    with pytest.raises(ValueError, match='mismatch'):
        validate_ai_label(model, {'ai_label_index': 1, 'ai_label': 'Fake'})
    with pytest.raises(ValueError, match='explicit'):
        validate_ai_label(model, {})


def test_threshold_ties_and_small_calibration_respect_empirical_fpr():
    result = calibrate_threshold([.1] * 18 + [.8, .8], .05)
    assert result['threshold'] == .8
    assert result['calibration_fpr'] == 0
    values = np.arange(20) / 20
    result = calibrate_threshold(values, .05)
    assert sum(values > result['threshold']) == 1
    assert result['calibration_fpr'] == .05
    assert calibrate_threshold([.1, .2, .3], .05)['threshold'] == .3


def test_conditional_asr_counts_failed_generation_and_excludes_preexisting_evasion():
    cohort, rewrites, scores = fixture_data()
    thresholds, summary, macro = summarize_detection(cohort, rewrites, scores,
        {'detectors': [{'name': 'd1'}], 'bootstrap': 30})
    result = summary[summary.domain == 'overall'].iloc[0]
    assert thresholds['d1']['threshold'] == .2
    assert result.baseline_detected == 2
    assert result.attack_successes == 1
    assert result.asr == .5
    assert result.generations_failed == 1
    assert result.baseline_detected_generation_failures == 1
    assert result.complete_pairs == 2
    assert result.human_test_fpr == pytest.approx(1 / 3)
    assert macro.loc[macro.domain == 'overall', 'macro_mean_asr'].iloc[0] == .5
    # Arbitrary machine test scores cannot alter the human-only threshold.
    changed = scores.copy()
    changed.loc[changed.role == 'machine', 'score'] = .01
    new_thresholds, zero, _ = summarize_detection(cohort, rewrites, changed,
        {'detectors': [{'name': 'd1'}], 'bootstrap': 0})
    assert new_thresholds == thresholds
    assert np.isnan(zero.loc[zero.domain == 'overall', 'asr'].iloc[0])


def test_missing_score_is_an_error_not_a_successful_evasion():
    cohort, rewrites, scores = fixture_data()
    with pytest.raises(ValueError, match='missing or duplicate'):
        summarize_detection(cohort, rewrites, scores.iloc[:-1], {'detectors': [{'name': 'd1'}]})
    changed = scores.copy()
    changed.loc[changed.role == 'human', 'source_group_id'] = 'wrong-group'
    with pytest.raises(ValueError, match='metadata'):
        summarize_detection(cohort, rewrites, changed, {'detectors': [{'name': 'd1'}]})


def test_source_leakage_and_incomplete_generation_are_rejected():
    cohort, rewrites, _ = fixture_data()
    with pytest.raises(ValueError, match='incomplete'):
        validate_inputs(cohort, rewrites[:-1])
    cohort[0]['source_group_id'] = 'g0'
    with pytest.raises(ValueError, match='disjoint'):
        validate_inputs(cohort, rewrites)


def test_changed_text_with_same_ids_cannot_reuse_stale_scores_or_relabel_manifest(tmp_path):
    from humanizers.detection import run

    cohort, rewrites, scores = fixture_data()
    config = {'detectors': [{'name': 'd1'}], 'bootstrap': 0}
    cohort_path, rewrites_path, config_path = [tmp_path / name for name in ('cohort.jsonl', 'rewrites.jsonl', 'config.json')]
    cohort_path.write_text(''.join(json.dumps(row) + '\n' for row in cohort))
    rewrites_path.write_text(''.join(json.dumps(row) + '\n' for row in rewrites))
    config_path.write_text(json.dumps(config))
    scores.to_csv(tmp_path / 'detector_scores.csv', index=False)
    manifest = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in
                [('cohort_sha256', cohort_path), ('rewrites_sha256', rewrites_path), ('config_sha256', config_path)]}
    (tmp_path / 'detection_run.json').write_text(json.dumps(manifest))
    rewrites[0]['modified'] = 'A different text under the same stable pair identifier.'
    with pytest.raises(ValueError, match='text hash'):
        summarize_detection(cohort, rewrites, scores, config)
    rewrites_path.write_text(''.join(json.dumps(row) + '\n' for row in rewrites))
    with pytest.raises(ValueError, match='inputs/config differ'):
        run(cohort_path, rewrites_path, config_path, tmp_path, analysis_only=True)
    assert json.loads((tmp_path / 'detection_run.json').read_text()) == manifest


def test_three_model_ensemble_votes_after_each_threshold_without_raw_score_averaging():
    cohort, rewrites, scores = fixture_data()
    frames = [scores.assign(detector=f'd{i}') for i in range(3)]
    thresholds, summary, macro = summarize_detection(cohort, rewrites, pd.concat(frames),
        {'detectors': [{'name': f'd{i}'} for i in range(3)], 'bootstrap': 0})
    assert 'majority_vote' in thresholds
    row = summary[(summary.detector == 'majority_vote') & (summary.domain == 'overall')].iloc[0]
    assert row.asr == .5
    assert macro.n_detectors.eq(3).all()


def test_detector_covers_all_chunks_and_reuses_identity_score(monkeypatch, tmp_path):
    import math
    import sys
    import torch
    from humanizers.detection import score_detector

    class Tokenizer:
        is_fast = False
        model_max_length = 8

        def encode(self, text, add_special_tokens=False):
            return list(range(len(text.split())))

        def num_special_tokens_to_add(self, pair=False):
            return 2

        def __call__(self, texts, **kwargs):
            lengths = [len(text.split()) + 2 for text in texts]
            ids = torch.zeros((len(texts), max(lengths)), dtype=torch.long)
            attention = ids.clone()
            for i, n in enumerate(lengths):
                attention[i, :n] = 1
            return {'input_ids': ids, 'attention_mask': attention}

    class Model:
        config = SimpleNamespace(num_labels=2, id2label={0: 'Fake', 1: 'Real'},
                                 max_position_embeddings=10, model_type='roberta', pad_token_id=1)
        calls = 0

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask):
            self.calls += 1
            # Six payload tokens -> p(Fake)=.9; three -> p(Fake)=.1.
            values = [math.log(9) if length.item() == 8 else -math.log(9)
                      for length in attention_mask.sum(axis=1)]
            return SimpleNamespace(logits=torch.tensor([[x, 0.] for x in values]))

    model = Model()
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
        AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=lambda *a, **k: model)))
    text = 'one two three four five six seven eight nine'
    frame, _ = score_detector([{'text': text}, {'text': text}],
        {'name': 'fake', 'model_id': 'fake', 'revision': 'test', 'ai_label_index': 0}, tmp_path)
    assert frame.tokens.tolist() == [9, 9]
    assert frame.n_chunks.tolist() == [2, 2]
    assert frame.score.tolist() == pytest.approx([(.9 * 6 + .1 * 3) / 9] * 2)
    assert model.calls == 1

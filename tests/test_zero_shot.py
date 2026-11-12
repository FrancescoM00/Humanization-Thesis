import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from humanizers.detection import text_records
from humanizers.zero_shot import (binoculars_components, calibrate_low_threshold,
    cached_weight_snapshot, select_writable_cache,
    ratio_from_components, summarize_zero_shot, token_windows)


def test_complete_pinned_snapshot_is_reused_but_partial_shards_are_not(tmp_path):
    snapshot = tmp_path/'models--owner--model'/'snapshots'/('a'*40)
    snapshot.mkdir(parents=True)
    (snapshot/'config.json').write_text('{}')
    (snapshot/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'one.safetensors', 'b': 'two.safetensors'}}))
    (snapshot/'one.safetensors').write_text('completed')
    (snapshot/'two.safetensors.incomplete').write_text('partial')
    assert cached_weight_snapshot('owner/model', 'a'*40, [tmp_path]) is None
    (snapshot/'two.safetensors').write_text('completed')
    assert cached_weight_snapshot('owner/model', 'a'*40, [tmp_path]) == snapshot
    assert cached_weight_snapshot('owner/model', 'b'*40, [tmp_path]) is None


def test_cache_selection_skips_a_read_only_mount_without_deleting_existing_data(tmp_path, monkeypatch):
    from pathlib import Path
    first, second = tmp_path/'read_only', tmp_path/'writable'
    first.mkdir(); (first/'existing').write_text('keep')
    original = Path.write_text
    def write(path, *args, **kwargs):
        if path.parent == first:
            raise OSError(30, 'Read-only file system')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', write)
    assert select_writable_cache([first, second], missing_models=1, min_free_bytes=1) == second
    assert (first/'existing').read_text() == 'keep'


def test_resumed_falcons_in_distinct_caches_need_no_download_budget(tmp_path, monkeypatch):
    import humanizers.zero_shot as module
    read_cache, write_cache = tmp_path/'old', tmp_path/'fallback'
    settings = dict(module.DEFAULT_MODELS)
    expected = {}
    for prefix, root in [('observer', read_cache), ('performer', write_cache)]:
        snapshot = root/('models--'+settings[prefix+'_model_id'].replace('/', '--'))/'snapshots'/settings[prefix+'_revision']
        snapshot.mkdir(parents=True)
        (snapshot/'config.json').write_text('{}')
        (snapshot/'model.safetensors').write_text('complete')
        expected[prefix] = snapshot
    monkeypatch.setenv('HUMANIZATION_HF_READ_CACHE', str(read_cache))
    monkeypatch.setenv('HUMANIZATION_HF_WRITE_CACHE', str(write_cache))
    def select(candidates, *, missing_models):
        assert missing_models == 0
        assert candidates == [write_cache]
        return write_cache
    monkeypatch.setattr(module, 'select_writable_cache', select)
    writable, snapshots = module._prepare_model_storage(settings)
    assert writable == write_cache
    assert snapshots == expected


def test_binoculars_observed_nll_and_cross_entropy_have_correct_direction_and_masks():
    # p(observer)=[.75,.25], q(performer)=[.5,.5] at every position.
    # Both NLL and cross-entropy equal log(2), hence ratio one.
    p = torch.log(torch.tensor([[[.75, .25]] * 4]))
    q = torch.log(torch.tensor([[[.5, .5]] * 4]))
    ids = torch.tensor([[0, 1, 0, 1]])
    mask = torch.tensor([[1, 1, 1, 0]])
    result = binoculars_components(p, q, ids, mask, positions_per_batch=2)
    assert result['nll_tokens'] == 2
    assert result['cross_entropy_tokens'] == 3  # Includes last real position.
    assert result['nll_sum'] == pytest.approx(2 * np.log(2))
    assert result['cross_entropy_sum'] == pytest.approx(3 * np.log(2))
    assert ratio_from_components([result])['score'] == pytest.approx(1)
    # The last input token is genuinely observed even if its ID is EOS/pad-like.
    # Attention mask, not equality to a token ID, decides inclusion.
    assert binoculars_components(p, q, ids, torch.ones_like(ids))['cross_entropy_tokens'] == 4


def test_cross_entropy_is_observer_expectation_of_performer_log_probabilities():
    p = torch.log(torch.tensor([[[.9, .1], [.9, .1]]]))
    q = torch.log(torch.tensor([[[.2, .8], [.2, .8]]]))
    result = binoculars_components(p, q, torch.tensor([[0, 1]]), torch.ones((1, 2), dtype=torch.long))
    expected_cross = -.9 * np.log(.2) - .1 * np.log(.8)
    assert result['nll_sum'] == pytest.approx(-np.log(.8))
    assert result['cross_entropy_sum'] / 2 == pytest.approx(expected_cross)
    assert ratio_from_components([result])['score'] == pytest.approx(-np.log(.8) / expected_cross)


def test_windows_cover_every_target_once_and_do_not_duplicate_denominator_boundaries():
    ids = list(range(11))
    windows = list(token_windows(ids, max_tokens=4))
    assert [x for w, _ in windows for x in w[1:]] == ids[1:]
    assert [x for w, skip in windows for x in w[skip:]] == ids
    assert max(len(w) for w, _ in windows) == 4
    with pytest.raises(ValueError, match='at least two'):
        list(token_windows([0], 4))


def test_low_human_quantile_has_strict_direction_and_tie_handling():
    values = np.arange(1, 21, dtype=float)
    threshold = calibrate_low_threshold(values, .05)
    assert threshold['threshold'] == 2
    assert sum(values < threshold['threshold']) == 1
    assert threshold['calibration_fpr'] == .05
    assert calibrate_low_threshold([.8, .8] + [1.] * 18, .05)['calibration_fpr'] == 0
    assert calibrate_low_threshold([.8, 1.], .05)['threshold'] == .8


def fixture_data():
    cohort = [{'id': 'cal', 'source_group_id': 'cal-g', 'split': 'calibration',
               'domain': 'Essay', 'generator': 'g', 'human': 'Human calibration.'}]
    rewrites = []
    for i in range(3):
        cohort.append({'id': f't{i}', 'source_group_id': f'g{i}', 'split': 'test',
                       'domain': 'Essay', 'generator': 'g', 'human': f'Human {i}.', 'machine': f'Machine {i}.'})
        rewrites.append({'id': f'r{i}', 'source_id': f't{i}', 'source_group_id': f'g{i}', 'humanizer': 'rewrite',
                         'original': f'Machine {i}.', 'modified': 'Rewrite.' if i != 1 else '',
                         'status': 'ok' if i != 1 else 'generation_error'})
    # Lower means AI: two original positives, one evasion and one failed writer.
    lookup = {('cal', 'calibration_human'): 1., ('t0', 'human'): 1.2,
        ('t1', 'human'): 1.1, ('t2', 'human'): 1.2,
        ('t0', 'machine'): .7, ('t1', 'machine'): .8, ('t2', 'machine'): 1.1,
        ('t0', 'rewrite'): 1.2, ('t2', 'rewrite'): 1.3}
    scores = pd.DataFrame([{k: v for k, v in r.items() if k != 'text'} | {'detector': 'bino',
        'score': lookup[r['source_id'], r['role']], 'text_sha256': hashlib.sha256(r['text'].encode()).hexdigest()}
        for r in text_records(cohort, rewrites)])
    return cohort, rewrites, scores


def test_low_ratio_asr_counts_failed_generations_and_excludes_already_undetected_original():
    cohort, rewrites, scores = fixture_data()
    config = {'zero_shot_detectors': [{'name': 'bino'}], 'bootstrap': 20}
    thresholds, summary = summarize_zero_shot(cohort, rewrites, scores, config)
    r = summary[summary.domain == 'overall'].iloc[0]
    assert thresholds['bino']['threshold'] == 1.
    assert r.baseline_detected == 2
    assert r.attack_successes == 1
    assert r.asr == .5
    assert r.generations_failed == 1
    assert r.baseline_detected_generation_failures == 1
    assert r.paired_original_tpr == .5
    assert r.paired_rewrite_tpr == 0
    assert r.human_test_fpr == 0
    with pytest.raises(ValueError, match='missing or duplicate'):
        summarize_zero_shot(cohort, rewrites, scores.iloc[:-1], config)

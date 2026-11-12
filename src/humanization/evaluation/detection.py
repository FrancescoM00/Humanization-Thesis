from __future__ import annotations
from collections import defaultdict
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import time
import numpy as np
import pandas as pd
from humanization.datasets.records import read_jsonl, digest, text_records, _score_key

def calibrate_threshold(scores, target_fpr=0.05):
    """Fissa un quantile empirico conservativo degli umani, con confronto stretto score > t.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or (not np.isfinite(values).all()):
        raise ValueError('Calibration needs a nonempty, finite vector of human scores')
    if not 0 <= target_fpr < 1 or np.any((values < 0) | (values > 1)):
        raise ValueError('FPR must be in [0, 1), and scores must be in [0, 1]')
    allowed = math.floor(len(values) * target_fpr)
    threshold = float(np.sort(values)[len(values) - allowed - 1])
    return {'threshold': threshold, 'calibration_n': len(values), 'target_fpr': target_fpr, 'calibration_fpr': float(np.mean(values > threshold)), 'decision': 'AI iff score > threshold; ties are classified human'}

def summarize_detection(cohort, rewrites, scores, config):
    """Restituisce soglie e riepiloghi per classificatore """
    expected_rows = text_records(cohort, rewrites)
    expected = {_score_key(row) for row in expected_rows}
    expected_metadata = {_score_key(row): row for row in expected_rows}
    detectors = [d for d in config['detectors'] if d.get('enabled', True)]
    names = [d['name'] for d in detectors]
    if not names or len(set(names)) != len(names):
        raise ValueError('Detector names must be nonempty and unique')
    frame = scores.copy().fillna({key: '' for key in ('source_group_id', 'domain', 'generator', 'split', 'role', 'humanizer', 'rewrite_id')})
    if set(frame['detector']) != set(names):
        raise ValueError('Score table does not cover exactly the configured detectors')
    thresholds, blocks = ({}, [])
    for name in names:
        part = frame[frame.detector == name].copy()
        keys = [_score_key(row) for row in part.to_dict('records')]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError(f'{name}: missing or duplicate scores; a scoring failure is not human classification')
        for row in part.to_dict('records'):
            reference = expected_metadata[_score_key(row)]
            for key in ('source_group_id', 'domain', 'generator', 'split', 'rewrite_id'):
                if str(row.get(key, '')) != str(reference.get(key, '')):
                    raise ValueError(f'{name}: score metadata do not match the frozen cohort/rewrites')
            if 'text_sha256' in row and row['text_sha256'] != hashlib.sha256(reference['text'].encode()).hexdigest():
                raise ValueError(f'{name}: score text hash does not match the input text')
        values = part.score.to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Detector scores must be finite values in [0, 1]')
        calibration = part[(part.role == 'calibration_human') & (part.split == 'calibration')]
        thresholds[name] = calibrate_threshold(calibration.score, config.get('calibration_fpr', 0.05))
        part['is_ai'] = part.score > thresholds[name]['threshold']
        blocks.append(part)
    if len(names) == 3 and config.get('majority_vote', True):
        combined = pd.concat(blocks, ignore_index=True)
        identifiers = ['source_id', 'source_group_id', 'domain', 'generator', 'split', 'role', 'humanizer', 'rewrite_id']
        vote = combined.groupby(identifiers, dropna=False, sort=False).is_ai.agg(['sum', 'count']).reset_index()
        if not vote['count'].eq(3).all():
            raise ValueError('Majority vote needs all three detector decisions for every text')
        vote['is_ai'] = vote['sum'] >= 2
        vote['score'] = vote['is_ai'].astype(float)
        vote['detector'] = 'majority_vote'
        calibration = vote[vote.role == 'calibration_human']
        thresholds['majority_vote'] = {'threshold': 0.5, 'calibration_n': len(calibration), 'target_fpr': None, 'calibration_fpr': float(calibration.is_ai.mean()), 'decision': 'AI iff at least two of three calibrated detectors vote AI'}
        blocks.append(vote)
    tests = [row for row in cohort if row['split'] == 'test']
    systems = sorted({row['humanizer'] for row in rewrites})
    rewritten = {(row['humanizer'], str(row['source_id'])): row for row in rewrites}
    summaries = []
    for part in blocks:
        name = part.detector.iloc[0]
        lookup = {_score_key(row): row for row in part.to_dict('records')}
        for domain in ['overall', *sorted({row['domain'] for row in tests})]:
            selected = [row for row in tests if domain == 'overall' or row['domain'] == domain]
            human = [lookup[str(row['id']), 'human', '']['is_ai'] for row in selected]
            machine = [lookup[str(row['id']), 'machine', '']['is_ai'] for row in selected]
            for system in systems:
                base, successes, groups, complete_before, complete_after, deltas = ([], [], [], [], [], [])
                n_ok, n_failed_baseline = (0, 0)
                for row, detected in zip(selected, machine):
                    source = str(row['id'])
                    rewrite = rewritten[system, source]
                    ok = rewrite['status'] == 'ok'
                    after = lookup[source, 'rewrite', system] if ok else None
                    n_ok += int(ok)
                    if detected:
                        base.append(source)
                        successes.append(bool(ok and (not after['is_ai'])))
                        groups.append(row['source_group_id'])
                        n_failed_baseline += int(not ok)
                    if ok:
                        before = lookup[source, 'machine', '']
                        complete_before.append(before['is_ai'])
                        complete_after.append(after['is_ai'])
                        deltas.append(after['score'] - before['score'])
                seed = int(digest([config.get('seed', 42), name, system, domain])[:8], 16)
                low, high = _asr_interval(successes, groups, seed, config.get('bootstrap', 500))
                rate = lambda values: float(np.mean(values)) if len(values) else math.nan
                summaries.append({'detector': name, 'humanizer': system, 'domain': domain, 'n_test_sources': len(selected), 'human_test_fpr': rate(human), 'original_machine_tpr': rate(machine), 'baseline_detected': len(base), 'attack_successes': sum(successes), 'asr': rate(successes), 'asr_ci95_low': low, 'asr_ci95_high': high, 'generations_ok': n_ok, 'generations_failed': len(selected) - n_ok, 'baseline_detected_generation_failures': n_failed_baseline, 'complete_pairs': len(complete_before), 'paired_original_tpr': rate(complete_before), 'paired_rewrite_tpr': rate(complete_after), 'paired_tpr_delta': rate(complete_after) - rate(complete_before), 'paired_original_fnr': 1 - rate(complete_before), 'paired_rewrite_fnr': 1 - rate(complete_after), 'mean_model_score_delta_complete': rate(deltas), 'threshold': thresholds[name]['threshold'], 'calibration_n': thresholds[name]['calibration_n'], 'calibration_fpr': thresholds[name]['calibration_fpr']})
    summary = pd.DataFrame(summaries)
    macro = []
    for (system, domain), part in summary[summary.detector.isin(names)].groupby(['humanizer', 'domain']):
        macro.append({'humanizer': system, 'domain': domain, 'n_detectors': len(part), 'n_detectors_with_asr': int(part.asr.notna().sum()), 'macro_mean_asr': part.asr.mean(), 'macro_mean_original_tpr': part.original_machine_tpr.mean(), 'macro_mean_human_fpr': part.human_test_fpr.mean(), 'macro_mean_paired_tpr_delta': part.paired_tpr_delta.mean()})
    return (thresholds, summary, pd.DataFrame(macro))

def calibrate_low_threshold(scores, target_fpr=0.05):
    """Quantile inferiore conservativo degli umani; score < soglia indica IA."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or (not np.isfinite(values).all()) or np.any(values <= 0):
        raise ValueError('Calibration needs finite positive Binoculars ratios')
    if not 0 <= target_fpr < 1:
        raise ValueError('FPR must be in [0, 1)')
    allowed = math.floor(len(values) * target_fpr)
    threshold = float(np.sort(values)[allowed])
    return {'threshold': threshold, 'calibration_n': len(values), 'target_fpr': target_fpr, 'calibration_fpr': float(np.mean(values < threshold)), 'decision': 'AI iff raw Binoculars ratio < threshold; ties are human', 'score_is_probability': False, 'orientation': 'lower_is_more_ai_like'}

def summarize_zero_shot(cohort, rewrites, scores, config):
    records = text_records(cohort, rewrites)
    key = lambda r: (str(r['source_id']), r['role'], r.get('humanizer') or '')
    expected = {key(r): r for r in records}
    detectors = [d for d in config['zero_shot_detectors'] if d.get('enabled', True)]
    names = [d['name'] for d in detectors]
    if not names or len(set(names)) != len(names) or set(scores.detector) != set(names):
        raise ValueError('Scores must cover exactly the configured zero-shot detectors')
    frame = scores.copy().fillna({k: '' for k in ['source_group_id', 'domain', 'generator', 'split', 'role', 'humanizer', 'rewrite_id']})
    thresholds, summaries = ({}, [])
    tests = [r for r in cohort if r['split'] == 'test']
    systems = sorted({r['humanizer'] for r in rewrites})
    rewritten = {(r['humanizer'], str(r['source_id'])): r for r in rewrites}
    for name in names:
        part = frame[frame.detector == name].copy()
        rows = part.to_dict('records')
        if len(rows) != len(expected) or {key(r) for r in rows} != set(expected):
            raise ValueError(f'{name}: missing or duplicate scores')
        for r in rows:
            reference = expected[key(r)]
            if any((str(r.get(k, '')) != str(reference.get(k, '')) for k in ['source_group_id', 'domain', 'generator', 'split', 'rewrite_id'])):
                raise ValueError(f'{name}: score metadata differ from the frozen inputs')
            if r.get('text_sha256') != hashlib.sha256(reference['text'].encode()).hexdigest():
                raise ValueError(f'{name}: score text hash differs from input')
        if not np.isfinite(part.score).all() or (part.score <= 0).any():
            raise ValueError('Binoculars ratios must be finite and positive')
        calibration = part[(part.role == 'calibration_human') & (part.split == 'calibration')]
        thresholds[name] = calibrate_low_threshold(calibration.score, config.get('calibration_fpr', 0.05))
        part['is_ai'] = part.score < thresholds[name]['threshold']
        lookup = {key(r): r for r in part.to_dict('records')}
        for domain in ['overall', *sorted({r['domain'] for r in tests})]:
            selected = [r for r in tests if domain == 'overall' or r['domain'] == domain]
            human = [lookup[str(r['id']), 'human', '']['is_ai'] for r in selected]
            machine = [lookup[str(r['id']), 'machine', '']['is_ai'] for r in selected]
            for system in systems:
                success, groups, before_all, after_all, deltas = ([], [], [], [], [])
                n_ok = failed_baseline = 0
                for source, detected in zip(selected, machine):
                    source_id = str(source['id'])
                    rewrite = rewritten[system, source_id]
                    ok = rewrite['status'] == 'ok'
                    after = lookup[source_id, 'rewrite', system] if ok else None
                    n_ok += int(ok)
                    if detected:
                        success.append(bool(ok and (not after['is_ai'])))
                        groups.append(source['source_group_id'])
                        failed_baseline += int(not ok)
                    if ok:
                        before = lookup[source_id, 'machine', '']
                        before_all.append(before['is_ai'])
                        after_all.append(after['is_ai'])
                        deltas.append(after['score'] - before['score'])
                low, high = _asr_interval(success, groups, int(digest([config.get('seed', 42), name, system, domain])[:8], 16), config.get('bootstrap', 500))
                mean = lambda x: float(np.mean(x)) if len(x) else math.nan
                summaries.append({'detector': name, 'humanizer': system, 'domain': domain, 'n_test_sources': len(selected), 'human_test_fpr': mean(human), 'original_machine_tpr': mean(machine), 'baseline_detected': len(success), 'attack_successes': sum(success), 'asr': mean(success), 'asr_ci95_low': low, 'asr_ci95_high': high, 'generations_ok': n_ok, 'generations_failed': len(selected) - n_ok, 'baseline_detected_generation_failures': failed_baseline, 'complete_pairs': len(before_all), 'paired_original_tpr': mean(before_all), 'paired_rewrite_tpr': mean(after_all), 'paired_tpr_delta': mean(after_all) - mean(before_all), 'paired_original_fnr': 1 - mean(before_all), 'paired_rewrite_fnr': 1 - mean(after_all), 'mean_model_score_delta_complete': mean(deltas), 'threshold': thresholds[name]['threshold'], 'calibration_n': thresholds[name]['calibration_n'], 'calibration_fpr': thresholds[name]['calibration_fpr'], 'score_orientation': 'lower_is_more_ai_like', 'score_is_probability': False})
    return (thresholds, pd.DataFrame(summaries))
from humanization.evaluation.statistics import _asr_interval

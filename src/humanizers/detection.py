"""Open detector evaluation with human-only calibration and conditional ASR.

Detector scores are token-weighted means over non-overlapping, complete chunks.
They are model scores, not calibrated probabilities of human authorship. Each
detector receives its own frozen threshold; a three-model majority vote combines
binary decisions, never raw scores. Failed generations remain in ASR denominators.
"""
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


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_ai_label(model_config, detector):
    """Require the explicit model-card class direction, checking actual config."""
    index = detector.get('ai_label_index')
    if isinstance(index, bool) or not isinstance(index, int) or index not in (0, 1):
        raise ValueError('Every binary detector requires explicit ai_label_index 0 or 1')
    if int(model_config.num_labels) != 2:
        raise ValueError('Only binary sequence-classification detectors are supported')
    labels = {int(key): str(value) for key, value in model_config.id2label.items()}
    if set(labels) != {0, 1}:
        raise ValueError('Detector config must expose exactly two labels, 0 and 1')
    expected = detector.get('ai_label')
    if expected is not None and labels[index].casefold() != str(expected).casefold():
        raise ValueError(f"AI label mismatch: config index {index} is {labels[index]!r}, expected {expected!r}")
    return index, labels


def calibrate_threshold(scores, target_fpr=.05):
    """Freeze a conservative empirical human quantile, with strict score > t.

    At most floor(n * target_fpr) calibration texts exceed the threshold, also
    with tied scores. This is an empirical calibration bound, not a guarantee on
    future human texts. No machine or rewritten score enters this calculation.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Calibration needs a nonempty, finite vector of human scores')
    if not 0 <= target_fpr < 1 or np.any((values < 0) | (values > 1)):
        raise ValueError('FPR must be in [0, 1), and scores must be in [0, 1]')
    allowed = math.floor(len(values) * target_fpr)
    threshold = float(np.sort(values)[len(values) - allowed - 1])
    return {'threshold': threshold, 'calibration_n': len(values),
            'target_fpr': target_fpr,
            'calibration_fpr': float(np.mean(values > threshold)),
            'decision': 'AI iff score > threshold; ties are classified human'}


def validate_inputs(cohort, rewrites):
    ids = [str(row['id']) for row in cohort]
    if len(ids) != len(set(ids)) or not ids or any(not value.strip() for value in ids):
        raise ValueError('Cohort IDs must be nonempty and unique')
    if any(row.get('split') not in {'calibration', 'test'} for row in cohort):
        raise ValueError('Cohort split must be calibration or test')
    by_id = {str(row['id']): row for row in cohort}
    groups = {split: {row['source_group_id'] for row in cohort if row['split'] == split}
              for split in ('calibration', 'test')}
    if not all(groups.values()) or groups['calibration'] & groups['test']:
        raise ValueError('Calibration/test must be nonempty and disjoint by source_group_id')
    for row in cohort:
        if not isinstance(row.get('human'), str) or not row['human'].strip():
            raise ValueError('Every source needs nonempty human text')
        if row['split'] == 'test' and (not isinstance(row.get('machine'), str) or not row['machine'].strip()):
            raise ValueError('Every test source needs nonempty machine text')
    systems, seen = defaultdict(set), set()
    test_ids = {str(row['id']) for row in cohort if row['split'] == 'test'}
    for row in rewrites:
        source = str(row['source_id'])
        system = row['humanizer']
        key = system, source
        if source not in test_ids or not system or key in seen:
            raise ValueError('Rewrites require one result per system/test source, without duplicates')
        if row.get('original') != by_id[source]['machine'] or row.get('source_group_id') != by_id[source]['source_group_id']:
            raise ValueError('Rewrite original/group does not match its frozen cohort source')
        if not row.get('status'):
            raise ValueError('Every rewrite result requires an explicit status')
        if row['status'] == 'ok' and (not isinstance(row.get('modified'), str) or not row['modified'].strip()):
            raise ValueError('An ok rewrite must contain nonempty modified text')
        seen.add(key)
        systems[system].add(source)
    if not systems or any(sources != test_ids for sources in systems.values()):
        raise ValueError('Generation file is incomplete: finish all test sources for each included system')
    return by_id


def text_records(cohort, rewrites):
    """Include calibration humans, paired test controls, and successful rewrites."""
    sources = validate_inputs(cohort, rewrites)
    rows = []
    for source in cohort:
        common = {key: source.get(key, '') for key in ('source_group_id', 'domain', 'generator', 'split')}
        common['source_id'] = str(source['id'])
        roles = [('calibration_human', source['human'])] if source['split'] == 'calibration' else [('human', source['human']), ('machine', source['machine'])]
        for role, text in roles:
            rows.append(dict(common, role=role, humanizer='', rewrite_id='', text=text))
    for rewrite in rewrites:
        if rewrite['status'] != 'ok':
            continue
        source = sources[str(rewrite['source_id'])]
        rows.append({'source_id': str(source['id']), 'source_group_id': source['source_group_id'],
                     'domain': source['domain'], 'generator': source.get('generator', ''),
                     'split': 'test', 'role': 'rewrite', 'humanizer': rewrite['humanizer'],
                     'rewrite_id': rewrite['id'], 'text': rewrite['modified']})
    return rows


def _score_key(row):
    return str(row['source_id']), row['role'], row.get('humanizer') or ''


def _asr_interval(success, group_ids, seed, n_bootstrap):
    if not len(success) or n_bootstrap < 1:
        return math.nan, math.nan
    grouped = defaultdict(list)
    for value, group in zip(success, group_ids):
        grouped[group].append(float(value))
    aggregates = np.array([(sum(values), len(values)) for _, values in sorted(grouped.items())])
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(aggregates), size=(n_bootstrap, len(aggregates)))
    sampled = aggregates[index].sum(axis=1)
    return tuple(np.quantile(sampled[:, 0] / sampled[:, 1], [.025, .975]))


def summarize_detection(cohort, rewrites, scores, config):
    """Return thresholds, per-classifier summaries, and a raw-score-free macro."""
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
    thresholds, blocks = {}, []
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
        thresholds[name] = calibrate_threshold(calibration.score, config.get('calibration_fpr', .05))
        part['is_ai'] = part.score > thresholds[name]['threshold']
        blocks.append(part)
    # Three-classifier majority is fixed a priori; no ensemble score averaging.
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
        thresholds['majority_vote'] = {'threshold': .5, 'calibration_n': len(calibration),
                                       'target_fpr': None, 'calibration_fpr': float(calibration.is_ai.mean()),
                                       'decision': 'AI iff at least two of three calibrated detectors vote AI'}
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
            human = [lookup[(str(row['id']), 'human', '')]['is_ai'] for row in selected]
            machine = [lookup[(str(row['id']), 'machine', '')]['is_ai'] for row in selected]
            for system in systems:
                base, successes, groups, complete_before, complete_after, deltas = [], [], [], [], [], []
                n_ok, n_failed_baseline = 0, 0
                for row, detected in zip(selected, machine):
                    source = str(row['id'])
                    rewrite = rewritten[system, source]
                    ok = rewrite['status'] == 'ok'
                    after = lookup[(source, 'rewrite', system)] if ok else None
                    n_ok += int(ok)
                    if detected:
                        base.append(source)
                        successes.append(bool(ok and not after['is_ai']))
                        groups.append(row['source_group_id'])
                        n_failed_baseline += int(not ok)
                    if ok:
                        before = lookup[(source, 'machine', '')]
                        complete_before.append(before['is_ai'])
                        complete_after.append(after['is_ai'])
                        deltas.append(after['score'] - before['score'])
                seed = int(digest([config.get('seed', 42), name, system, domain])[:8], 16)
                low, high = _asr_interval(successes, groups, seed, config.get('bootstrap', 500))
                rate = lambda values: float(np.mean(values)) if len(values) else math.nan
                summaries.append({'detector': name, 'humanizer': system, 'domain': domain,
                                  'n_test_sources': len(selected), 'human_test_fpr': rate(human),
                                  'original_machine_tpr': rate(machine),
                                  'baseline_detected': len(base), 'attack_successes': sum(successes),
                                  'asr': rate(successes), 'asr_ci95_low': low, 'asr_ci95_high': high,
                                  'generations_ok': n_ok, 'generations_failed': len(selected) - n_ok,
                                  'baseline_detected_generation_failures': n_failed_baseline,
                                  'complete_pairs': len(complete_before),
                                  'paired_original_tpr': rate(complete_before), 'paired_rewrite_tpr': rate(complete_after),
                                  'paired_tpr_delta': rate(complete_after) - rate(complete_before),
                                  'paired_original_fnr': 1 - rate(complete_before),
                                  'paired_rewrite_fnr': 1 - rate(complete_after),
                                  'mean_model_score_delta_complete': rate(deltas),
                                  'threshold': thresholds[name]['threshold'],
                                  'calibration_n': thresholds[name]['calibration_n'],
                                  'calibration_fpr': thresholds[name]['calibration_fpr']})
    summary = pd.DataFrame(summaries)
    macro = []
    for (system, domain), part in summary[summary.detector.isin(names)].groupby(['humanizer', 'domain']):
        macro.append({'humanizer': system, 'domain': domain, 'n_detectors': len(part),
                      'n_detectors_with_asr': int(part.asr.notna().sum()),
                      'macro_mean_asr': part.asr.mean(),
                      'macro_mean_original_tpr': part.original_machine_tpr.mean(),
                      'macro_mean_human_fpr': part.human_test_fpr.mean(),
                      'macro_mean_paired_tpr_delta': part.paired_tpr_delta.mean()})
    return thresholds, summary, pd.DataFrame(macro)


def _require_device(device):
    import torch
    selected = torch.device(device)
    if selected.type not in {'cpu', 'cuda'}:
        raise ValueError('Choose cpu or cuda')
    if selected.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: enable a Kaggle GPU accelerator or select --device cpu')
    return selected


def score_detector(records, detector, output, device='cpu'):
    """Run one checkpoint, persisting text-keyed scores after every document."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from semantics.metrics import token_chunks
    selected = _require_device(device)
    revision = detector.get('revision')
    if not revision:
        raise ValueError('Pin a detector checkpoint revision before inference')
    if detector.get('preprocessing', 'none') != 'none':
        raise ValueError('This experiment implements the documented raw-input variant only (preprocessing=none)')
    if detector['model_id'] == 'TrustSafeAI/RADAR-Vicuna-7B':
        version = tuple(int(x) for x in re.match(r'(\d+)\.(\d+)', torch.__version__).groups())
        if version < (2, 6):
            raise RuntimeError('RADAR uses PyTorch .bin weights: use torch >= 2.6 with the Kaggle CUDA build')
    fingerprint = digest({'detector': detector, 'device': str(selected),
                          'aggregation': 'full-text-nonoverlap-token-weighted-probability-v1',
                          'implementation': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers')})
    cache_path = Path(output) / 'detector_cache' / (fingerprint + '.jsonl')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if cache_path.exists():
        for item in read_jsonl(cache_path):
            cached[item['text_sha256']] = item
    tokenizer = AutoTokenizer.from_pretrained(detector['model_id'], revision=revision, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        detector['model_id'], revision=revision, trust_remote_code=False, torch_dtype=torch.float32)
    index, labels = validate_ai_label(model.config, detector)
    model.to(selected).eval()
    context_limit = getattr(model.config, 'max_position_embeddings', 512)
    # RoBERTa position embeddings include padding-position offsets.
    if getattr(model.config, 'model_type', '') == 'roberta':
        context_limit -= int(getattr(model.config, 'pad_token_id', 1)) + 1
    total_budget = min(int(detector.get('max_tokens', 512)), int(tokenizer.model_max_length), context_limit)
    budget = total_budget - tokenizer.num_special_tokens_to_add(pair=False)
    if budget < 1:
        raise ValueError('Detector context budget is too small')
    batch_size = int(detector.get('batch_size', 8))
    results = []
    started = time.monotonic()
    try:
        for position, row in enumerate(records):
            text = row['text']
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            if text_hash not in cached:
                chunks = token_chunks(text, tokenizer, max_tokens=budget)
                weights = [len(tokenizer.encode(chunk, add_special_tokens=False)) for chunk in chunks]
                predictions = []
                for start in range(0, len(chunks), batch_size):
                    batch = tokenizer(chunks[start:start + batch_size], padding=True, truncation=False, return_tensors='pt')
                    if batch['input_ids'].shape[1] > total_budget:
                        raise RuntimeError('Detector tokenizer exceeded the verified context; input was not truncated')
                    batch = {key: value.to(selected) for key, value in batch.items()}
                    with torch.inference_mode():
                        predictions.extend(model(**batch).logits.softmax(-1)[:, index].cpu().tolist())
                if not weights or not all(weights):
                    raise ValueError('A detector input is empty after tokenization')
                item = {'text_sha256': text_hash, 'score': float(np.average(predictions, weights=weights)),
                        'n_chunks': len(chunks), 'tokens': sum(weights)}
                cached[text_hash] = item
                with cache_path.open('a') as handle:
                    handle.write(json.dumps(item) + '\n')
            item = cached[text_hash]
            results.append({key: value for key, value in row.items() if key != 'text'} |
                           {'detector': detector['name'], **item})
            if position % 25 == 0 or position + 1 == len(records):
                print(f"{detector['name']}: {position + 1}/{len(records)} texts", flush=True)
    finally:
        del model, tokenizer
        gc.collect()
        if selected.type == 'cuda':
            torch.cuda.empty_cache()
    metadata = {'name': detector['name'], 'model_id': detector['model_id'], 'revision': revision,
                'id2label': labels, 'ai_label_index': index, 'device': str(selected),
                'preprocessing': 'none', 'full_text_coverage': True,
                'aggregation': 'token-weighted mean of non-overlapping chunk AI scores',
                'max_tokens_including_specials': total_budget, 'cache_fingerprint': fingerprint,
                'elapsed_seconds': time.monotonic() - started}
    return pd.DataFrame(results), metadata


def run(cohort_path, rewrites_path, config_path, output, device='cpu', analysis_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cohort, rewrites = read_jsonl(cohort_path), read_jsonl(rewrites_path)
    config = json.loads(Path(config_path).read_text())
    records = text_records(cohort, rewrites)
    metadata = []
    score_path = output / 'detector_scores.csv'
    if analysis_only:
        previous_path = output / 'detection_run.json'
        if not previous_path.exists():
            raise ValueError('Analysis-only requires the inference provenance in detection_run.json')
        previous = json.loads(previous_path.read_text())
        for key, path in [('cohort_sha256', cohort_path), ('rewrites_sha256', rewrites_path), ('config_sha256', config_path)]:
            if previous.get(key) != hashlib.sha256(Path(path).read_bytes()).hexdigest():
                raise ValueError('Analysis-only inputs/config differ from the recorded inference run')
        scores = pd.read_csv(score_path, dtype={'source_id': str})
    else:
        frames = []
        for detector in config['detectors']:
            if not detector.get('enabled', True):
                continue
            frame, details = score_detector(records, detector, output, device=device)
            frames.append(frame)
            metadata.append(details)
            frame.to_csv(output / ('scores_' + re.sub(r'[^a-zA-Z0-9_-]', '_', detector['name']) + '.csv'), index=False)
        scores = pd.concat(frames, ignore_index=True)
        scores.to_csv(score_path, index=False)
    thresholds, summary, macro = summarize_detection(cohort, rewrites, scores, config)
    (output / 'thresholds.json').write_text(json.dumps(thresholds, indent=2) + '\n')
    summary.to_csv(output / 'detection_summary.csv', index=False)
    macro.to_csv(output / 'detection_macro_summary.csv', index=False)
    manifest = {'cohort_sha256': hashlib.sha256(Path(cohort_path).read_bytes()).hexdigest(),
                'rewrites_sha256': hashlib.sha256(Path(rewrites_path).read_bytes()).hexdigest(),
                'config_sha256': hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                'analysis_only': analysis_only, 'detectors': metadata,
                'n_text_records_per_detector': len(records),
                'asr_denominator': 'test sources detected AI before rewriting; failed generations remain failures',
                'calibration': 'human-only, source-group-disjoint from test; score > threshold',
                'majority_vote': 'two of three calibrated binary decisions, only when exactly three detectors are used',
                'asr_intervals': 'source-group bootstrap, fixed thresholds and checkpoints; calibration/model uncertainty excluded',
                'bootstrap': config.get('bootstrap', 500),
                'macro_mean': 'equal detector weight over defined ASRs; majority vote is excluded',
                'naturalness_measured': False,
                'python': sys.version.split()[0]}
    # Preserve inference provenance when only recomputing aggregate statistics.
    if analysis_only and (output / 'detection_run.json').exists():
        previous = json.loads((output / 'detection_run.json').read_text())
        manifest['detectors'] = previous.get('detectors', [])
    (output / 'detection_run.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(summary[summary.domain == 'overall'].to_string(index=False), flush=True)
    return summary

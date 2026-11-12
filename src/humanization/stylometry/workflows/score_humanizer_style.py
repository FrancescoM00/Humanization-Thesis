"""Describe humanizer style with scales fitted on separate human controls.

No detector score enters feature selection or calibration. Distances are style
changes, not probabilities of human authorship, naturalness or correctness.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from humanization.semantics.lexical_guards import definitions, score_lexical_guards
from humanization.stylometry.touched import DISTANCE_FEATURE_CANDIDATES, SHORTLIST, pos_js
from humanization.stylometry.features import FEATURE_SCHEMA, extract_records

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def validate_inputs(cohort, pairs):
    """Reject source leakage and pairs that do not belong to the frozen test."""
    if not cohort or len({r['id'] for r in cohort}) != len(cohort):
        raise ValueError('Require nonempty cohort with unique IDs')
    groups = {}
    for row in cohort:
        if row['split'] not in {'calibration', 'test'}:
            raise ValueError('Unknown cohort split')
        if not all((isinstance(row.get(k), str) and row[k].strip() for k in ['source_group_id', 'human', 'machine', 'domain', 'generator'])):
            raise ValueError('Missing cohort metadata or text')
        previous = groups.setdefault(row['source_group_id'], row['split'])
        if previous != row['split']:
            raise ValueError('Source group overlaps calibration and test')
    calibration = [r for r in cohort if r['split'] == 'calibration']
    if len({r['source_group_id'] for r in calibration}) < 2:
        raise ValueError('At least two separate calibration human sources are needed')
    tests = {r['id']: r for r in cohort if r['split'] == 'test'}
    if len({p['id'] for p in pairs}) != len(pairs):
        raise ValueError('Duplicate pair IDs')
    valid = []
    for pair in pairs:
        if pair['source_id'] not in tests:
            raise ValueError('Pair source is absent from the frozen test cohort')
        source = tests[pair['source_id']]
        for key in ['source_group_id', 'domain', 'generator']:
            if pair.get(key) != source[key]:
                raise ValueError(f'Pair/cohort metadata differ: {key}')
        if pair['original'] != source['machine']:
            raise ValueError('Pair original differs from the frozen machine text')
        if not pair.get('humanizer'):
            raise ValueError('Every pair needs an explicit humanizer label')
        if pair.get('status', 'ok') == 'ok':
            if not isinstance(pair.get('modified'), str) or not pair['modified'].strip():
                raise ValueError('An ok pair must contain a nonempty rewrite')
            valid.append(pair)
    if not valid:
        raise ValueError('No successful pairs to score')
    return (calibration, valid)

def fit_scales(calibration, features):
    """Equal source weight; missing/constant calibration features are excluded."""
    rows = []
    for row in calibration:
        rows.append({'source_group_id': row['source_group_id'], **features[digest(row['human'])]})
    frame = pd.DataFrame(rows).groupby('source_group_id')[list(FEATURE_SCHEMA)].mean()
    means, scales, count = (frame.mean(), frame.std(ddof=1), frame.count())
    records = []
    for name in FEATURE_SCHEMA:
        finite = count[name] >= 2 and np.isfinite(scales[name]) and (scales[name] > 1e-12)
        retained = name in DISTANCE_FEATURE_CANDIDATES and finite
        reason = 'retained' if retained else 'not_predeclared_for_distance' if name not in DISTANCE_FEATURE_CANDIDATES else 'constant_or_missing_in_calibration'
        records.append({'feature': name, 'calibration_mean': means[name], 'calibration_sd_ddof1': scales[name], 'n_calibration_sources': count[name], 'distance_retained': retained, 'reason': reason})
    table = pd.DataFrame(records)
    names = table.loc[table.distance_retained, 'feature'].tolist()
    if not names:
        raise ValueError('No variable predeclared feature remains on calibration humans')
    return (names, means, scales, table)

def style_table(pairs, features, names, means, scales):
    rows = []
    for pair in pairs:
        before, after = [features[digest(pair[key])] for key in ['original', 'modified']]
        a = np.asarray([before[name] for name in names], dtype=float)
        b = np.asarray([after[name] for name in names], dtype=float)
        scale, center = (scales[names].to_numpy(), means[names].to_numpy())

        def rms(vector):
            return float(np.sqrt(np.mean((vector / scale) ** 2))) if np.isfinite(vector).all() else np.nan
        row = {key: pair[key] for key in ['id', 'source_id', 'source_group_id', 'domain', 'generator', 'humanizer']}
        row.update(style_rms=rms(b - a), pos_js=pos_js(before, after), rms_original_to_human_centroid=rms(a - center), rms_modified_to_human_centroid=rms(b - center), rms_missing_features=int((~np.isfinite(a) | ~np.isfinite(b)).sum()), original_n_words=before['n_words'], modified_n_words=after['n_words'], original_n_sentences=before['n_sentences'], modified_n_sentences=after['n_sentences'], word_ratio=after['n_words'] / before['n_words'] if before['n_words'] else np.nan, unchanged=pair['original'] == pair['modified'])
        row['delta_human_centroid_distance'] = row['rms_modified_to_human_centroid'] - row['rms_original_to_human_centroid']
        for feature in SHORTLIST:
            row['before__' + feature], row['after__' + feature] = (before[feature], after[feature])
            row['delta__' + feature] = after[feature] - before[feature]
        rows.append(row)
    return pd.DataFrame(rows)

def review_forms(cohort, pairs, output, per_cell, seed):
    """Export separate blinded annotation forms and keys; never automatic gold."""
    rng = random.Random(seed)
    cells = {}
    for pair in pairs:
        cells.setdefault((pair['domain'], pair['humanizer']), []).append(pair)
    sampled = []
    for key in sorted(cells):
        candidates = sorted(cells[key], key=lambda row: row['id'])
        sampled.extend(rng.sample(candidates, min(per_cell, len(candidates))))
    rng.shuffle(sampled)
    semantic, semantic_key = ([], [])
    for index, pair in enumerate(sampled, 1):
        review_id = f'S{index:04d}'
        semantic.append({'review_id': review_id, 'original': pair['original'], 'modified': pair['modified'], 'semantic_preserved_yes_partial_no': '', 'error_type': '', 'notes': ''})
        semantic_key.append({'review_id': review_id, **{key: pair[key] for key in ['id', 'source_id', 'source_group_id', 'domain', 'humanizer']}})
    for name, rows in [('semantic_review_form', semantic), ('semantic_review_key', semantic_key)]:
        pd.DataFrame(rows).to_csv(output / (name + '.csv'), index=False)
    (output / 'review_instructions.md').write_text('# Annotazione manuale esplorativa\n\nConsegnare al valutatore soltanto i file `*_review_form.csv`, non le chiavi. La semantica richiede di confrontare originale e riscrittura: yes conserva tutte le informazioni rilevanti, partial presenta una perdita/aggiunta/modifica, no altera il contenuto principale. Specificare error_type e motivazione; usare notes per i casi ambigui.\n\nL’ordine è casuale. Le chiavi separate conservano condizioni e fonti per l’analisi successiva. Annotazioni vuote non sono gold.\n')
    return {'semantic_pairs': len(semantic)}

def _argument_error(message):
    raise ValueError(message)

def run(args):
    if args.review_per_cell < 1:
        _argument_error('--review-per-cell must be positive')
    cohort, all_pairs = (read_jsonl(args.cohort), read_jsonl(args.pairs))
    calibration, pairs = validate_inputs(cohort, all_pairs)
    args.output.mkdir(parents=True, exist_ok=True)
    unique, references = ({}, [])
    for row in cohort:
        roles = ['human'] if row['split'] == 'calibration' else ['human', 'machine']
        for role in roles:
            text = row[role]
            unique[digest(text)] = text
            references.append({'source_id': row['id'], 'source_group_id': row['source_group_id'], 'split': row['split'], 'role': role, 'pair_id': '', 'text_sha256': digest(text)})
    for pair in pairs:
        text = pair['modified']
        unique[digest(text)] = text
        references.append({'source_id': pair['source_id'], 'source_group_id': pair['source_group_id'], 'split': 'test', 'role': 'rewrite', 'pair_id': pair['id'], 'text_sha256': digest(text)})
    extracted = extract_records([{'text_sha256': key, 'text': text} for key, text in unique.items()], model=args.spacy_model)
    pd.DataFrame(extracted).to_csv(args.output / 'features.csv', index=False)
    pd.DataFrame(references).to_csv(args.output / 'text_index.csv', index=False)
    features = {r['text_sha256']: {k: r[k] for k in FEATURE_SCHEMA} for r in extracted}
    names, means, scales, calibration_table = fit_scales(calibration, features)
    calibration_table.to_csv(args.output / 'calibration_scales.csv', index=False)
    style_table(pairs, features, names, means, scales).to_csv(args.output / 'style_pairs.csv', index=False)
    score_lexical_guards(pairs).to_csv(args.output / 'guards.csv', index=False)
    reviews = review_forms(cohort, pairs, args.output, args.review_per_cell, args.seed)
    manifest = {'cohort_sha256': hashlib.sha256(args.cohort.read_bytes()).hexdigest(), 'pairs_sha256': hashlib.sha256(args.pairs.read_bytes()).hexdigest(), 'n_pairs_received': len(all_pairs), 'n_pairs_scored': len(pairs), 'excluded_status_counts': dict(Counter((p.get('status', 'ok') for p in all_pairs if p.get('status', 'ok') != 'ok'))), 'calibration_humans': len(calibration), 'calibration_sources': len({r['source_group_id'] for r in calibration}), 'calibration_domains': dict(Counter((r['domain'] for r in calibration))), 'calibration_uses_machine_or_test_texts': False, 'scale_weighting': 'equal source weight; mean of human feature rows within source; sample SD ddof=1', 'n_features_extracted': len(FEATURE_SCHEMA), 'distance_candidates': DISTANCE_FEATURE_CANDIDATES, 'distance_retained': names, 'distance_dropped': [n for n in DISTANCE_FEATURE_CANDIDATES if n not in names], 'imputation': 'none; absent or constant calibration features excluded; incomplete test RMS is NaN', 'style_rms_definition': 'sqrt(mean(((phi_modified-phi_original)/calibration_human_sd)**2))', 'pos_js_definition': 'sqrt Jensen-Shannon divergence of POS distributions, base 2', 'centroid_distances_are_naturalness_or_authorship_scores': False, 'length_matched_historical_passage_level_not_OpAI_sentence_scales': True, 'lexical_guard_definitions': definitions(), 'reviews': reviews, 'automatic_semantic_or_naturalness_verdict_created': False, 'spacy_model': args.spacy_model}
    (args.output / 'style_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'pairs_scored': len(pairs), 'distance_features': len(names), 'output': str(args.output)}, indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'cohort': None, 'pairs': None, 'output': None, 'spacy_model': 'en_core_web_sm', 'review_per_cell': 2, 'seed': 42}
    values.update(overrides)
    return SimpleNamespace(**values)

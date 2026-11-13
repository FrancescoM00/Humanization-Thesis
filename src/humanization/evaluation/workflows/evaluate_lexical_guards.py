"""Estrae le variazioni delle guardie lessicali e valuta il loro contributo al NLI sul test escluso."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from humanization.semantics.lexical_guards import GUARD_FEATURES, definitions, score_lexical_guards, self_check
from humanization.evaluation.semantic import _metadata, _cohorts, _source_weights, _threshold, _group_key, _operating_point, source_bootstrap_auc

def read_pairs(paths):
    rows = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row['id'] in rows and any((row.get(k) != rows[row['id']].get(k) for k in ['original', 'modified'])):
                raise ValueError('Different texts share a guard pair ID')
            rows[row['id']] = row
    return list(rows.values())

def fit_model(calibration, features):
    weights = _source_weights(calibration)
    for label in [0, 1]:
        mask = calibration.is_altered.to_numpy() == label
        weights[mask] /= weights[mask].sum()
    weights *= len(weights) / weights.sum()
    scaler = StandardScaler().fit(calibration[features], sample_weight=weights)
    model = LogisticRegression(C=1, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(scaler.transform(calibration[features]), calibration.is_altered, sample_weight=weights)
    return (scaler, model)

def apply_frozen_models(pairs, guards, scores, models):
    """Valuta ogni coppia fornita; non sostituisce NLI mancanti con guardie o etichette."""
    from scipy.special import expit
    predicted = guards.merge(scores[['id', 'nli_forward_min', 'nli_reverse_min']], on='id', how='left', validate='one_to_one')
    predicted['risk_nli_min'] = 1 - predicted[['nli_forward_min', 'nli_reverse_min']].min(axis=1, skipna=False)
    for name, params in models.items():
        values = predicted[params['features']].to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1)
        output = np.full(len(values), np.nan)
        z = (values[valid] - np.array(params['scaler_mean'])) / np.array(params['scaler_scale'])
        output[valid] = expit(z @ np.array(params['coefficients']) + params['intercept'])
        predicted[name] = output
    predicted['risk_guard_max'] = predicted[GUARD_FEATURES].max(axis=1)
    mapping = {pair['id']: pair for pair in pairs}
    predicted['source_group'] = predicted.id.map(lambda value: _group_key(mapping[value]))
    for key in ['dataset', 'perturbation', 'evaluation_split']:
        predicted[key] = predicted.id.map(lambda value: mapping[value].get(key))
    return predicted

def evaluate(pairs, guards, scores, output, n_bootstrap=1000, seed=42):
    meta = _metadata(pairs, seed)
    frame = meta.merge(guards, on='id', validate='one_to_one').merge(scores[['id', 'nli_forward_min', 'nli_reverse_min']], on='id', validate='one_to_one')
    frame['risk_nli_min'] = 1 - frame[['nli_forward_min', 'nli_reverse_min']].min(axis=1)
    cohorts = _cohorts(frame)
    primary = cohorts['opai_matched_primary']
    cal = primary[primary.evaluation_split == 'calibration']
    assert not set(cal.source_group) & set(primary[primary.evaluation_split == 'test'].source_group)
    models = {}
    for name, features in [('risk_guards_logistic', GUARD_FEATURES), ('risk_nli_guards_logistic', ['risk_nli_min'] + GUARD_FEATURES)]:
        scaler, model = fit_model(cal, features)
        frame[name] = model.predict_proba(scaler.transform(frame[features]))[:, 1]
        models[name] = {'features': features, 'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(), 'coefficients': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]), 'C': 1, 'penalty': 'l2', 'fit_cohort': 'opai_matched_primary:calibration', 'probabilities_are_proxy_classifier_scores_not_semantic_probabilities': True}
    frame['risk_guard_max'] = frame[GUARD_FEATURES].max(axis=1)
    risk_columns = ['risk_nli_min', 'risk_guard_max'] + GUARD_FEATURES + list(models)
    predictions = apply_frozen_models(pairs, guards, scores, models)
    predictions[['id', 'source_group', 'dataset', 'perturbation', 'evaluation_split'] + risk_columns].to_csv(output / 'guard_model_scores.csv', index=False)
    rng = np.random.default_rng(seed)
    auc_rows, strata_rows, threshold_rows, alerts = ([], [], [], [])
    threshold_by_metric = {metric: _threshold(cal.assign(**{name: frame[name] for name in models}), frame[metric], 0.1) for metric in risk_columns}
    cohorts = _cohorts(frame)
    for cohort_name in ['opai_matched_primary', 'opai_matched_subtle_secondary']:
        cohort = cohorts[cohort_name]
        for split in ['calibration', 'test']:
            part = cohort[cohort.evaluation_split == split]
            for metric in risk_columns:
                result = source_bootstrap_auc(part, frame[metric], rng, n_bootstrap)
                if result:
                    auc_rows.append({'cohort': cohort_name, 'split': split, 'metric': metric, **result})
                threshold_rows.append({'cohort': cohort_name, 'split': split, 'metric': metric, 'threshold': threshold_by_metric[metric], 'threshold_is_secondary_calibration_diagnostic': True, **_operating_point(part, frame[metric], threshold_by_metric[metric])})
        test = cohort[cohort.evaluation_split == 'test']
        for condition, negative in test[test.is_altered == 1].groupby('perturbation'):
            positive = test[test.id.isin(set(negative.matched_positive_id))]
            part = pd.concat([positive, negative])
            for metric in risk_columns:
                result = source_bootstrap_auc(part, frame[metric], rng, n_bootstrap)
                if result:
                    strata_rows.append({'cohort': cohort_name, 'condition': condition, 'metric': metric, **result, 'secondary_sensitivity': _operating_point(negative, frame[metric], threshold_by_metric[metric])['sensitivity']})
    for metric in risk_columns:
        for operation, part in [('all', frame[(frame.dataset == 'opai_positive') & (frame.evaluation_split == 'test')])]:
            for name, subgroup in [(operation, part)] + list(part.groupby('perturbation')):
                point = _operating_point(subgroup, frame[metric], threshold_by_metric[metric])
                alerts.append({'operation': name, 'metric': metric, 'threshold': threshold_by_metric[metric], 'n_alerts': int((frame.loc[subgroup.index, metric] > threshold_by_metric[metric]).sum()), 'n_presumed_preserved': len(subgroup), 'false_positive_rate': point['false_positive_rate']})
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(output / 'metric_auc.csv', index=False)
    strata = pd.DataFrame(strata_rows)
    strata.to_csv(output / 'auc_by_condition.csv', index=False)
    pd.DataFrame(threshold_rows).to_csv(output / 'secondary_operating_points.csv', index=False)
    alert_table = pd.DataFrame(alerts)
    alert_table.to_csv(output / 'positive_all_alerts.csv', index=False)
    manifest = {'seed': seed, 'n_bootstrap': n_bootstrap, 'train_ids': cal.id.tolist(), 'train_source_groups': sorted(cal.source_group.unique()), 'n_train': len(cal), 'n_train_sources': cal.source_group.nunique(), 'models': models, 'feature_definitions': definitions(), 'rules_are_fixed_not_tuned_on_test': True, 'labels_are_semantic_proxy_not_human_gold': True, 'threshold_policy': 'Secondary diagnostics only: primary calibration presumed positives, nominal FPR <=10%; no transferred universal threshold.', 'guards_code_sha256': hashlib.sha256((ROOT / 'src/humanization/semantics/lexical_guards.py').read_bytes()).hexdigest()}
    (output / 'model_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    lookup = guards.set_index('id')
    pair_lookup = {pair['id']: pair for pair in pairs}
    hedge_audit = []
    for pair in pairs:
        if pair.get('dataset') != 'opai_matched_negative' or pair.get('perturbation') != 'matched_hedge_removal':
            continue
        positive = pair_lookup[pair['metadata']['matched_positive_id']]
        left, changed = (lookup.loc[positive['id']], lookup.loc[pair['id']])
        before, good, corrupted = [int(value) for value in [left.guard_hedge_original_count, left.guard_hedge_modified_count, changed.guard_hedge_modified_count]]
        target_before = pair.get('target_original', '').casefold()
        target_after = pair.get('target_modified', '').casefold()
        hedge_audit.append({'id': pair['id'], 'matched_positive_id': positive['id'], 'evaluation_split': pair.get('evaluation_split'), 'original_hedge_count': before, 'positive_output_hedge_count': good, 'corrupted_output_hedge_count': corrupted, 'positive_absolute_count_change': abs(good - before), 'corrupted_absolute_count_change': abs(corrupted - before), 'corruption_brings_count_closer_to_original': abs(corrupted - before) < abs(good - before), 'possibly_removed_but_could_remains': 'could possibly' in target_before and 'possibly' not in target_after and ('could' in target_after)})
    hedge_audit = pd.DataFrame(hedge_audit)
    hedge_audit.to_csv(output / 'hedge_count_audit.csv', index=False)
    direct_hedge = frame[(frame.dataset == 'controlled_subtle') & (frame.perturbation == 'hedge_removal')]
    all_positive = frame[frame.dataset == 'opai_positive']
    direct_cohort = pd.concat([all_positive, direct_hedge])
    diagnostic_metrics = ['guard_hedge_count_loss'] + GUARD_FEATURES + ['risk_nli_min'] + list(models)
    diagnostic_rows = []
    for split in ['all', 'calibration', 'test']:
        part = direct_cohort if split == 'all' else direct_cohort[direct_cohort.evaluation_split == split]
        for metric in diagnostic_metrics:
            result = source_bootstrap_auc(part, frame[metric], rng, n_bootstrap)
            if result:
                diagnostic_rows.append({'cohort': 'direct_mgt_hedge_vs_opai_cross_corpus_diagnostic', 'split': split, 'metric': metric, **result})
    pd.DataFrame(diagnostic_rows).to_csv(output / 'direct_hedge_diagnostic.csv', index=False)
    loss_alerts = []
    loss_threshold = _threshold(cal, frame.guard_hedge_count_loss, 0.1)
    for split in ['calibration', 'test']:
        for population, source in [('all_opai_positive', all_positive), ('direct_mgt_hedge', direct_hedge)]:
            part = source[source.evaluation_split == split]
            for rule, threshold in [('any_hedge_count_loss', 0), ('secondary_calibrated_loss', loss_threshold)]:
                point = _operating_point(part, frame.guard_hedge_count_loss, threshold)
                loss_alerts.append({'split': split, 'population': population, 'rule': rule, 'threshold': threshold, 'n_alerts': int((part.guard_hedge_count_loss > threshold).sum()), **point})
    pd.DataFrame(loss_alerts).to_csv(output / 'hedge_loss_alerts.csv', index=False)
    selected = ['risk_nli_min', 'risk_guards_logistic', 'risk_nli_guards_logistic']
    main = auc[(auc.split == 'test') & auc.metric.isin(selected)]
    return main

def _argument_error(message):
    raise ValueError(message)

def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if args.self_check:
        print(json.dumps(self_check()), flush=True)
    paths = args.pairs or [ROOT / 'data/processed/semantic_fairness/pairs.jsonl']
    pairs = read_pairs(paths)

    def progress(done, total):
        (args.output / 'progress.json').write_text(json.dumps({'processed_unique_texts': done, 'total_unique_texts': total}) + '\n')
    guards = score_lexical_guards(pairs, progress)
    guards.to_csv(args.output / 'guard_scores.csv', index=False)
    (args.output / 'guard_definitions.json').write_text(json.dumps(definitions(), indent=2) + '\n')
    (args.output / 'extraction_manifest.json').write_text(json.dumps({'pairs_files': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths], 'n_pairs': len(guards), 'guard_features': GUARD_FEATURES}, indent=2) + '\n')
    print(f'Extracted lexical guard scores for {len(guards)} pairs.', flush=True)
    if not args.extract_only:
        scores = pd.read_csv(args.scores)
        result = evaluate(pairs, guards, scores, args.output, args.bootstrap)
        print(result.to_string(index=False), flush=True)

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'pairs': None, 'output': ROOT / 'results/new_runs/03_semantics/fairness/guards_full', 'scores': ROOT / 'results/published/03_semantics/fairness/nli.csv', 'bootstrap': 1000, 'extract_only': False, 'self_check': False}
    values.update(overrides)
    return SimpleNamespace(**values)

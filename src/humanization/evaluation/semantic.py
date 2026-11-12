"""Compare semantic scores with positive controls, keeping source groups intact.

The class of interest is an intended semantic alteration. Similarity/support
scores therefore receive a fixed minus sign; unsupported fractions keep their
sign. Scores are never flipped after seeing an AUC below 0.5. Operation names
and rule-based corruptions are proxy labels, not human semantic gold labels.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from humanization.evaluation.reporting import markdown_table


POSITIVE_OPERATIONS = {'opai_polish', 'opai_paraphrase', 'opai_style', 'opai_style_rewrite'}
EXCLUDED = {'identity', 'sentence_shuffle', 'shuffle', 'opai_compress', 'opai_expand'}
CORE_METRICS = ['lexical_rouge_l_f1', 'lexical_bleu', 'embedding_cosine', 'bert_chunk_f1']
AGGREGATORS = ['mean', 'q10', 'min', 'low_support_fraction']
MATCHED_SUBTLE = {'matched_number_change', 'matched_quantifier_strengthening',
                  'matched_hedge_removal', 'matched_unsupported_detail'}


def _group_key(pair):
    meta = pair.get('metadata', {})
    key = pair.get('source_group_id') or meta.get('source_group_id') or pair['source_id']
    # Prefix corpora so unrelated IDs can never accidentally become one source.
    family = 'opai' if pair.get('dataset', '').startswith('opai') else 'mgtbench'
    return family + ':' + str(key)


def _metadata(pairs, seed):
    records = []
    splits = {}
    for pair in pairs:
        dataset = pair.get('dataset', '')
        operation = pair.get('perturbation', '')
        if dataset in {'local_control', 'manual_sanity'} or operation in EXCLUDED:
            continue
        meta = pair.get('metadata', {})
        if dataset == 'opai_positive' and operation in POSITIVE_OPERATIONS:
            label = 0
        elif dataset == 'opai_matched_negative':
            label = 1
        elif dataset in {'controlled', 'controlled_subtle'}:
            label = 1
        else:
            continue
        group = _group_key(pair)
        supplied = meta.get('evaluation_split') or pair.get('evaluation_split')
        if supplied:
            split = {'holdout': 'test', 'validation': 'test', 'train': 'calibration'}.get(supplied, supplied)
            if split not in {'calibration', 'test'}:
                raise ValueError(f'Unknown evaluation split {supplied!r}')
            if group in splits and splits[group] != split:
                raise ValueError('Source group crosses the supplied calibration/test split')
            splits[group] = split
        records.append({'id': pair['id'], 'source_group': group, 'source_id': pair['source_id'],
                        'dataset': dataset, 'perturbation': operation, 'domain': pair.get('domain'),
                        'generator': pair.get('generator'), 'is_altered': label,
                        'positive_operation': meta.get('operation', operation if label == 0 else ''),
                        'matched_positive_id': meta.get('matched_positive_id'),
                        'original_words': len(pair['original'].split()),
                        'length_ratio': len(pair['modified'].split()) / len(pair['original'].split())})
    for row in records:
        group = row['source_group']
        # Deterministic split is independent of all scores and alteration labels.
        if group not in splits:
            key = hashlib.sha256(f'{seed}:{group}'.encode()).digest()
            splits[group] = 'calibration' if int.from_bytes(key[:8], 'big') % 2 == 0 else 'test'
        row['evaluation_split'] = splits[group]
    frame = pd.DataFrame(records)
    if frame.id.duplicated().any():
        raise ValueError('Repeated pair IDs in prepared semantic extension')
    return frame


def _metric_scores(frame):
    """Return alteration-oriented scores and their fixed display definitions."""
    risk = pd.DataFrame(index=frame.index)
    definitions = []
    metrics = [name for name in CORE_METRICS if name in frame]
    metrics += [name for name in frame if name.startswith('bert_chunk_f1') and
                name != 'bert_chunk_f1' and ('rescal' in name or ('baseline' in name and 'distance' in name))]
    for name in metrics:
        risk[name] = -pd.to_numeric(frame[name], errors='coerce')
        definitions.append({'metric': name, 'family': 'similarity', 'risk_definition': '-' + name})
    for aggregator in AGGREGATORS:
        forward, reverse = 'nli_forward_' + aggregator, 'nli_reverse_' + aggregator
        for name in (forward, reverse):
            if name in frame:
                sign = 1 if aggregator == 'low_support_fraction' else -1
                risk[name] = sign * pd.to_numeric(frame[name], errors='coerce')
                definitions.append({'metric': name, 'family': 'nli_directional',
                                    'risk_definition': ('' if sign == 1 else '-') + name})
        if forward in frame and reverse in frame:
            name = 'nli_bidirectional_' + aggregator
            values = frame[[forward, reverse]].apply(pd.to_numeric, errors='coerce')
            # Requiring preservation in both directions: weakest support, or
            # largest unsupported fraction, governs the bidirectional score.
            if aggregator == 'low_support_fraction':
                risk[name] = values.max(axis=1, skipna=False)
                definition = f'max({forward}, {reverse})'
            else:
                risk[name] = -values.min(axis=1, skipna=False)
                definition = f'-min({forward}, {reverse})'
            definitions.append({'metric': name, 'family': 'nli_bidirectional', 'risk_definition': definition})
    return risk, pd.DataFrame(definitions)


def _source_weights(frame):
    # Each source receives equal total weight *within each class*. A source
    # with several editors or corruptions must not dominate either class.
    return 1 / frame.groupby(['source_group', 'is_altered']).id.transform('count').to_numpy()


def _auc_sorted(labels, scores, weights):
    order = np.argsort(scores, kind='stable')
    ordered = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1]
    positive = weights[order] * labels[order]
    negative = weights[order] * (1 - labels[order])
    pos_ties = np.add.reduceat(positive, starts)
    neg_ties = np.add.reduceat(negative, starts)
    denominator = positive.sum() * negative.sum()
    return float(np.sum(pos_ties * (np.cumsum(neg_ties) - .5 * neg_ties)) / denominator) if denominator else np.nan


def source_bootstrap_auc(frame, score, rng, n_bootstrap=1000):
    score = score.reindex(frame.index)
    valid = frame.loc[np.isfinite(score)].copy()
    values = np.asarray(score.loc[valid.index], dtype=float)
    labels = valid.is_altered.to_numpy(dtype=int)
    if len(valid) < 2 or len(np.unique(labels)) != 2:
        return None
    weights = _source_weights(valid)
    groups, unique = pd.factorize(valid.source_group, sort=True)
    point = _auc_sorted(labels, values, weights)
    n_preserved_sources = valid.loc[valid.is_altered == 0, 'source_group'].nunique()
    n_altered_sources = valid.loc[valid.is_altered == 1, 'source_group'].nunique()
    base = {'auc': point, 'n_pairs': len(valid), 'n_preserved': int((labels == 0).sum()),
            'n_altered': int(labels.sum()), 'n_sources': len(unique),
            'n_preserved_sources': n_preserved_sources, 'n_altered_sources': n_altered_sources}
    if min(n_preserved_sources, n_altered_sources) < 5:
        # One altered source can give a degenerate [1,1] bootstrap interval.
        return {**base, 'ci_low': np.nan, 'ci_high': np.nan, 'valid_bootstrap_draws': 0,
                'uncertainty_note': 'insufficient_sources_for_bootstrap_interval'}
    order = np.argsort(values, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(values[order])) + 1]
    draws = []
    # Multinomial counts are a cluster bootstrap: all rows of a resampled
    # source travel together, including both classes in the matched cohort.
    for begin in range(0, n_bootstrap, 100):
        counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)),
                                 size=min(100, n_bootstrap - begin))
        sampled = counts[:, groups[order]] * weights[order]
        pos = np.add.reduceat(sampled * labels[order], starts, axis=1)
        neg = np.add.reduceat(sampled * (1 - labels[order]), starts, axis=1)
        numerator = (pos * (np.cumsum(neg, axis=1) - .5 * neg)).sum(axis=1)
        denominator = pos.sum(axis=1) * neg.sum(axis=1)
        draws.extend((numerator[denominator > 0] / denominator[denominator > 0]).tolist())
    low, high = np.quantile(draws, [.025, .975]) if len(draws) >= 2 else (np.nan, np.nan)
    return {**base, 'ci_low': low, 'ci_high': high, 'valid_bootstrap_draws': len(draws),
            'uncertainty_note': 'small_sample_exploratory' if min(n_preserved_sources, n_altered_sources) < 10 else ''}


def _cohorts(frame):
    positive = frame.is_altered == 0
    matched = frame.dataset == 'opai_matched_negative'
    linked = set(frame.loc[matched, 'matched_positive_id'].dropna())
    cohorts = {}
    if matched.any() and linked:
        cohorts['opai_matched_primary'] = frame[matched | (positive & frame.id.isin(linked))]
        subtle = matched & frame.perturbation.isin(MATCHED_SUBTLE)
        linked_subtle = set(frame.loc[subtle, 'matched_positive_id'].dropna())
        if subtle.any() and linked_subtle:
            cohorts['opai_matched_subtle_secondary'] = frame[subtle | (positive & frame.id.isin(linked_subtle))]
    cross = frame.dataset.isin(['controlled', 'controlled_subtle'])
    if cross.any() and positive.any():
        cohorts['mixed_corpus_secondary'] = frame[positive | cross]
    subtle = frame.dataset == 'controlled_subtle'
    if subtle.any() and positive.any():
        cohorts['subtle_cross_corpus_secondary'] = frame[positive | subtle]
    if 'controlled_test_validity' in frame:
        reviewed = frame.controlled_test_validity.notna() & (frame.is_altered == 1)
        for name, accepted in [('reviewed_valid_secondary', ['valid']),
                               ('reviewed_valid_qualified_secondary', ['valid', 'qualified'])]:
            selected = reviewed & frame.controlled_test_validity.isin(accepted)
            if selected.any() and positive.any():
                cohorts[name] = frame[positive | selected]
    return cohorts


def _threshold(frame, score, target_fpr=.1):
    score = score.reindex(frame.index)
    part = frame[(frame.is_altered == 0) & np.isfinite(score)].copy()
    if part.empty:
        return np.nan
    values = score.loc[part.index].to_numpy()
    weights = _source_weights(part)
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order]) / weights.sum()
    # Strict greater-than at this observed quantile is conservative with ties.
    return float(values[order[min(np.searchsorted(cumulative, 1 - target_fpr), len(values) - 1)]])


def _operating_point(frame, score, threshold):
    score = score.reindex(frame.index)
    part = frame.loc[np.isfinite(score)].copy()
    alerts = score.loc[part.index].to_numpy() > threshold
    labels = part.is_altered.to_numpy(dtype=int)
    weights = _source_weights(part)
    def rate(mask):
        return float(np.average(alerts[mask], weights=weights[mask])) if mask.any() else np.nan
    fpr, sensitivity = rate(labels == 0), rate(labels == 1)
    return {'false_positive_rate': fpr, 'sensitivity': sensitivity,
            'balanced_accuracy': (1 - fpr + sensitivity) / 2,
            'n_preserved': int((labels == 0).sum()), 'n_altered': int(labels.sum()),
            'n_sources': part.source_group.nunique()}


def analyze_semantic_extension(pairs, scores, output, *, seed=42, n_bootstrap=1000, target_fpr=.1, review=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(pairs, seed)
    if scores.id.duplicated().any():
        raise ValueError('Repeated score IDs; refusing an ambiguous score merge')
    if not set(metadata.id).issubset(set(scores.id)):
        missing = sorted(set(metadata.id) - set(scores.id))
        raise ValueError(f'Missing semantic scores for {len(missing)} eligible pairs; first ID {missing[0]}')
    metric_columns = [name for name in scores if name not in metadata.columns or name == 'id']
    frame = metadata.merge(scores[metric_columns], on='id', validate='one_to_one')
    if review is not None:
        if review.id.duplicated().any():
            raise ValueError('Repeated pair IDs in direct textual review')
        frame = frame.merge(review[['id', 'controlled_test_validity']], on='id', how='left', validate='one_to_one')
    risk, definitions = _metric_scores(frame)
    definitions.to_csv(output / 'metric_definitions.csv', index=False)
    # Export labels, grouping and split, without benchmark texts.
    metadata.to_csv(output / 'evaluation_pairs.csv', index=False)
    split_overlap = set(frame.loc[frame.evaluation_split == 'calibration', 'source_group']) & set(
        frame.loc[frame.evaluation_split == 'test', 'source_group'])
    if split_overlap:
        raise ValueError('Calibration/test source leakage')
    rng = np.random.default_rng(seed)
    cohorts = _cohorts(frame)
    if not cohorts:
        raise ValueError('Both positive controls and intended alterations are required')
    auc_rows, strata_rows, threshold_rows, operating_strata, selections = [], [], [], [], []
    for cohort, part in cohorts.items():
        for split in ['all', 'calibration', 'test']:
            selected = part if split == 'all' else part[part.evaluation_split == split]
            for metric in risk:
                result = source_bootstrap_auc(selected, risk.loc[selected.index, metric], rng, n_bootstrap)
                if result:
                    auc_rows.append({'cohort': cohort, 'split': split, 'metric': metric, **result})
        for split in ['all', 'test']:
            base = part if split == 'all' else part[part.evaluation_split == split]
            for negative_type, negative in base[base.is_altered == 1].groupby('perturbation'):
                if cohort.startswith('opai_matched_'):
                    positives = base[base.id.isin(set(negative.matched_positive_id))]
                else:
                    positives = base[base.is_altered == 0]
                subset = pd.concat([positives, negative])
                for metric in risk:
                    result = source_bootstrap_auc(subset, risk.loc[subset.index, metric], rng, n_bootstrap)
                    if result:
                        strata_rows.append({'cohort': cohort, 'split': split, 'stratum': 'negative_type',
                                            'value': negative_type, 'metric': metric, **result})
            for field in ['domain', 'generator']:
                for value, subset in base.groupby(field):
                    for metric in risk:
                        result = source_bootstrap_auc(subset, risk.loc[subset.index, metric], rng, n_bootstrap)
                        if result:
                            strata_rows.append({'cohort': cohort, 'split': split, 'stratum': field,
                                                'value': value, 'metric': metric, **result})
            for operation, positives in base[base.is_altered == 0].groupby('perturbation'):
                if cohort.startswith('opai_matched_'):
                    negatives = base[base.matched_positive_id.isin(set(positives.id))]
                else:
                    negatives = base[base.is_altered == 1]
                subset = pd.concat([positives, negatives])
                for metric in risk:
                    result = source_bootstrap_auc(subset, risk.loc[subset.index, metric], rng, n_bootstrap)
                    if result:
                        strata_rows.append({'cohort': cohort, 'split': split, 'stratum': 'positive_operation',
                                            'value': operation, 'metric': metric, **result})
        calibration = part[part.evaluation_split == 'calibration']
        test = part[part.evaluation_split == 'test']
        for metric in risk:
            threshold = _threshold(calibration, risk[metric], target_fpr)
            if not np.isfinite(threshold):
                continue
            for split, subset in [('calibration', calibration), ('test', test)]:
                threshold_rows.append({'cohort': cohort, 'split': split, 'metric': metric,
                                       'threshold': threshold, 'alert_rule': 'alteration_risk > threshold',
                                       'calibration_target_fpr': target_fpr,
                                       **_operating_point(subset, risk[metric], threshold)})
            for (label, operation), subset in test.groupby(['is_altered', 'perturbation']):
                rates = _operating_point(subset, risk[metric], threshold)
                operating_strata.append({'cohort': cohort, 'split': 'test', 'metric': metric,
                                         'class': 'altered' if label else 'presumed_preserved',
                                         'operation': operation, 'threshold': threshold, **rates})
        candidates = [row for row in auc_rows if row['cohort'] == cohort and row['split'] == 'calibration'
                      and row['metric'].startswith('nli_bidirectional_')]
        if candidates and not cohort.startswith('reviewed_'):
            priority = {'nli_bidirectional_' + name: i for i, name in enumerate(AGGREGATORS)}
            best = sorted(candidates, key=lambda row: (-row['auc'], priority.get(row['metric'], 99)))[0]
            heldout = next((row for row in auc_rows if row['cohort'] == cohort and row['split'] == 'test'
                            and row['metric'] == best['metric']), None)
            operating = next((row for row in threshold_rows if row['cohort'] == cohort and row['split'] == 'test'
                              and row['metric'] == best['metric']), None)
            selections.append({'cohort': cohort, 'selection_rule': 'highest calibration AUC among four bidirectional NLI aggregators',
                               'selected_metric': best['metric'], 'calibration': best,
                               'held_out_test': heldout, 'held_out_operating_point': operating})
    auc_table = pd.DataFrame(auc_rows)
    auc_table.to_csv(output / 'metric_auc.csv', index=False)
    pd.DataFrame(strata_rows).to_csv(output / 'auc_by_stratum.csv', index=False)
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(output / 'false_positive_rates.csv', index=False)
    pd.DataFrame(operating_strata).to_csv(output / 'operating_points_by_operation.csv', index=False)
    primary = 'opai_matched_primary' if 'opai_matched_primary' in cohorts else 'mixed_corpus_secondary'
    # This population includes every presumed-positive OpAI rewrite, including
    # those with no applicable matched corruption. Do not retune thresholds.
    all_positive_fpr_rows = []
    all_positive = frame[(frame.dataset == 'opai_positive') & (frame.is_altered == 0)]
    primary_thresholds = thresholds[(thresholds.cohort == primary) & (thresholds.split == 'calibration')]
    for _, threshold_row in primary_thresholds.iterrows():
        metric, threshold = threshold_row.metric, float(threshold_row.threshold)
        for split in ['calibration', 'test']:
            controls = all_positive[all_positive.evaluation_split == split]
            populations = [('all', controls)] + list(controls.groupby('perturbation'))
            for operation, part in populations:
                if part.empty:
                    continue
                valid = part.loc[np.isfinite(risk.loc[part.index, metric])]
                if valid.empty:
                    continue
                alerts = risk.loc[valid.index, metric].to_numpy() > threshold
                all_positive_fpr_rows.append({'threshold_calibration_cohort': primary,
                                              'split': split, 'operation': operation, 'metric': metric,
                                              'threshold': threshold, 'alert_rule': 'alteration_risk > threshold',
                                              'false_positive_rate': float(np.average(alerts, weights=_source_weights(valid))),
                                              'unweighted_false_positive_rate': float(alerts.mean()),
                                              'n_alerts': int(alerts.sum()), 'n_presumed_preserved': len(valid),
                                              'n_sources': valid.source_group.nunique(),
                                              'labels_are_presumed_not_human_validated': True})
    pd.DataFrame(all_positive_fpr_rows).to_csv(output / 'positive_controls_all_fpr.csv', index=False)
    primary_selection = next((item for item in selections if item['cohort'] == primary), None)
    transfer_rows = []
    if primary_selection and primary_selection['held_out_operating_point']:
        chosen_metric = primary_selection['selected_metric']
        chosen_threshold = primary_selection['held_out_operating_point']['threshold']
        for target_cohort, part in cohorts.items():
            target = part[part.evaluation_split == 'test']
            auc_match = auc_table[(auc_table.cohort == target_cohort) & (auc_table.split == 'test') &
                                  (auc_table.metric == chosen_metric)]
            if not len(auc_match):
                continue
            auc = auc_match.iloc[0]
            transfer_rows.append({'selection_cohort': primary, 'test_cohort': target_cohort,
                                  'selected_metric': chosen_metric, 'threshold': chosen_threshold,
                                  'auc': auc.auc, 'ci_low': auc.ci_low, 'ci_high': auc.ci_high,
                                  **_operating_point(target, risk[chosen_metric], chosen_threshold)})
    pd.DataFrame(transfer_rows).to_csv(output / 'selected_aggregator_transfer.csv', index=False)
    selection_manifest = {'positive_class_in_auc': 'intended semantic alteration', 'score_orientation': 'higher risk = more altered; fixed before evaluation',
                          'split_unit': 'original source; editors and all operations remain together',
                          'source_overlap_calibration_test': len(split_overlap), 'primary_cohort': primary,
                          'n_bootstrap': n_bootstrap, 'seed': seed, 'target_calibration_false_positive_rate': target_fpr,
                          'selections': selections, 'selected_primary_transfer': transfer_rows,
                          'labels_are_semantic_proxy_not_gold': True,
                          'direct_textual_review_n_pairs': len(review) if review is not None else 0,
                          'review_is_assistant_textual_assessment_not_human_gold': review is not None,
                          'all_positive_controls_fpr_file': 'positive_controls_all_fpr.csv',
                          'all_positive_controls_n_pairs': len(all_positive)}
    (output / 'aggregator_selection.json').write_text(json.dumps(selection_manifest, indent=2) + '\n')
    _correlations(frame, risk, output)
    _report(frame, auc_table, thresholds, definitions, selection_manifest, output)
    return selection_manifest


def _correlations(frame, risk, output):
    names = [name for name in risk if name in CORE_METRICS or name.startswith('bert_chunk_f1')
             or name.startswith('nli_bidirectional_')]
    # Show preservation orientation for all columns (high = more preserved).
    values = -risk[names]
    values.corr(method='spearman').to_csv(output / 'metric_spearman.csv')
    corr = values.corr(method='spearman')
    labels = {'lexical_rouge_l_f1': 'ROUGE-L', 'lexical_bleu': 'BLEU', 'embedding_cosine': 'Coseno',
              'bert_chunk_f1': 'BERTScore chunk'}
    display = [labels.get(name, name.replace('nli_bidirectional_', 'NLI↔ ').replace('bert_chunk_f1_', 'BERT ')) for name in names]
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(names)), display, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(len(names)), display, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            value = corr.iloc[i, j]
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(value) > .6 else 'black')
    ax.set_title('Correlazione Spearman fra metriche\nControlli positivi e alterazioni; identità escluse')
    fig.colorbar(image, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.savefig(output / 'metric_spearman.png', dpi=170)
    fig.savefig(output / 'metric_spearman.pdf')
    plt.close(fig)
    primary = _cohorts(frame).get('opai_matched_primary')
    if primary is not None:
        (-risk.loc[primary.index, names]).corr(method='spearman').to_csv(output / 'metric_spearman_opai_matched.csv')


def _report(frame, auc_table, thresholds, definitions, selection, output):
    primary = selection['primary_cohort']
    main = auc_table[(auc_table.cohort == primary) & (auc_table.split == 'test')]
    selected = next((item for item in selection['selections'] if item['cohort'] == primary), None)
    count_table = frame.groupby(['dataset', 'perturbation', 'evaluation_split']).agg(
        n_pairs=('id', 'size'), n_sources=('source_group', 'nunique')).reset_index()
    operating = thresholds[(thresholds.cohort == primary) & (thresholds.split == 'test')]
    reviewed = auc_table[(auc_table.cohort.str.startswith('reviewed_')) & (auc_table.split == 'all') &
                         auc_table.metric.isin(CORE_METRICS + ['nli_bidirectional_' + name for name in AGGREGATORS])]
    subtle = auc_table[(auc_table.cohort == 'opai_matched_subtle_secondary') & (auc_table.split == 'test') &
                       auc_table.metric.isin(CORE_METRICS + ['nli_bidirectional_' + name for name in AGGREGATORS])]
    subtle_selection = next((item for item in selection['selections'] if item['cohort'] == 'opai_matched_subtle_secondary'), None)
    matched_negatives = frame[frame.dataset == 'opai_matched_negative']
    easy_count = int(matched_negatives.perturbation.isin(['matched_negation', 'matched_entity_substitution']).sum())
    heldout_counts = Counter(matched_negatives.loc[matched_negatives.evaluation_split == 'test', 'perturbation'])
    all_positive_fpr = pd.read_csv(output / 'positive_controls_all_fpr.csv')
    all_positive_test = all_positive_fpr[(all_positive_fpr.split == 'test') & (all_positive_fpr.operation == 'all')]
    linked_positive_n = int((_cohorts(frame)[primary].is_altered == 0).sum())
    selection_text = ('Nessun aggregatore selezionabile: mancano entrambe le classi in calibration.' if selected is None else
                      f"Aggregatore scelto usando soltanto calibration: **{selected['selected_metric']}**. "
                      'Le prestazioni test riportate sotto non partecipano alla scelta. La scelta resta esplorativa perché le etichette sono proxy.')
    lines = ['# Controlli positivi e scelta dell’aggregatore semantico', '',
             'Il confronto valuta la separazione fra modifiche presumibilmente conservative (Polish, Paraphrase, Style Rewrite OpAI) '
             'e alterazioni intenzionali. Le etichette operative e le regole **non sono gold label umane**. '
             'Identità, fixture e shuffle sono esclusi dall’AUC principale; lo shuffle può cambiare coerenza senza cambiare le proposizioni.', '',
             '## Disegno e campione', '', markdown_table(count_table), '',
             'L’unità di split e bootstrap è il documento sorgente: editor, operazioni e versioni della stessa fonte viaggiano insieme. '
             'Ogni fonte riceve uguale peso entro ciascuna classe. Il test è separato dalla calibrazione; nessuna inversione del segno viene fatta dopo aver osservato l’AUC.', '',
             ('Il confronto principale **opai_matched_primary** usa gli stessi originali e le stesse trasformazioni OpAI: '
              'un negativo è la versione positiva ulteriormente corrotta, e conserva lo stesso originale del positivo. '
              'Si includono soltanto positivi collegati ad almeno un negativo applicabile. Questo riduce il confondente del corpus, '
              'ma la separazione può ancora dipendere dalle perturbazioni artificiali.' if primary == 'opai_matched_primary' else
              '**Limite principale:** positivi OpAI e negativi MGTBench provengono da corpora diversi; un’AUC elevata può riflettere corpus, dominio, lunghezza o editor. '
              'Questa AUC non basta a stimare gli errori sugli humanizer.'), '',
             'Le analisi mixed_corpus_secondary e subtle_cross_corpus_secondary conservano il confondente OpAI/MGTBench e sono diagnostiche. '
             'I domini senza entrambe le classi non producono un’AUC di dominio. '
             'L’AUC aggregata dipende dalla composizione osservata dei tipi di alterazione; non stima una prevalenza reale degli errori degli humanizer. '
             'Le AUC stratificate evitano che un buon risultato sulle negazioni nasconda errori su numeri, quantificatori o cautele.', '',
             (f'Nel campione OpAI appaiato, **{easy_count}/{len(matched_negatives)} negativi** sono negazioni o sostituzioni di entità. '
              'La metrica principale assegna uguale peso alle fonti entro ciascuna classe, ma rimane dipendente dalla composizione degli errori. '
              'Il confronto separato dei soli sottili sotto riduce questa dominanza.' if len(matched_negatives) else ''), '',
             '## AUC sul test separato', '',
             'Classe 1 = alterazione intenzionale; un valore più alto dello score di rischio significa maggiore alterazione. '
             'AUC 0,5 indica ranking casuale; valori inferiori non sono ribaltati. IC 95% bootstrap percentile per fonte. '
             'Nei campioni appaiati finiti il ricampionamento può spostare gli IC, soprattutto per AUC deboli vicino a 0,5, '
             'e un IC percentile può non contenere la stima puntuale. Non usare questi intervalli per dichiarare '
             'ROUGE/BERTScore superiori al caso o piccole differenze fra aggregatori come dimostrate.', '',
             markdown_table(main[['metric', 'n_pairs', 'n_sources', 'auc', 'ci_low', 'ci_high']]), '',
             selection_text, '',
             'Il confronto di tutti gli aggregatori sul test è descrittivo; il valore del solo aggregatore scelto in calibration è la valutazione successiva alla scelta. '
             'La conservazione richiede entrambe le direzioni: per mean/q10/min si prende il supporto minore tra forward e reverse; '
             'per la quota non supportata si prende la maggiore.', '',
             '## Controllo difficile entro fonte: sole alterazioni sottili OpAI', '',
             ('Il cohort opai_matched_subtle_secondary contiene cambi numerici, rafforzamenti di quantificatori, '
              'rimozioni di cautele e aggiunte di dettagli non supportati, insieme agli **esatti positivi collegati**. '
              'Usa gli split per fonte già fissati e non introduce confronti OpAI/MGTBench. La sua selezione in calibration è esplorativa, '
              'non una ricerca di un aggregatore universale.' if len(subtle) else 'Nessun confronto appaiato sottile disponibile.'), '',
             markdown_table(subtle[['metric', 'n_pairs', 'n_sources', 'auc', 'ci_low', 'ci_high']]) if len(subtle) else '', '',
             (f"Aggregatore scelto soltanto sulla calibrazione dei sottili: **{subtle_selection['selected_metric']}**. "
              'La scelta e la soglia principali vengono inoltre trasferite a questo test senza riottimizzazione nella sezione successiva.'
              if subtle_selection else ''), '',
             (f"Nel test ci sono soltanto **{heldout_counts.get('matched_unsupported_detail', 0)} aggiunte di dettaglio** e "
              f"**{heldout_counts.get('matched_hedge_removal', 0)} rimozioni di cautela**. "
              'Un’AUC su un solo caso non è conclusiva; gli IC bootstrap sono omessi quando una classe ha meno di cinque fonti. '
              'Fra cinque e nove fonti i risultati sono esplicitamente piccoli campioni esplorativi. '
              '`auc_by_stratum.csv` conserva i conteggi di coppie e fonti per ciascuna classe e la nota di incertezza.'
              if len(subtle) else ''), '',
             '## Falsi allarmi sulle buone riscritture', '',
             f'Per ciascuna metrica la soglia è fissata sui soli positivi di calibration, mirando a FPR ≤ {selection["target_calibration_false_positive_rate"]:.0%}. '
             'Si segnala alterazione solo quando il rischio supera strettamente la soglia; i pareggi sono conservativi. '
             'Sul test si misurano FPR (riscritture presunte buone segnalate) e sensibilità alle alterazioni. Non sono tassi su parafrasi certificate da annotatori.', '',
             markdown_table(operating[['metric', 'threshold', 'false_positive_rate', 'sensitivity', 'balanced_accuracy']]), '',
             'I falsi allarmi per Polish, Paraphrase e Style separatamente e la sensibilità per tipo di alterazione '
             'sono in `operating_points_by_operation.csv`, sempre con soglia fissata dalla calibrazione globale.', '',
             '### Allarmi su tutti i controlli positivi OpAI', '',
             f'Il confronto AUC principale usa {linked_positive_n} positivi collegati a una corruzione applicabile. '
             f'Per misurare gli allarmi sull’intera popolazione positiva, `positive_controls_all_fpr.csv` include tutti '
             f'i {selection["all_positive_controls_n_pairs"]} positivi OpAI, anche quelli senza negativo appaiato: '
             'calibration e test, nel complesso e separatamente per Polish, Paraphrase e Style. '
             'La soglia è **quella già fissata sulla calibrazione principale** e non viene ricalibrata su questa popolazione più ampia. '
             f'La FPR di calibrazione sull’intera popolazione non è quindi forzata al {selection["target_calibration_false_positive_rate"]:.0%}; il test resta separato. '
             'Sono allarmi su riscritture presumibilmente conservative, non falsi allarmi certificati da annotazione umana.', '',
             markdown_table(all_positive_test[['metric', 'n_presumed_preserved', 'n_sources', 'false_positive_rate', 'n_alerts']]), '',
             '## Trasferimento della scelta primaria', '',
             'La stessa metrica e la stessa soglia scelte in calibration sul confronto principale vengono applicate anche '
             'ai test secondari, senza riottimizzazione. I confronti MGTBench/OpAI restano confusi dal corpus e servono come diagnostica.', '',
             markdown_table(pd.DataFrame(selection['selected_primary_transfer'])[
                 ['test_cohort', 'selected_metric', 'auc', 'ci_low', 'ci_high', 'false_positive_rate', 'sensitivity']])
             if selection['selected_primary_transfer'] else 'Nessun trasferimento valutabile.', '',
             '## Sensibilità alla revisione diretta delle perturbazioni', '',
             ('Sono disponibili 40 coppie lette direttamente dall’assistente senza usare i punteggi individuali, '
              '**non un’annotazione umana né un gold indipendente**. L’analisi principale conserva tutti i casi intenzionalmente alterati. '
              'Le tabelle secondarie sotto limitano soltanto i negativi al sottocampione revisionato: prima valid, poi valid+qualified. '
              'I positivi restano i controlli OpAI, quindi permane il confondente fra corpora; il piccolo campione non consente '
              'di correggere o certificare le altre perturbazioni. Non viene scelto un nuovo aggregatore su questi sottocampioni.'
              if len(reviewed) else 'Nessuna revisione testuale con validità esplicita disponibile per questa esecuzione.'), '',
             markdown_table(reviewed[['cohort', 'metric', 'n_altered', 'n_preserved', 'auc', 'ci_low', 'ci_high']]) if len(reviewed) else '', '',
             ('Un piccolo audit testuale dei positivi è documentato separatamente in [positive_review9.md](positive_review9.md). '
              'Anche questo è un giudizio dell’assistente, non annotazione umana. Alcune riscritture possono modificare specificità o agency: '
              'la FPR resta quindi un tasso di allarmi sui positivi presunti, non un tasso su parafrasi certificate.'
              if (output / 'positive_review9.md').is_file() else ''), '',
             '## Correlazioni e lettura delle metriche', '',
             '![Correlazioni Spearman](metric_spearman.png)', '',
             'La correlazione è descrittiva, non source-weighted, ed è calcolata sulle coppie valutate senza identità ripetute; '
             'può essere guidata dalle differenze fra condizioni. La matrice del solo confronto OpAI appaiato è disponibile separatamente. '
             'BLEU resta nel report per confrontabilità; se ridondante con ROUGE-L e poco discriminante, può diventare una misura secondaria.', '',
             'Il rescaling BERTScore è una trasformazione affine crescente per ciascuno score con baseline fissa: ne cambia la scala '
             'e la distanza dalla baseline, **non il ranking e quindi non l’AUC**. Per l’approssimazione documentale a chunk '
             'la lettura resta distinta dal BERTScore standard su una frase.', '',
             'L’NLI è supporto aggregato, non una prova logica. Forward basso è compatibile con aggiunte non supportate o modifiche; '
             'reverse basso è compatibile con omissioni o modifiche: nessuna delle due direzioni identifica da sola la causa. '
             'Il minimo può essere sensibile a una sola frase difficile; q10 e media possono diluire errori isolati.', '',
             'File: `metric_auc.csv`, `auc_by_stratum.csv`, `false_positive_rates.csv`, `aggregator_selection.json`, '
             '`evaluation_pairs.csv`, `metric_definitions.csv`, `metric_spearman.csv`, `operating_points_by_operation.csv`, '
             '`selected_aggregator_transfer.csv`, `positive_controls_all_fpr.csv`. '
             'I testi restano nella directory dati ignorata.', '']
    (output / 'aggregator_report.md').write_text('\n'.join(lines))
    if len(main):
        shown = main[main.metric.isin(CORE_METRICS + ['nli_bidirectional_' + name for name in AGGREGATORS])]
        fig, ax = plt.subplots(figsize=(9, 5))
        y = np.arange(len(shown))
        ax.axvline(.5, color='gray', linestyle='--', linewidth=.8)
        ax.errorbar(shown.auc, y, xerr=[np.maximum(shown.auc - shown.ci_low, 0), np.maximum(shown.ci_high - shown.auc, 0)],
                    fmt='o', capsize=3, color='#167c80')
        ax.set_yticks(y, shown.metric, fontsize=8)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel('AUC: alterazione intenzionale vs riscrittura presunta conservativa\nIC 95% bootstrap per documento sorgente')
        ax.set_title('Test separato: ' + primary)
        fig.tight_layout()
        fig.savefig(output / 'aggregator_auc.png', dpi=170)
        fig.savefig(output / 'aggregator_auc.pdf')
        plt.close(fig)

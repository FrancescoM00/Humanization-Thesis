"""Aggregazione equa, errori distribuiti e diagnostica della lunghezza su fonti congelate.

Tutti gli score primari sono continui. La calibrazione serve solo a stimare
percentili di riferimento opzionali condizionati alla lunghezza; non viene
mai usata per invertire un'AUC.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

from humanization.evaluation.semantic import (
    _metadata as old_metadata, _cohorts as old_cohorts, _group_key,
    source_bootstrap_auc, _source_weights,
)
from humanization.semantics.metrics import _sentences
from humanization.evaluation.reporting import markdown_table

AGGREGATORS = ('mean', 'q10', 'min')
FAMILIES = ('nli', 'aligned_cosine', 'aligned_bert_f1_rescaled',
            'aligned_bert_top3_f1_rescaled')
DISPLAY = {'nli': 'NLI', 'aligned_cosine': 'Coseno allineato',
           'aligned_bert_f1_rescaled': 'BERTScore allineato',
           'aligned_bert_top3_f1_rescaled': 'BERTScore sugli stessi top3 NLI'}


def merge_score_tables(paths):
    """Unisce righe con ID disgiunti o coincidenti senza sostituire gli score implicitamente."""
    output = None
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        incoming = pd.read_csv(path)
        if incoming.id.duplicated().any():
            raise ValueError(f'Duplicate scores in {path}')
        incoming = incoming.set_index('id')
        if output is None:
            output = incoming
            continue
        common_ids = output.index.intersection(incoming.index)
        for column in output.columns.intersection(incoming.columns):
            a, b = output.loc[common_ids, column], incoming.loc[common_ids, column]
            valid = a.notna() & b.notna()
            if not valid.any():
                continue
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                same = np.allclose(a[valid], b[valid], atol=2e-6, rtol=1e-6)
            else:
                same = (a[valid].astype(str).to_numpy() == b[valid].astype(str).to_numpy()).all()
            if not same:
                raise ValueError(f'Conflicting score column {column!r} in {path}')
        output = output.combine_first(incoming)
    if output is None:
        raise ValueError('At least one score table is required')
    return output.reset_index()


def pair_metadata(pairs, seed=42):
    base = old_metadata(pairs, seed)
    additions = []
    for pair in pairs:
        if pair.get('dataset') not in {'opai_distributed_negative', 'opai_distributed_positive'}:
            continue
        meta = pair.get('metadata', {})
        split = pair.get('evaluation_split') or meta.get('evaluation_split')
        if split not in {'calibration', 'test'}:
            raise ValueError('Distributed pairs must inherit a frozen calibration/test split')
        additions.append({'id': pair['id'], 'source_group': _group_key(pair),
                          'source_id': pair['source_id'], 'dataset': pair['dataset'],
                          'perturbation': pair['perturbation'], 'domain': pair['domain'],
                          'generator': pair['generator'], 'is_altered': int(pair['dataset']=='opai_distributed_negative'),
                          'positive_operation': meta.get('base_operation', meta.get('operation', '')),
                          'matched_positive_id': meta.get('matched_positive_id'),
                          'original_words': len(pair['original'].split()),
                          'length_ratio': len(pair['modified'].split()) / len(pair['original'].split()),
                          'evaluation_split': split})
    frame = pd.concat([base, pd.DataFrame(additions)], ignore_index=True)
    if frame.id.duplicated().any():
        raise ValueError('Duplicated pair IDs in the combined experiment')
    by_id = {pair['id']: pair for pair in pairs}
    sentence_counts = {}
    for column, text_key in [('original_sentences', 'original'), ('modified_sentences', 'modified')]:
        values = []
        for row_id in frame.id:
            text = by_id[row_id][text_key]
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest not in sentence_counts:
                sentence_counts[digest] = len(_sentences(text))
            values.append(sentence_counts[digest])
        frame[column] = values
    frame['max_sentences'] = frame[['original_sentences', 'modified_sentences']].max(axis=1)
    for key in ['corruption_family', 'n_corrupted_sentences', 'maximum_applicable_sentences', 'common_panel', 'panel', 'experiment_cohort']:
        frame[key] = frame.id.map(lambda value: by_id[value].get('metadata', {}).get(key))
    distributed = frame.dataset.isin(['opai_distributed_negative','opai_distributed_positive'])
    frame.loc[distributed & frame.experiment_cohort.isna(),'experiment_cohort'] = 'initial_source_panel'
    expanded = frame[distributed & (frame.experiment_cohort=='expanded_source_panel')]
    legacy_groups = set(frame.loc[frame.dataset=='opai_positive','source_group'])
    if set(expanded.source_group) & legacy_groups:
        raise ValueError('The expanded source panel overlaps old OpAI positive sources')
    for _, part in frame.groupby('source_group'):
        if part.evaluation_split.nunique() != 1:
            raise ValueError('An original source crosses calibration and test')
    return frame


def metric_quality(frame):
    quality = pd.DataFrame(index=frame.index)
    definitions = []
    for family in FAMILIES:
        for aggregator in AGGREGATORS:
            forward, reverse = [f'{family}_{direction}_{aggregator}' for direction in ['forward', 'reverse']]
            if forward not in frame or reverse not in frame:
                continue
            for column in [forward, reverse]:
                quality[column] = pd.to_numeric(frame[column], errors='coerce')
                definitions.append({'metric': column, 'family': family, 'aggregator': aggregator,
                                    'direction': 'forward' if column == forward else 'reverse',
                                    'higher_means': 'more_preserved'})
            combined = f'{family}_bidirectional_{aggregator}'
            quality[combined] = frame[[forward, reverse]].min(axis=1, skipna=False)
            definitions.append({'metric': combined, 'family': family, 'aggregator': aggregator,
                                'direction': 'bidirectional_worst', 'higher_means': 'more_preserved'})
    if {'nli_forward_low_support_fraction', 'nli_reverse_low_support_fraction'}.issubset(frame):
        column = 'nli_bidirectional_supported_fraction'
        quality[column] = 1 - frame[['nli_forward_low_support_fraction', 'nli_reverse_low_support_fraction']].max(axis=1, skipna=False)
        definitions.append({'metric': column, 'family': 'nli', 'aggregator': 'supported_fraction',
                            'direction': 'bidirectional_worst', 'higher_means': 'more_preserved'})
    for column in ['embedding_cosine', 'bert_chunk_f1', 'lexical_rouge_l_f1', 'lexical_bleu']:
        if column in frame:
            quality[column] = frame[column]
            definitions.append({'metric': column, 'family': 'document_baseline', 'aggregator': 'original_document_score',
                                'direction': 'original_protocol', 'higher_means': 'more_preserved'})
    return quality, pd.DataFrame(definitions)


def _paired_subset(frame, negatives):
    positives = frame[(frame.is_altered == 0) & frame.id.isin(set(negatives.matched_positive_id))]
    return pd.concat([positives, negatives]).sort_index()


def analysis_cohorts(frame):
    legacy = frame[~frame.dataset.isin(['opai_distributed_negative','opai_distributed_positive'])]
    cohorts = {key: value for key, value in old_cohorts(legacy).items()
               if key in {'opai_matched_primary', 'opai_matched_subtle_secondary'}}
    distributed = frame[frame.dataset == 'opai_distributed_negative']
    for (family, experiment), family_part in distributed.groupby(['corruption_family','experiment_cohort']):
        for panel, part in family_part.groupby('panel'):
            for k, negative in part.groupby('n_corrupted_sentences'):
                name = f'distributed:{family}:{experiment}:{panel}:k{int(k)}'
                cohorts[name] = _paired_subset(frame, negative)
    return cohorts


def auc_table(cohorts, quality, rng, n_bootstrap, metrics=None):
    rows = []
    names = list(quality) if metrics is None else metrics
    for cohort, part in cohorts.items():
        for split in ['all_descriptive', 'calibration', 'test']:
            selected = part if split == 'all_descriptive' else part[part.evaluation_split == split]
            for metric in names:
                result = source_bootstrap_auc(selected, -quality[metric], rng, n_bootstrap)
                if result:
                    rows.append({'cohort': cohort, 'split': split, 'metric': metric, **result})
    return pd.DataFrame(rows)


def _rho_point(a, b):
    if len(a) < 3 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return np.nan
    return float(spearmanr(a, b).statistic)


def _source_rho(part, metric, length, rng, n_bootstrap):
    # Ogni fonte originale ha un voto dopo la media su editor e operazioni.
    values = part[['source_group', metric, length]].dropna().groupby('source_group')[[metric, length]].mean()
    point = _rho_point(values[length].to_numpy(), values[metric].to_numpy())
    draws = []
    if len(values) >= 10 and np.isfinite(point):
        raw = values[[length, metric]].to_numpy()
        for _ in range(n_bootstrap):
            sampled = raw[rng.integers(len(raw), size=len(raw))]
            rho = _rho_point(sampled[:, 0], sampled[:, 1])
            if np.isfinite(rho):
                draws.append(rho)
    low, high = np.quantile(draws, [.025, .975]) if len(draws) >= 2 else (np.nan, np.nan)
    return {'rho': point, 'ci_low': low, 'ci_high': high,
            'n_pairs': len(part), 'n_sources': len(values), 'valid_bootstrap_draws': len(draws)}


def length_correlations(frame, quality, output, rng, n_bootstrap):
    positives = frame[frame.dataset == 'opai_positive'].copy()
    for name in quality:
        positives[name] = quality.loc[positives.index, name]
    columns = [f'{family}_bidirectional_{aggregator}' for family in FAMILIES[:3]
               for aggregator in AGGREGATORS if f'{family}_bidirectional_{aggregator}' in positives]
    lengths = ['max_sentences']
    if {'nli_forward_n_hypothesis_chunks', 'nli_reverse_n_hypothesis_chunks'}.issubset(positives):
        positives['max_nli_units'] = positives[['nli_forward_n_hypothesis_chunks', 'nli_reverse_n_hypothesis_chunks']].max(axis=1)
        lengths.append('max_nli_units')
    rows, within = [], []
    for length in lengths:
        for metric in columns:
            result = _source_rho(positives, metric, length, rng, n_bootstrap)
            rows.append({'stratum': 'all', 'value': 'all', 'metric': metric, 'length': length,
                         'estimand': 'between_source_spearman_after_source_average', **result})
            for field in ['domain', 'generator', 'perturbation']:
                for value, part in positives.groupby(field):
                    result = _source_rho(part, metric, length, rng, min(n_bootstrap, 500))
                    rows.append({'stratum': field, 'value': value, 'metric': metric, 'length': length,
                                 'estimand': 'between_source_spearman_after_source_average', **result})
            valid = positives[['source_group', length, metric]].dropna().copy()
            a, b = rankdata(valid[length]), rankdata(valid[metric])
            valid['a'], valid['b'] = a, b
            valid[['a', 'b']] = valid[['a', 'b']] - valid.groupby('source_group')[['a', 'b']].transform('mean')
            weights = 1 / valid.groupby('source_group').source_group.transform('count').to_numpy()
            av, bv = valid.a.to_numpy(), valid.b.to_numpy()
            denominator = np.sqrt(np.sum(weights * av * av) * np.sum(weights * bv * bv))
            rho = float(np.sum(weights * av * bv) / denominator) if denominator else np.nan
            within.append({'metric': metric, 'length': length, 'rank_residual_correlation': rho,
                           'n_pairs': len(valid), 'n_sources': valid.source_group.nunique(),
                           'n_sources_with_length_variation': int((valid.groupby('source_group')[length].nunique() > 1).sum()),
                           'note': 'Within-source rank-centred diagnostic, not a causal length effect; no bootstrap CI claimed.'})
    table = pd.DataFrame(rows)
    table.to_csv(output/'length_correlations.csv', index=False)
    pd.DataFrame(within).to_csv(output/'length_within_source.csv', index=False)
    if 'nli_bidirectional_min' in positives:
        grouped = positives.groupby('source_group').agg(length=('max_sentences', 'mean'),
                                                       support=('nli_bidirectional_min', 'mean'), domain=('domain', 'first'))
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for domain, part in grouped.groupby('domain'):
            ax.scatter(part.length, part.support, label=domain, alpha=.7, s=25)
        ax.set_xlabel('Numero massimo di frasi fra originale e riscrittura')
        ax.set_ylabel('Minimo bidirezionale NLI medio della fonte')
        ax.set_title('Controlli positivi presunti: lunghezza e minimo NLI')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output/'length_positive_nli.png', dpi=170)
        fig.savefig(output/'length_positive_nli.pdf')
        plt.close(fig)
    return table


def _weighted_quantile(values, weights, quantiles):
    order = np.argsort(values, kind='stable')
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return np.array([values[min(np.searchsorted(cumulative, q), len(values)-1)] for q in quantiles])


def length_percentiles(frame, quality, output):
    """Stima i percentili di riferimento sui soli positivi di calibrazione, mai una soglia."""
    calibration = frame[(frame.dataset == 'opai_positive') & (frame.evaluation_split == 'calibration')]
    if calibration.empty:
        return pd.DataFrame(index=frame.index), None
    weights = _source_weights(calibration)
    cuts = np.unique(_weighted_quantile(calibration.max_sentences.to_numpy(), weights, [1/3, 2/3]))
    cuts = cuts[cuts < calibration.max_sentences.max()]
    bins = pd.Series(np.searchsorted(cuts, frame.max_sentences.to_numpy(), side='left'), index=frame.index)
    definitions = {'fit_population': 'OpAI presumed-positive calibration only', 'length_variable': 'max_sentences',
                   'cutpoints': cuts.tolist(), 'bin_rule': 'right-closed intervals: <= first cut, then <= next cut, then above',
                   'minimum_reference_sources': 5, 'minimum_reference_pairs': 15,
                   'quality_transform': 'source-weighted mid empirical CDF of support within the length bin',
                   'primary_output_is_continuous': True, 'threshold_fitted': False,
                   'use': 'Exploratory length-conditional reference percentile; not probability of correctness and not validated on DIPPER.',
                   'metrics': {}}
    adjusted = pd.DataFrame(index=frame.index)
    ref_rows = []
    metrics = [f'{family}_bidirectional_{agg}' for family in FAMILIES[:3]
               for agg in AGGREGATORS if f'{family}_bidirectional_{agg}' in quality]
    for metric in metrics:
        references = {}
        for bucket in sorted(bins.unique()):
            selected = calibration[(bins.loc[calibration.index] == bucket) & quality.loc[calibration.index, metric].notna()]
            fallback = selected.source_group.nunique() < 5 or len(selected) < 15
            if fallback:
                selected = calibration[quality.loc[calibration.index, metric].notna()]
            values = quality.loc[selected.index, metric].to_numpy()
            ws = _source_weights(selected)
            order = np.argsort(values, kind='stable')
            values, ws = values[order], ws[order]
            references[int(bucket)] = (values, np.r_[0, np.cumsum(ws)] / np.sum(ws))
            ref_rows.append({'metric': metric, 'length_bin': int(bucket), 'n_reference_pairs': len(selected),
                             'n_reference_sources': selected.source_group.nunique(), 'fallback_to_pooled': fallback,
                             'support_q05': _weighted_quantile(values, ws, [.05])[0],
                             'support_median': _weighted_quantile(values, ws, [.5])[0]})
        result = np.full(len(frame), np.nan)
        for position, (index, value) in enumerate(quality[metric].items()):
            if not np.isfinite(value):
                continue
            values, cumulative = references[int(bins.loc[index])]
            left, right = np.searchsorted(values, value, side='left'), np.searchsorted(values, value, side='right')
            result[position] = .5 * (cumulative[left] + cumulative[right])
        adjusted['length_percentile__'+metric] = result
    output_frame = frame[['id', 'source_group', 'evaluation_split', 'max_sentences']].copy()
    output_frame['length_bin'] = bins
    output_frame = pd.concat([output_frame, adjusted], axis=1)
    output_frame.to_csv(output/'length_adjusted_scores.csv', index=False)
    pd.DataFrame(ref_rows).to_csv(output/'length_calibration_reference.csv', index=False)
    (output/'length_calibration.json').write_text(json.dumps(definitions, indent=2)+'\n')
    return adjusted, bins


def distribution_table(frame, quality, output):
    metrics = [name for name in quality if '_bidirectional_' in name and not 'top3' in name]
    rows = []
    for keys, part in frame.groupby(['dataset', 'generator', 'perturbation', 'evaluation_split'], dropna=False):
        for metric in metrics:
            values = quality.loc[part.index, metric].dropna()
            if values.empty:
                continue
            quantiles = values.quantile([.05,.1,.25,.5,.75,.9,.95]).to_numpy()
            rows.append(dict(zip(['dataset','editor','operation','split'], keys), metric=metric,
                             n_pairs=len(values), n_sources=part.loc[values.index].source_group.nunique(),
                             mean=values.mean(), minimum=values.min(), maximum=values.max(),
                             **dict(zip(['q05','q10','q25','median','q75','q90','q95'],quantiles))))
    pd.DataFrame(rows).to_csv(output/'continuous_score_distributions.csv', index=False)


def analyze_fairness(pairs, scores, output, *, seed=42, n_bootstrap=1000, allow_partial=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta = pair_metadata(pairs, seed)
    if scores.id.duplicated().any():
        raise ValueError('Repeated score IDs')
    needed = ['id'] + [c for c in scores if c not in meta]
    frame = meta.merge(scores[needed], on='id', how='left', validate='one_to_one')
    quality, definitions = metric_quality(frame)
    if not len(quality.columns):
        raise ValueError('No semantic metrics supplied')
    required = [f'{family}_bidirectional_{aggregator}' for family in FAMILIES[:3] for aggregator in AGGREGATORS]
    if not allow_partial:
        absent = [name for name in required if name not in quality]
        if absent:
            raise ValueError('Required equally aggregated metrics missing: '+', '.join(absent))
        missing = quality[required].isna().any(axis=1)
        if missing.any():
            raise ValueError(f'Missing required scores for {int(missing.sum())} evaluation pairs; complete all stages or use --allow-partial only for a diagnostic run')
    definitions.to_csv(output/'metric_definitions.csv', index=False)
    frame[list(meta.columns)].to_csv(output/'evaluation_pairs.csv', index=False)
    rng = np.random.default_rng(seed)
    cohorts = analysis_cohorts(frame)
    auc = auc_table(cohorts, quality, rng, n_bootstrap)
    auc.to_csv(output/'fair_metric_auc.csv', index=False)
    main_metrics = [f'{family}_bidirectional_{agg}' for family in FAMILIES[:3] for agg in AGGREGATORS]
    main_metrics = [metric for metric in main_metrics if metric in quality]
    strata = []
    primary = cohorts.get('opai_matched_primary')
    if primary is not None:
        test = primary[primary.evaluation_split == 'test']
        for field in ['perturbation', 'domain', 'generator']:
            if field == 'perturbation':
                parts = [(name, _paired_subset(test, negative)) for name, negative in test[test.is_altered == 1].groupby(field)]
            else:
                parts = list(test.groupby(field))
            for value, part in parts:
                for metric in main_metrics:
                    result = source_bootstrap_auc(part, -quality[metric], rng, n_bootstrap)
                    if result:
                        strata.append({'stratum': field, 'value': value, 'metric': metric, **result})
    pd.DataFrame(strata).to_csv(output/'fair_auc_by_stratum.csv', index=False)
    length = length_correlations(frame, quality, output, rng, n_bootstrap)
    adjusted, bins = length_percentiles(frame, quality, output)
    adjusted_auc = auc_table(cohorts, adjusted, rng, n_bootstrap) if len(adjusted.columns) else pd.DataFrame()
    adjusted_auc.to_csv(output/'length_adjusted_auc.csv', index=False)
    bin_rows = []
    if bins is not None:
        for cohort, part in cohorts.items():
            if cohort.startswith('distributed:'):
                continue
            test = part[part.evaluation_split == 'test']
            for bucket in sorted(bins.loc[test.index].unique()):
                subset = test[bins.loc[test.index] == bucket]
                for metric in main_metrics:
                    result = source_bootstrap_auc(subset, -quality[metric], rng, n_bootstrap)
                    if result:
                        bin_rows.append({'cohort': cohort, 'length_bin': int(bucket), 'metric': metric, **result})
    pd.DataFrame(bin_rows).to_csv(output/'within_length_bin_auc.csv', index=False)
    distribution_table(frame, quality, output)
    legacy = []
    if 'nli_bidirectional_min' in quality:
        positives = frame[(frame.dataset == 'opai_positive') & (frame.evaluation_split == 'test')]
        for key, part in [('all', positives), *list(positives.groupby('generator'))]:
            values = quality.loc[part.index, 'nli_bidirectional_min']
            legacy.append({'editor': key, 'split': 'test', 'historical_support_cutoff': .040749866515398,
                           'n_pairs': len(part), 'n_alerts': int((values < .040749866515398).sum()),
                           'note': 'Historical OpAI-only diagnostic; not primary and not transferable to DIPPER.'})
    pd.DataFrame(legacy).to_csv(output/'historical_threshold_diagnostic.csv', index=False)
    summary = {'n_prepared_pairs': len(pairs), 'n_evaluated_pairs': len(frame),
               'n_positive_controls': int((frame.dataset == 'opai_positive').sum()),
               'n_distributed_pairs': int((frame.dataset == 'opai_distributed_negative').sum()),
               'n_distributed_positive_controls': int((frame.dataset=='opai_distributed_positive').sum()),
               'n_metrics': len(quality.columns), 'n_bootstrap': n_bootstrap,
               'calibration_test_source_overlap': 0,
               'primary_endpoint': 'Continuous alteration ranking; no universal decision threshold.',
               'labels_are_proxy_not_human_gold': True,
               'DIPPER_or_other_humanizers_executed': False, 'partial_diagnostic_allowed': allow_partial,
               'length_adjustment_fitted_only_on_calibration_positives': True,
               'cohorts': {name: {'n_pairs': len(part), 'n_sources': part.source_group.nunique()} for name, part in cohorts.items()}}
    (output/'analysis_manifest.json').write_text(json.dumps(summary, indent=2)+'\n')
    _plots_and_report(frame, quality, auc, adjusted_auc, length, summary, output)
    return summary


def _plots_and_report(frame, quality, auc, adjusted_auc, length, summary, output):
    rows = []
    primary = auc[(auc.cohort == 'opai_matched_primary') & (auc.split == 'test')]
    for family in FAMILIES[:3]:
        for agg in AGGREGATORS:
            metric = f'{family}_bidirectional_{agg}'
            selected = primary[primary.metric == metric]
            if len(selected):
                row = selected.iloc[0]
                rows.append({'famiglia': DISPLAY[family], 'aggregatore': agg,
                             'AUC': row.auc, 'IC_basso': row.ci_low, 'IC_alto': row.ci_high,
                             'positivi': row.n_preserved, 'alterati': row.n_altered})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output/'fair_primary_comparison.csv', index=False)
    if len(comparison):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for offset, (family, part) in zip([-.18,0,.18], comparison.groupby('famiglia', sort=False)):
            part = part.set_index('aggregatore').reindex(AGGREGATORS)
            x = np.arange(3) + offset
            ax.errorbar(x, part.AUC, yerr=[np.maximum(part.AUC-part.IC_basso,0),np.maximum(part.IC_alto-part.AUC,0)],
                        fmt='o', capsize=3, label=family)
        ax.axhline(.5, color='gray', linestyle='--', linewidth=.8)
        ax.set_xticks(range(3), ['Media', 'q10', 'Minimo'])
        ax.set_ylim(0,1.03)
        ax.set_ylabel('AUC sul test separato')
        ax.set_title('Stesso livello di aggregazione per tutte le metriche')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output/'fair_aggregation_auc.png', dpi=170)
        fig.savefig(output/'fair_aggregation_auc.pdf')
        plt.close(fig)
    distributed = auc[auc.cohort.str.startswith('distributed:')].copy()
    if len(distributed):
        distributed[['family','experiment','panel','k']] = distributed.cohort.str.split(':', expand=True).iloc[:,1:5].set_axis(['family','experiment','panel','k'],axis=1)
        distributed['k'] = distributed.k.str.removeprefix('k').astype(int)
        distributed.to_csv(output/'distributed_auc.csv', index=False)
        families = list(distributed.family.unique())
        fig, axes = plt.subplots(2, len(families), figsize=(7*len(families),8), squeeze=False)
        for row, split in enumerate(['all_descriptive','test']):
            for ax, family in zip(axes[row], families):
                family_part = distributed[distributed.family==family]
                chosen_experiment = 'expanded_source_panel' if (family_part.experiment=='expanded_source_panel').any() else 'initial_source_panel'
                part = family_part[(family_part.experiment==chosen_experiment) & family_part.panel.str.endswith('common_5_primary') & (family_part.split==split)]
                for metric in [f'nli_bidirectional_{agg}' for agg in AGGREGATORS]+['nli_bidirectional_supported_fraction']:
                    values = part[part.metric==metric].sort_values('k')
                    if len(values):
                        ax.plot(values.k, values.auc, marker='o', label=metric.replace('nli_bidirectional_',''))
                ax.set_xticks([1,2,3,5])
                ax.set_ylim(0,1.03)
                ax.axhline(.5,color='gray',linestyle='--',linewidth=.8)
                ax.set_title(family+' / '+chosen_experiment+'\n'+('tutte le fonti, descrittivo' if split=='all_descriptive' else 'test separato'))
                ax.set_xlabel('Frasi corrotte, stesso pannello di originali')
                ax.set_ylabel('AUC')
                ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output/'distributed_aggregators.png',dpi=170)
        fig.savefig(output/'distributed_aggregators.pdf')
        plt.close(fig)
    selected_length = length[(length.stratum=='all') & (length.length=='max_sentences') & length.metric.str.endswith('_min')]
    adjusted_test = adjusted_auc[(adjusted_auc.cohort=='opai_matched_primary') & (adjusted_auc.split=='test')] if len(adjusted_auc) else pd.DataFrame()
    top3 = primary[primary.metric.str.startswith('aligned_bert_top3_f1_rescaled_bidirectional')]
    counts = frame.groupby(['dataset','evaluation_split']).agg(n_pairs=('id','size'),n_sources=('source_group','nunique')).reset_index()
    lines = ['# Confronto equo delle metriche e dipendenza dalla lunghezza','',
             'Le metriche allineate lavorano sulle stesse frasi/chunk, con massimo supporto/similarità per unità ipotesi e media, q10 o minimo. '
             'Il confronto principale usa originali e riscritture OpAI appaiati, con sorgenti separate fra calibrazione e test. '
             'Le etichette rimangono proxy operative, non gold umane.','',
             '## Campione','',markdown_table(counts),'',
             '## Stesso aggregatore per similarità e NLI','',markdown_table(comparison),'',
             '![Confronto equo](fair_aggregation_auc.png)','',
             'Coseno e BERTScore allineati prendono il massimo su tutte le unità del documento opposto. '
             'NLI conserva il protocollo precedente: massimo fra fino a tre evidenze recuperate con MiniLM. '
             'La tabella di sensibilità sotto limita anche BERTScore agli stessi tre candidati NLI. '
             'I documenti lunghi sono rappresentati da frasi/chunk senza troncamento silenzioso. '
             'L’aggregazione bidirezionale prende il supporto minore delle due direzioni; tutte le AUC hanno orientamento prefissato.','',
             markdown_table(top3[['metric','auc','ci_low','ci_high']]) if len(top3) else 'Sensibilità top3 non disponibile.','',
             'Gli IC sono bootstrap percentile per sorgente. Nei confronti appaiati finiti possono essere spostati, '
             'specialmente vicino ad AUC 0,5, e non contenere il punto stimato. Non dimostrano differenze piccole fra aggregatori. '
             'BERTScore riscalato è una trasformazione affine crescente: non cambia ranking o AUC.','',
             '## Errori concentrati e distribuiti','',
             'I pannelli common_5_primary ed expanded_common_5_primary usano gli stessi output positivi per K=1,2,3,5: i primi K edit sono annidati e '
             'riguardano frasi distinte. Le famiglie negazione e sottili sono separate. '
             'I pannelli applicability_2to4_secondary non sono confrontabili come se avessero la stessa popolazione per ogni K. '
             'I negativi sono confrontati soltanto con gli esatti positivi collegati, mai con un altro campione di parafrasi.','',
             'Se presente, expanded_source_panel è il pannello principale per le curve K: usa fonti aggiuntive distinte dai 99 originali '
             'dei controlli precedenti. Initial_source_panel rimane separato come controllo esplorativo di applicabilità. '
             'Le due popolazioni non vengono mescolate né confrontate come se cambiassero soltanto K. '
             'I riferimenti delle fasce di lunghezza restano sempre i 300 positivi originali.','',
             'Le curve su tutte le fonti sono descrittive e non partecipano ad alcuna scelta dell’aggregatore. '
             'Il test separato resta la valutazione principale, ma il pannello con cinque frasi applicabili può essere molto piccolo; '
             'i relativi conteggi e gli avvisi di numerosità sono mantenuti nelle tabelle.','',
             '![Errori distribuiti](distributed_aggregators.png)' if len(distributed) else 'Punteggi delle corruzioni distribuite non ancora disponibili.','',
             'Le corruzioni a regole non certificano la distribuzione delle derive degli humanizer reali; aumentare K prova la sensibilità '
             'alla diffusione dell’errore, non sostituisce una loro valutazione. Dettagli per K/famiglia in `distributed_auc.csv`.','',
             '## Il minimo dipende dalla lunghezza?','',
             markdown_table(selected_length[['metric','rho','ci_low','ci_high','n_sources']]),'',
             '![Lunghezza e minimo](length_positive_nli.png)','',
             'Il riepilogo principale è Spearman fra medie per sorgente sui soli positivi presunti. '
             'I risultati per dominio, editor e operazione sono in `length_correlations.csv`. '
             'Il diagnostico entro sorgente in `length_within_source.csv` centra i ranghi per fonte: non è una stima causale della lunghezza '
             'e può essere poco informativo se quasi nessuna fonte cambia numero di frasi. '
             'Una correlazione fra lunghezza e minimo può dipendere anche da contenuto o difficoltà; non va attribuita automaticamente al modello.','',
             '## Normalizzazione esplorativa per fasce','',
             'I confini delle fasce sono i tercili pesati del numero massimo di frasi fra originale e riscrittura, '
             'appresi soltanto dai positivi di calibrazione. Ogni supporto diventa il suo percentile empirico pesato entro fascia; '
             'le fasce troppo piccole ripiegano sulla calibrazione positiva complessiva. '
             'È un punteggio continuo relativo a OpAI, non una probabilità di correttezza né una soglia universale. '
             'I parametri sono in `length_calibration.json` e `length_calibration_reference.csv`.','',
             markdown_table(adjusted_test[['metric','auc','ci_low','ci_high']]) if len(adjusted_test) else 'Non disponibile.','',
             'Le AUC entro fascia sono in `within_length_bin_auc.csv`; questo controllo non usa le etichette del test per adattare il percentile.','',
             '## Suite: punteggi continui e distribuzioni','',
             '**L’output primario è continuo.** `continuous_score_distributions.csv` riporta quantili per editor, operazione, condizione e split. '
             'Gli editor OpAI non sono humanizer intercambiabili: DIPPER e altri humanizer non sono stati eseguiti qui. '
             'La precedente soglia 0,04075 rimane soltanto nel file storico diagnostico, senza essere adottata come risultato trasferibile. '
             'Qualunque soglia operativa futura richiede una calibrazione sulla distribuzione bersaglio.','',
             'Le guardie lessicali complementari e il loro confronto con NLI sono in `guards_full/`: '
             'rilevano cambi di cautele, negazioni, valori numerici e quantificatori, senza certificare da sole un errore semantico.','',
             'File completi: `fair_metric_auc.csv`, `fair_primary_comparison.csv`, `fair_auc_by_stratum.csv`, '
             '`length_adjusted_auc.csv`, `continuous_score_distributions.csv`, `analysis_manifest.json`. '
             'Le valutazioni precedenti sono conservate come esperimenti storici.','']
    (output/'report.md').write_text('\n'.join(lines))

"""Ricostruisce le tabelle sugli humanizer dagli output salvati.

"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
METRICS = ['word_ratio', 'original_n_words', 'modified_n_words', 'style_rms', 'pos_js', 'delta__mattr_50', 'delta_human_centroid_distance', 'embedding_cosine', 'bert_chunk_f1', 'nli_forward_mean', 'nli_reverse_mean', 'nli_bidirectional_min', 'nli_bidirectional_q10', 'nli_forward_low_support_fraction', 'nli_reverse_low_support_fraction', 'aligned_cosine_bidirectional_mean', 'aligned_cosine_bidirectional_min', 'aligned_bert_f1_rescaled_bidirectional_mean', 'aligned_bert_f1_rescaled_bidirectional_min', 'lexical_rouge_l_f1', 'lexical_bleu']
PAIRED_METRICS = ['word_ratio', 'style_rms', 'delta__mattr_50', 'delta_human_centroid_distance', 'embedding_cosine', 'bert_chunk_f1', 'nli_forward_mean', 'nli_reverse_mean', 'nli_bidirectional_min', 'nli_bidirectional_q10', 'aligned_cosine_bidirectional_min', 'aligned_bert_f1_rescaled_bidirectional_min']

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_rewrites(run):
    run = Path(run)
    candidates = [run / 'rewriting/rewrites.jsonl', run / 'rewriting_cuda0/rewrites.jsonl']
    path = next((path for path in candidates if path.exists()), None)
    if path is None:
        raise FileNotFoundError('Expected rewriting/rewrites.jsonl or rewriting_cuda0/rewrites.jsonl')
    frame = pd.DataFrame(read_jsonl(path))
    keys = ['id', 'source_id', 'source_group_id', 'domain', 'generator', 'humanizer', 'status']
    if not set(keys).issubset(frame) or frame.id.duplicated().any():
        raise ValueError('Rewrite metadata are incomplete or IDs are duplicated')
    return (frame, path)

def cohort_path(run, study=False):
    name = 'study.jsonl' if study else 'pilot.jsonl'
    path = Path(run) / 'cohorts' / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path

def merged_pair_scores(run, pilot_ids=()):
    """Unisce i CSV grezzi delle fasi dei modelli"""
    run = Path(run)
    rewrites, _ = read_rewrites(run)
    columns = ['id', 'source_id', 'source_group_id', 'domain', 'generator', 'humanizer', 'status']
    frame = rewrites[columns].copy()
    frame['source_subset'] = np.where(frame.source_id.isin(set(pilot_ids)), 'pilot42', 'unseen84')
    files = sorted((run / 'semantic/scores').glob('*.csv'))
    files += [run / 'style/style_pairs.csv', run / 'style/guards.csv']
    if not files or not (run / 'semantic/scores/nli.csv').exists():
        raise FileNotFoundError('Saved semantic stage scores, including nli.csv, are required')
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        part = pd.read_csv(path)
        if 'id' not in part or part.id.duplicated().any():
            raise ValueError(f'Nonunique stage score IDs: {path}')
        if not set(part.id).issubset(set(frame.id)):
            raise ValueError(f'Stage scores contain foreign rewrite IDs: {path}')
        if set(part.id) != set(frame.loc[frame.status.eq('ok'), 'id']):
            raise ValueError(f'Stage scores do not cover exactly the successful rewrites: {path}')
        for column in ['source_id', 'source_group_id', 'domain', 'generator', 'humanizer']:
            if column in part:
                check = frame[['id', column]].merge(part[['id', column]], on='id', suffixes=('_a', '_b'), validate='one_to_one')
                if not check[column + '_a'].eq(check[column + '_b']).all():
                    raise ValueError(f'Stage source metadata disagree: {path}, {column}')
        additions = [column for column in part if column == 'id' or column not in frame]
        frame = frame.merge(part[additions], on='id', how='left', validate='one_to_one')
    for agg in ['min', 'q10']:
        frame['nli_bidirectional_' + agg] = frame[['nli_forward_' + agg, 'nli_reverse_' + agg]].min(axis=1, skipna=False)
    return frame

def semantic_summary(frame, subsets):
    rows = []
    for subset, ids in subsets.items():
        for name, part in frame[frame.source_id.isin(ids)].groupby('humanizer', sort=True):
            valid = part[part.status.eq('ok')]
            row = {'subset': subset, 'humanizer': name, 'n_sources': len(part), 'n_valid': len(valid), 'n_failed': len(part) - len(valid)}
            for key, cond in {'n_shorter': valid.word_ratio < 1, 'n_longer': valid.word_ratio > 1, 'n_equal': valid.word_ratio == 1, 'n_closer_human_centroid': valid.delta_human_centroid_distance < 0, 'n_farther_human_centroid': valid.delta_human_centroid_distance > 0, 'n_mattr_increase': valid['delta__mattr_50'] > 0, 'n_mattr_decrease': valid['delta__mattr_50'] < 0}.items():
                row[key] = int(cond.sum())
            for metric in METRICS:
                if metric in valid:
                    row[metric + '__mean'] = valid[metric].mean()
                    row[metric + '__median'] = valid[metric].median()
            for family in ['hedge', 'negation', 'number', 'quantifier']:
                for suffix in ['change', 'count_loss', 'count_gain']:
                    col = 'guard_' + family + '_' + suffix
                    if col in valid:
                        values = pd.to_numeric(valid[col], errors='raise') > 0
                        row['n_' + col] = int(values.sum())
            rows.append(row)
    return pd.DataFrame(rows)

def semantic_paired(frame, subsets, comparisons):
    rows = []
    valid = frame[frame.status.eq('ok')]
    for a, b in comparisons:
        for subset, ids in subsets.items():
            part = valid[valid.source_id.isin(ids)]
            pair = part[part.humanizer.eq(a)].merge(part[part.humanizer.eq(b)], on='source_id', suffixes=('_a', '_b'), validate='one_to_one')
            row = {'subset': subset, 'system_a': a, 'system_b': b, 'n_common_sources': len(pair)}
            for metric in PAIRED_METRICS:
                if metric + '_a' in pair and metric + '_b' in pair:
                    row[metric + '__mean_delta_b_minus_a'] = (pair[metric + '_b'] - pair[metric + '_a']).mean()
            row['n_b_longer_than_a'] = int((pair.modified_n_words_b > pair.modified_n_words_a).sum())
            row['n_b_shorter_than_a'] = int((pair.modified_n_words_b < pair.modified_n_words_a).sum())
            rows.append(row)
    return pd.DataFrame(rows)

def primary_nli(run, frame, subsets):
    path = Path(run) / 'semantic_context/contextual_nli.csv'
    ctx = pd.read_csv(path)
    ctx = ctx[ctx['mode'].eq('window2')]
    if ctx.duplicated(['id', 'direction']).any():
        raise ValueError('Duplicate contextual window2/direction scores')
    if not ctx.complete.map(lambda x: str(x).lower() == 'true').all():
        raise ValueError('Incomplete contextual window2 coverage cannot silently enter primary scores')
    pivot = ctx.pivot(index='id', columns='direction', values='min').rename(columns={'forward': 'primary_forward', 'reverse': 'primary_reverse'})
    scores = frame[['id', 'source_id', 'source_group_id', 'domain', 'generator', 'humanizer', 'status', 'source_subset']].merge(pivot, on='id', how='left', validate='one_to_one')
    scores['primary_mode'] = 'window2'
    scores['primary_aggregator'] = 'min'
    scores['primary_score'] = scores[['primary_forward', 'primary_reverse']].min(axis=1, skipna=False)
    if scores.loc[scores.status.eq('ok'), 'primary_score'].isna().any():
        raise ValueError('A successful rewrite is missing its primary NLI score')
    scores['primary_status'] = np.where(scores.primary_score.notna(), 'scored', 'generation_failed')
    rows = []
    for subset, ids in subsets.items():
        for name, part in scores[scores.source_id.isin(ids)].groupby('humanizer', sort=True):
            rows.append({'subset': 'all126' if subset == 'overall' else subset, 'humanizer': name, 'n_attempts': len(part), 'n_scored': int(part.primary_score.notna().sum()), 'n_failed': int(part.primary_score.isna().sum()), 'mean_primary_score': part.primary_score.mean(), 'median_primary_score': part.primary_score.median(), 'mean_forward_min': part.primary_forward.mean(), 'mean_reverse_min': part.primary_reverse.mean(), 'decision_threshold': 'none', 'human_validation': 'pending'})
    return (scores, pd.DataFrame(rows))

def detector_decisions(run):
    run = Path(run)
    frame = pd.read_csv(run / 'detection/detector_scores.csv').fillna({'humanizer': ''})
    thresholds = json.loads((run / 'detection/thresholds.json').read_text())
    parts = []
    for name, part in frame.groupby('detector', sort=False):
        part = part.copy()
        part['threshold'] = thresholds[name]['threshold']
        part['is_ai'] = part.score > part.threshold
        parts.append(part)
    if len(parts) != 3:
        raise ValueError('The historical majority requires the three supervised detectors')
    identifiers = ['source_id', 'source_group_id', 'domain', 'generator', 'split', 'role', 'humanizer']
    vote = pd.concat(parts).groupby(identifiers, sort=False, dropna=False).is_ai.agg(['sum', 'count']).reset_index()
    if not vote['count'].eq(3).all():
        raise ValueError('Incomplete supervised-detector majority')
    vote['is_ai'] = vote['sum'] >= 2
    vote['detector'] = 'majority_vote'
    vote['threshold'] = 0.5
    parts.append(vote)
    zero_path = run / 'zero_shot/zero_shot_scores.csv'
    if zero_path.exists():
        zero = pd.read_csv(zero_path).fillna({'humanizer': ''})
        zero_thresholds = json.loads((run / 'zero_shot/zero_shot_thresholds.json').read_text())
        for name, part in zero.groupby('detector', sort=False):
            part = part.copy()
            part['threshold'] = zero_thresholds[name]['threshold']
            part['is_ai'] = part.score < part.threshold
            parts.append(part)
    return pd.concat(parts, ignore_index=True)

def detection_tables(run, frame, subsets, draws=5000, seed=42):
    decisions = detector_decisions(run)
    systems = [s for s in frame.humanizer.unique() if s != 'identity']
    tables = defaultdict(list)
    for subset, ids in subsets.items():
        selected = frame[frame.source_id.isin(ids)]
        base = selected.drop_duplicates('source_id').set_index('source_id')
        panel = 'all126' if subset == 'overall' and len(base) == 126 else 'pilot42' if subset == 'overall' else subset
        for split in ['overall', 'domain', 'generator']:
            strata = [('overall', list(base.index))] if split == 'overall' else [(v, list(p.index)) for v, p in base.groupby(split, sort=True)]
            for stratum, chosen_ids in strata:
                for name, block in decisions[decisions.split.eq('test')].groupby('detector', sort=False):
                    if block.duplicated(['source_id', 'role', 'humanizer']).any():
                        raise ValueError('Duplicate detector text scores')
                    lookup = {(r.source_id, r.role, r.humanizer): r for r in block.itertuples(index=False)}
                    for system in systems:
                        p = selected[selected.humanizer.eq(system)].set_index('source_id').loc[chosen_ids]
                        before = np.array([lookup[s, 'machine', ''].is_ai for s in p.index], dtype=bool)
                        human = np.array([lookup[s, 'human', ''].is_ai for s in p.index], dtype=bool)
                        ok = p.status.eq('ok').to_numpy()
                        after = np.array([lookup[s, 'rewrite', system].is_ai if good else False for s, good in zip(p.index, ok)], dtype=bool)
                        success = ok & ~after
                        eligible = before
                        domain = stratum if split == 'domain' else 'overall'
                        generator = stratum if split == 'generator' else 'overall'
                        seed_parts = [seed, name, system, domain] if panel == 'all126' and generator == 'overall' else [seed, name, system, panel, domain, generator]
                        asr_seed = digest_seed(*seed_parts)
                        low, high = bootstrap_mean(success[eligible], p.source_group_id.to_numpy()[eligible], asr_seed, draws)
                        delta = after[ok].astype(int) - before[ok].astype(int)
                        dlo, dhi = bootstrap_mean(delta, p.source_group_id.to_numpy()[ok], digest_seed('paired_tpr', *seed_parts), draws)
                        wlo, whi = wilson(int(success[eligible].sum()), int(eligible.sum()))
                        mean = lambda v: float(np.mean(v)) if len(v) else math.nan
                        row = {'panel': panel, 'domain': stratum if split == 'domain' else 'overall', 'generator': stratum if split == 'generator' else 'overall', 'detector': name, 'humanizer': system, 'n_sources': len(p), 'n_source_groups': p.source_group_id.nunique(), 'human_false_positives': int(human.sum()), 'human_test_fpr': mean(human), 'original_detected_all': int(before.sum()), 'original_tpr_all': mean(before), 'successful_generations': int(ok.sum()), 'failed_generations': int((~ok).sum()), 'failed_original_detected': int((before & ~ok).sum()), 'baseline_detected': int(eligible.sum()), 'attack_successes': int(success[eligible].sum()), 'asr': mean(success[eligible]), 'asr_boot_ci95_low': low, 'asr_boot_ci95_high': high, 'asr_wilson95_low': wlo, 'asr_wilson95_high': whi, 'paired_original_detected': int(before[ok].sum()), 'paired_rewrite_detected': int(after[ok].sum()), 'paired_original_tpr': mean(before[ok]), 'paired_rewrite_tpr': mean(after[ok]), 'paired_tpr_delta': mean(delta), 'paired_tpr_delta_ci95_low': dlo, 'paired_tpr_delta_ci95_high': dhi, 'new_detections': int((~before & after & ok).sum()), 'lost_detections': int((before & ~after & ok).sum()), 'retained_ai': int((before & after & ok).sum()), 'retained_human': int((~before & ~after & ok).sum()), 'rewrite_detection_per_all_sources_observed': float((after & ok).sum() / len(p)), 'threshold': float(block.threshold.iloc[0])}
                        tables[split].append(row)
    return ({key: pd.DataFrame(value) for key, value in tables.items()}, decisions)

def paired_writers(frame, decisions, subsets, comparisons, draws=5000, seed=42):
    rows = []
    for subset, ids in subsets.items():
        part = frame[frame.source_id.isin(ids)]
        panel = 'all126' if subset == 'overall' else subset
        for name, block in decisions[decisions.split.eq('test')].groupby('detector', sort=False):
            lookup = {(r.source_id, r.role, r.humanizer): bool(r.is_ai) for r in block.itertuples(index=False)}
            for a, b in comparisons:
                aa = part[part.humanizer.eq(a)].set_index('source_id').sort_index()
                bb = part[part.humanizer.eq(b)].set_index('source_id').reindex(aa.index)
                before = np.array([lookup[s, 'machine', ''] for s in aa.index], dtype=bool)
                oka, okb = (aa.status.eq('ok').to_numpy(), bb.status.eq('ok').to_numpy())
                aftera = np.array([lookup[s, 'rewrite', a] if ok else False for s, ok in zip(aa.index, oka)])
                afterb = np.array([lookup[s, 'rewrite', b] if ok else False for s, ok in zip(aa.index, okb)])
                ea = (oka & ~aftera)[before]
                eb = (okb & ~afterb)[before]
                groups = aa.source_group_id.to_numpy()
                lo, hi = bootstrap_mean(eb.astype(int) - ea.astype(int), groups[before], historical_seed(['writer_asr_diff', panel, name, a, b], seed), draws)
                common = oka & okb
                delta = afterb[common].astype(int) - aftera[common].astype(int)
                dlo, dhi = bootstrap_mean(delta, groups[common], historical_seed(['writer_tpr_diff', panel, name, a, b], seed), draws)
                rows.append({'panel': panel, 'detector': name, 'system_a': a, 'system_b': b, 'baseline_detected': int(before.sum()), 'asr_a': float(ea.mean()), 'asr_b': float(eb.mean()), 'asr_b_minus_a': float(eb.mean() - ea.mean()), 'asr_difference_ci95_low': lo, 'asr_difference_ci95_high': hi, 'common_successful_sources': int(common.sum()), 'rewrite_tpr_b_minus_a': float(delta.mean()), 'tpr_difference_ci95_low': dlo, 'tpr_difference_ci95_high': dhi})
    return pd.DataFrame(rows)

def null_comparisons(run, frame, decisions, subsets, draws=5000, seed=42):
    controls_path = Path(run) / 'null_controls/rewrites.jsonl'
    controls = read_jsonl(controls_path)
    writers = ['qwen_humanize', 'qwen14b_humanize', 'dipper', 'dipper_context3']
    references, _ = read_rewrites(run)
    references = {(r.humanizer, r.source_id): r for r in references.itertuples(index=False)}
    conditions = sorted({r['humanizer'] for r in controls if r.get('metadata', {}).get('protocol') == 'humanizer_null_controls_v1'})
    lookup = {(r.detector, r.source_id, r.role, r.humanizer): bool(r.is_ai) for r in decisions.itertuples(index=False)}
    thresholds = decisions.groupby('detector').threshold.first().to_dict()
    names = list(decisions.detector.unique())
    result = []
    rate = lambda values: float(np.mean(values)) if len(values) else math.nan
    for subset, ids in subsets.items():
        panel = 'all126' if subset == 'overall' else subset
        source = frame[frame.source_id.isin(ids)].drop_duplicates('source_id').set_index('source_id')
        for name in names:
            for condition in conditions:
                rows = [r for r in controls if r['humanizer'] == condition and r['source_id'] in ids]
                if not rows:
                    continue
                eligible = [r for r in rows if r['status'] == 'ok' and r.get('metadata', {}).get('comparison_eligible')]
                writer = rows[0]['metadata'].get('reference_humanizer', '')
                comparisons = [(writer, eligible)]
                if condition == 'null_lexical_noise':
                    comparisons += [(w, [r for r in eligible if references[w, r['source_id']].status == 'ok']) for w in writers]
                for writer, eligible in comparisons:
                    before = np.array([lookup[name, r['source_id'], 'machine', ''] for r in eligible], dtype=bool)
                    after = np.array([lookup[name, r['source_id'], 'rewrite', condition] for r in eligible], dtype=bool)
                    rewritten = np.array([lookup[name, r['source_id'], 'rewrite', writer] for r in eligible], dtype=bool) if writer else np.array([], dtype=bool)
                    cs = ~after[before]
                    rs = ~rewritten[before] if writer else np.array([], dtype=bool)
                    diff = rs.astype(int) - cs.astype(int) if writer else np.array([])
                    groups = np.array([source.loc[r['source_id'], 'source_group_id'] for r in eligible])
                    low, high = bootstrap_mean(cs, groups[before], historical_seed(['null_asr', panel, name, condition, writer], seed), draws)
                    dl, dh = bootstrap_mean(diff, groups[before] if writer else [], historical_seed(['null_asr_diff', panel, name, condition, writer], seed), draws)
                    td = rewritten.astype(int) - after.astype(int) if writer else np.array([])
                    tl, th = bootstrap_mean(td, groups if writer else [], historical_seed(['null_tpr_diff', panel, name, condition, writer], seed), draws)
                    result.append({'panel': panel, 'detector': name, 'condition': condition, 'reference_humanizer': writer, 'total_sources': len(rows), 'eligible_sources': len(eligible), 'ineligible_sources': len(rows) - len(eligible), 'eligible_original_detected': int(before.sum()), 'control_successes': int(cs.sum()), 'control_asr': rate(cs), 'control_asr_ci95_low': low, 'control_asr_ci95_high': high, 'reference_successes': int(rs.sum()) if writer else math.nan, 'reference_asr': rate(rs), 'delta_asr_reference_minus_control': rate(diff), 'delta_asr_ci95_low': dl, 'delta_asr_ci95_high': dh, 'control_only_successes': int((cs & ~rs).sum()) if writer else math.nan, 'reference_only_successes': int((rs & ~cs).sum()) if writer else math.nan, 'eligible_original_tpr': rate(before), 'eligible_control_tpr': rate(after), 'eligible_reference_tpr': rate(rewritten), 'reference_minus_control_tpr': rate(td), 'reference_minus_control_tpr_ci95_low': tl, 'reference_minus_control_tpr_ci95_high': th, 'threshold': float(thresholds[name]), 'ai_direction': 'lower' if name == 'binoculars_falcon7b' else 'higher', 'causal_attribution': False})
    return pd.DataFrame(result)

def pilot_pair_comparison(qwen_run, dipper_run, output, draws=5000, seed=42):
    qwen, _ = read_rewrites(qwen_run)
    dipper, _ = read_rewrites(dipper_run)
    qwen = qwen[qwen.humanizer.eq('qwen_humanize')].set_index('source_id')
    dipper = dipper[dipper.humanizer.eq('dipper')].set_index('source_id')
    if set(qwen.index) != set(dipper.index) or not qwen.original.equals(dipper.original.reindex(qwen.index)):
        raise ValueError('Pilot comparison requires the exact same originals')
    qdec, ddec = (detector_decisions(qwen_run), detector_decisions(dipper_run))
    rows = []
    for detector in sorted(set(qdec.detector) & set(ddec.detector)):

        def lookup(table):
            p = table[table.detector.eq(detector) & table.split.eq('test')]
            return {(r.source_id, r.role, r.humanizer): bool(r.is_ai) for r in p.itertuples(index=False)}
        qlookup, dlookup = (lookup(qdec), lookup(ddec))
        ids = sorted(qwen.index)
        for source in ids:
            if qlookup[source, 'machine', ''] != dlookup[source, 'machine', '']:
                raise ValueError('Pilot baseline detector decisions differ')
        ids = [source for source in ids if qlookup[source, 'machine', '']]
        qa = np.array([qwen.loc[s, 'status'] == 'ok' and (not qlookup.get((s, 'rewrite', 'qwen_humanize'), True)) for s in ids], dtype=bool)
        da = np.array([dipper.loc[s, 'status'] == 'ok' and (not dlookup.get((s, 'rewrite', 'dipper'), True)) for s in ids], dtype=bool)
        groups = [qwen.loc[s, 'source_group_id'] for s in ids]
        lo, hi = bootstrap_mean(da.astype(int) - qa.astype(int), groups, seed, draws)
        rows.append({'detector': detector, 'n_test_sources': len(qwen), 'baseline_detected': len(ids), 'qwen_successes': int(qa.sum()), 'dipper_successes': int(da.sum()), 'qwen_asr': float(qa.mean()), 'dipper_asr': float(da.mean()), 'delta_asr_dipper_minus_qwen': float(da.mean() - qa.mean()), 'paired_delta_ci95_low': lo, 'paired_delta_ci95_high': hi, 'both_evasion': int((qa & da).sum()), 'qwen_only_evasion': int((qa & ~da).sum()), 'dipper_only_evasion': int((~qa & da).sum()), 'neither_evasion': int((~qa & ~da).sum()), 'bootstrap_draws': draws, 'bootstrap_seed': seed})
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(output) / 'paired_asr_comparison.csv', index=False)
    text = '# Confronto appaiato Qwen 3B e DIPPER sul pilot\n\n'
    text += 'ASR condizionata agli originali già rilevati; i fallimenti restano insuccessi. '
    text += f'Intervalli percentile per gruppo, {draws} ricampionamenti, seed {seed}.\n\n'
    text += '| Detector | Originali rilevati | ASR Qwen | ASR DIPPER | Differenza DIPPER−Qwen | IC 95% differenza |\n'
    text += '| --- | ---: | ---: | ---: | ---: | --- |\n'
    for row in frame.itertuples(index=False):
        text += f'| {row.detector} | {row.baseline_detected} | {100 * row.qwen_asr:.1f}% | {100 * row.dipper_asr:.1f}% | {100 * row.delta_asr_dipper_minus_qwen:+.1f} pp | [{100 * row.paired_delta_ci95_low:+.1f}; {100 * row.paired_delta_ci95_high:+.1f}] pp |\n'
    text += '\nI valori sono sulla scala 0–1. Gli intervalli sono esplorativi e mantengono fisse le soglie.\n'
    (Path(output) / 'paired_comparison.md').write_text(text)
    ids = set(qwen.index)
    qframe = merged_pair_scores(qwen_run, ids)
    dframe = merged_pair_scores(dipper_run, ids)
    merged = pd.concat([qframe, dframe[dframe.humanizer.eq('dipper')]], ignore_index=True)
    merged.to_csv(Path(output) / 'per_pair_summary.csv', index=False)
    semantic_summary(merged, {'overall': ids}).to_csv(Path(output) / 'pilot_semantic_style_summary.csv', index=False)
    semantic_paired(merged, {'overall': ids}, [('qwen_humanize', 'dipper')]).to_csv(Path(output) / 'semantic_paired_differences.csv', index=False)
    return frame

def compare_reference(output, reference, filenames):
    """Riporta esplicitamente le differenze numeriche; non sostituisce celle con quelle di riferimento."""
    rows = []
    for filename in filenames:
        actual_path, expected_path = (Path(output) / filename, Path(reference) / filename)
        if not actual_path.exists() or not expected_path.exists():
            continue
        actual, expected = (pd.read_csv(actual_path), pd.read_csv(expected_path))
        possible = ['panel', 'subset', 'domain', 'generator', 'detector', 'humanizer', 'system_a', 'system_b', 'condition', 'reference_humanizer', 'id']
        keys = [c for c in possible if c in actual and c in expected]
        if not keys:
            continue
        joined = actual.merge(expected, on=keys, suffixes=('_new', '_reference'), validate='one_to_one', how='outer', indicator=True)
        for column in actual:
            if column in keys or column not in expected or (not pd.api.types.is_numeric_dtype(actual[column])) or (not pd.api.types.is_numeric_dtype(expected[column])):
                continue
            a, b = (joined[column + '_new'].to_numpy(dtype=float), joined[column + '_reference'].to_numpy(dtype=float))
            finite = np.isfinite(a) & np.isfinite(b)
            delta = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
            missing_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
            rows.append({'table': filename, 'column': column, 'n_compared': int(finite.sum()), 'max_abs_difference': delta, 'missing_mismatch': missing_mismatch, 'foreign_or_missing_rows': int(joined['_merge'].ne('both').sum()), 'matches_at_1e_12': delta < 1e-12 and (not missing_mismatch) and joined['_merge'].eq('both').all()})
    result = pd.DataFrame(rows)
    result.to_csv(Path(output) / 'reproduction_check.csv', index=False)
    return result

def analyze_study(run, pilot_cohort, output, draws=5000, seed=42, reference=None):
    run, output = (Path(run), Path(output))
    output.mkdir(parents=True, exist_ok=True)
    pilot_ids = {row['id'] for row in read_jsonl(pilot_cohort) if row['split'] == 'test'}
    frame = merged_pair_scores(run, pilot_ids)
    all_ids = set(frame.source_id)
    if not pilot_ids.issubset(all_ids):
        raise ValueError('Pilot source IDs are not a subset of the study')
    subsets = {'overall': all_ids, 'unseen84': all_ids - pilot_ids, 'pilot42': pilot_ids}
    frame.to_csv(output / 'per_pair_summary.csv', index=False)
    semantic_summary(frame, subsets).to_csv(output / 'semantic_style_summary.csv', index=False)
    semantic_paired(frame, subsets, [('dipper', 'dipper_context3'), ('qwen_humanize', 'qwen14b_humanize')]).to_csv(output / 'semantic_paired_differences.csv', index=False)
    primary, summary = primary_nli(run, frame, subsets)
    primary.to_csv(output / 'nli_primary_scores.csv', index=False)
    summary.to_csv(output / 'nli_primary_summary.csv', index=False)
    detection, decisions = detection_tables(run, frame, subsets, draws, seed)
    for stratum, table in detection.items():
        table.to_csv(output / ('detection_overall.csv' if stratum == 'overall' else 'detection_by_' + stratum + '.csv'), index=False)
    paired_writers(frame, decisions, subsets, [('qwen_humanize', 'qwen14b_humanize'), ('dipper', 'dipper_context3'), ('qwen_humanize', 'dipper_context3')], draws, seed).to_csv(output / 'detection_paired_writers.csv', index=False)
    if (run / 'null_controls/rewrites.jsonl').exists():
        null_comparisons(run, frame, decisions, subsets, draws, seed).to_csv(output / 'null_comparisons.csv', index=False)
    checks = compare_reference(output, reference, ['detection_overall.csv', 'detection_by_domain.csv', 'detection_by_generator.csv', 'detection_paired_writers.csv', 'null_comparisons.csv', 'semantic_style_summary.csv', 'semantic_paired_differences.csv', 'nli_primary_summary.csv', 'nli_primary_scores.csv']) if reference else pd.DataFrame()
    _, rewrites_path = read_rewrites(run)
    inputs = [rewrites_path, run / 'detection/detector_scores.csv', run / 'detection/thresholds.json', Path(pilot_cohort), run / 'semantic_context/contextual_nli.csv', run / 'style/style_pairs.csv', run / 'style/guards.csv']
    inputs += sorted((run / 'semantic/scores').glob('*.csv'))
    inputs += [p for p in [run / 'zero_shot/zero_shot_scores.csv', run / 'zero_shot/zero_shot_thresholds.json', run / 'null_controls/rewrites.jsonl'] if p.exists()]
    manifest = {'source': str(run), 'n_rewrites': len(frame), 'n_sources': len(all_ids), 'n_pilot': len(pilot_ids), 'n_unseen': len(all_ids - pilot_ids), 'models_executed': False, 'thresholds_refitted': False, 'bootstrap_draws': draws, 'bootstrap_seed': seed, 'bootstrap_note': 'Historical SHA-256-derived seed keys and source-group sorting recovered from original analysis; same 5000 paired cluster draws. All numerical reference columns, including confidence intervals, are compared in reproduction_check.csv.', 'primary_nli': 'window2 + minimum bidirectional; no decision threshold', 'input_sha256': {str(p): sha(p) for p in inputs}, 'analysis_code_sha256': sha(__file__), 'reference_numeric_checks': len(checks), 'reference_mismatched_columns': int((~checks.matches_at_1e_12).sum()) if len(checks) else None}
    (output / 'reproduction_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest
from humanization.evaluation.statistics import digest_seed, historical_seed, bootstrap_mean, wilson

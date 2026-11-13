"""Descrive le coppie pulite DATE/MONEY/PERCENT esistenti, senza inferenza dei modelli."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from humanization.evaluation.workflows.compare_entity_policy import entities, paired_changes, read_pairs, summaries
FAMILY = {'DATE', 'MONEY', 'PERCENT'}
LABELS = {'local_nli_forward_mean': 'Entailment locale, originale → modificato', 'nli_bidirectional_min': 'Supporto minimo documento, bidirezionale', 'embedding_cosine': 'Coseno documento', 'local_embedding_cosine': 'Coseno frase', 'bert_chunk_f1': 'BERTScore F1 documento', 'local_bert_f1': 'BERTScore F1 frase'}

def score(row, metric):
    if metric == 'nli_bidirectional_min':
        return min(row['nli_forward_min'], row['nli_reverse_min'])
    return row[metric]

def _argument_error(message):
    raise ValueError(message)

def run(args):
    if args.bootstrap < 1:
        _argument_error('--bootstrap must be positive')
    pairs = read_pairs(args.data_dir / 'pairs.jsonl')
    old_pairs = read_pairs(args.baseline_data / 'pairs.jsonl')
    new_scores = pd.read_csv(args.results / 'scores.csv')
    old_scores = pd.read_csv(args.baseline_results / 'scores.csv')
    all_clean = entities(pairs, new_scores)
    selected = all_clean[all_clean.entity_type.isin(FAMILY)].copy()
    if len(all_clean) != 145 or len(selected) != 34:
        raise ValueError('This frozen sensitivity analysis expects 34 numeric pairs among 145 clean entities')
    old = entities(old_pairs, old_scores)
    old = old[old.source_id.isin(selected.source_id)].copy()
    if len(old) != len(selected):
        raise ValueError('The baseline does not cover all selected sources')
    paired, deltas = paired_changes(old, selected, args.bootstrap, args.seed)
    summary = pd.DataFrame(summaries(old, 'v1', 'numeric_clean_sources') + summaries(selected, 'clean', 'numeric_clean_sources'))
    category = []
    for kind, part in selected.groupby('entity_type'):
        for row in summaries(part, 'clean', kind):
            category.append({'entity_type': kind, **row})
    category = pd.DataFrame(category)
    identities = {p['source_id']: p for p in pairs.values() if p['dataset'] == 'controlled' and p['perturbation'] == 'identity'}
    local = {p['metadata']['control_for']: p for p in pairs.values() if p['dataset'] == 'local_control'}
    indexed = new_scores.set_index('id')
    references, cases = ([], [])
    for row in selected.to_dict('records'):
        p = pairs[row['id']]
        document, sentence = (identities[row['source_id']], local[row['id']])
        if document['original'] != p['original'] or document['modified'] != p['original'] or sentence['original'] != p['target_original'] or (sentence['modified'] != p['target_original']):
            raise ValueError('Identity control does not match the selected document/target')
        for metric in LABELS:
            control = sentence if metric.startswith('local_') else document
            references.append({'source_id': row['source_id'], 'source_group_id': row['source_group_id'], 'metric': metric, 'identity': score(indexed.loc[control['id']], metric), 'altered': row[metric]})
        cases.append({**row, 'target_original': p['target_original'], 'target_modified': p['target_modified']})
    references = pd.DataFrame(references)
    reference_summary = []
    for metric, part in references.groupby('metric', sort=False):
        values = part.groupby('source_group_id')[['identity', 'altered']].mean()
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f'Missing/nonfinite identity or altered score: {metric}')
        labels = np.r_[np.ones(len(values)), np.zeros(len(values))]
        quality = np.r_[values.identity, values.altered]
        reference_summary.append({'metric': metric, 'n_sources': len(values), 'identity_mean': values.identity.mean(), 'altered_mean': values.altered.mean(), 'identity_min': values.identity.min(), 'identity_max': values.identity.max(), 'altered_min': values.altered.min(), 'altered_max': values.altered.max(), 'identity_vs_altered_auc': roc_auc_score(labels, quality)})
    reference_summary = pd.DataFrame(reference_summary)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in [('summary', summary), ('paired_deltas', deltas), ('paired_scores', paired), ('by_category', category), ('identity_references', references), ('identity_summary', reference_summary), ('cases', pd.DataFrame(cases))]:
        frame.to_csv(args.output / (name + '.csv'), index=False)
    counts = dict(Counter(selected.entity_type))
    print(f'34 pairs ({counts}), {len(paired)} matched sources; results: {args.output}')

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'data_dir': ROOT / 'data/processed/semantic_clean', 'baseline_data': ROOT / 'data/processed/semantic_baseline', 'results': ROOT / 'results/published/03_semantics/entity_clean', 'baseline_results': ROOT / 'results/published/03_semantics/baseline', 'output': ROOT / 'results/new_runs/03_semantics/entity_clean/numeric34', 'bootstrap': 1000, 'seed': 42}
    values.update(overrides)
    return SimpleNamespace(**values)

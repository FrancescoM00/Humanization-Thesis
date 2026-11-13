"""Esporta tabelle e grafici PNG dalle metriche semantiche eseguite."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _interval(values, rng, n_bootstrap=1000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    draws = rng.choice(values, (n_bootstrap, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [.025, .975]))


def export_semantic_tables(pairs, output, seed=42):
    output = Path(output)
    manifest = json.loads((output / 'preparation_manifest.json').read_text())
    metadata = []
    for pair in pairs:
        item = {key: pair.get(key) for key in ['id', 'source_id', 'domain', 'generator', 'perturbation', 'dataset']}
        item['comparison'] = pair.get('metadata', {}).get('comparison', 'controlled')
        item['control_for'] = pair.get('metadata', {}).get('control_for')
        item['original_whitespace_words'] = len(pair['original'].split())
        item['modified_whitespace_words'] = len(pair['modified'].split())
        item['length_ratio'] = item['modified_whitespace_words'] / item['original_whitespace_words']
        item['text_changed'] = pair['original'] != pair['modified']
        item['expected_length_direction_observed'] = pair.get('metadata', {}).get('expected_length_direction_observed')
        metadata.append(item)
    frame = pd.DataFrame(metadata)
    expected_hash = manifest['pairs_sha256']
    for name in ['lexical', 'embedding', 'bertscore', 'nli']:
        path = output / (name + '.csv')
        run_path = output / (name + '_run.json')
        if not path.is_file():
            continue
        run = json.loads(run_path.read_text())
        if run['pairs_sha256'] != expected_hash:
            raise ValueError(f'Stale {name} scores from different input pairs; rerun this stage')
        scores = pd.read_csv(path).drop(columns=['pair_id'], errors='ignore')
        if scores.id.duplicated().any() or set(scores.id) != set(frame.id):
            raise ValueError(f'{name}: scores do not cover exactly the prepared pairs')
        frame = frame.merge(scores, on='id', validate='one_to_one')
    frame.to_csv(output / 'scores.csv', index=False)
    metric_columns = [c for c in frame.select_dtypes(include='number') if
                      c.startswith(('lexical_', 'embedding_cosine', 'local_', 'bert_chunk_', 'nli_forward_', 'nli_reverse_'))
                      and not c.endswith(('_chunks', '_tokens', '_threshold'))]
    rng = np.random.default_rng(seed)
    rows = []
    for key, part in frame.groupby(['dataset', 'perturbation', 'comparison'], sort=True):
        for metric in metric_columns:
            valid = part.dropna(subset=[metric])
            if not len(valid):
                continue

            values = valid.groupby('source_id')[metric].mean()
            low, high = _interval(values.to_numpy(), rng)
            rows.append(dict(zip(['dataset', 'perturbation', 'comparison'], key), metric=metric,
                             n_pairs=len(valid), n_sources=len(values), mean=values.mean(), median=values.median(),
                             ci_low=low, ci_high=high))
    summary = pd.DataFrame(rows)
    summary.to_csv(output / 'metric_summary.csv', index=False)
    deltas = []
    for prefix in ['nli_', 'local_nli_']:
        a, b = prefix + 'forward_mean', prefix + 'reverse_mean'
        if a in frame and b in frame:
            for key, part in frame.groupby(['dataset', 'perturbation', 'comparison'], sort=True):
                values = (part[a] - part[b]).groupby(part.source_id).mean().dropna()
                if len(values):
                    low, high = _interval(values.to_numpy(), rng)
                    deltas.append(dict(zip(['dataset', 'perturbation', 'comparison'], key),
                                       metric=prefix + 'forward_minus_reverse', n_sources=len(values),
                                       mean=values.mean(), ci_low=low, ci_high=high))
    pd.DataFrame(deltas).to_csv(output / 'directional_deltas.csv', index=False)
    paired_deltas = []
    identity = frame[(frame.dataset == 'controlled') & (frame.perturbation == 'identity')].set_index('source_id')
    local_identity = frame[frame.dataset == 'local_control'].set_index('control_for')
    for condition, part in frame[(frame.dataset == 'controlled') & (frame.perturbation != 'identity')].groupby('perturbation'):
        for metric in metric_columns:
            if metric.startswith('local_'):
                baseline = part.id.map(local_identity[metric]) if metric in local_identity else None
            else:
                baseline = part.source_id.map(identity[metric]) if metric in identity else None
            if baseline is None:
                continue
            values = (part[metric] - baseline).groupby(part.source_id).mean().dropna()
            if len(values):
                low, high = _interval(values.to_numpy(), rng)
                paired_deltas.append({'perturbation': condition, 'metric': metric,
                                      'n_sources': len(values), 'mean_delta_from_same_source_identity': values.mean(),
                                      'ci_low': low, 'ci_high': high})
    pd.DataFrame(paired_deltas).to_csv(output / 'paired_identity_deltas.csv', index=False)
    control = frame[frame.dataset == 'controlled']
    names = ['identity', 'negation', 'entity_substitution', 'sentence_shuffle', 'truncate', 'random_deletion']
    if 'lexical_rouge_l_f1' in control:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, column, title in zip(axes, ['lexical_rouge_l_f1', 'lexical_bleu'], ['ROUGE-L F1', 'BLEU (0–1)']):
            groups = [control.loc[control.perturbation == name, column].dropna().to_numpy() for name in names]
            ax.boxplot(groups, tick_labels=names, showfliers=False)
            ax.tick_params(axis='x', rotation=35, labelsize=8)
            ax.set_ylim(0, 1.03)
            ax.set_title(title)
        fig.suptitle('200 testi macchina: perturbazioni controllate, documenti interi')
        fig.tight_layout()
        fig.savefig(output / 'lexical_perturbations.png', dpi=170)
        plt.close(fig)
    opai_rows = []
    for (condition, comparison), part in frame[frame.dataset == 'opai'].groupby(['perturbation', 'comparison']):
        row = {'operation': condition, 'comparison': comparison, 'n_pairs': len(part),
               'n_sources': part.source_id.nunique(), 'changed': int(part.text_changed.sum()),
               'shorter': int((part.length_ratio < 1).sum()), 'longer': int((part.length_ratio > 1).sum()),
               'median_length_ratio': part.length_ratio.median()}
        for column in ['embedding_cosine', 'bert_chunk_f1', 'nli_forward_mean', 'nli_reverse_mean']:
            if column in part:
                row[column] = part.groupby('source_id')[column].mean().mean()
        if 'nli_forward_mean' in part:
            values = (part.nli_forward_mean - part.nli_reverse_mean).groupby(part.source_id).mean()
            row['forward_minus_reverse'] = values.mean()
            row['delta_ci_low'], row['delta_ci_high'] = _interval(values, rng)
            row['pair_fraction_forward_gt_reverse'] = float((part.nli_forward_mean > part.nli_reverse_mean).mean())
        opai_rows.append(row)
    opai_table = pd.DataFrame(opai_rows)
    opai_table.to_csv(output / 'opai_asymmetry.csv', index=False)
    paired = pd.DataFrame(paired_deltas)
    local_plot_metrics = ['local_embedding_cosine', 'local_bert_f1', 'local_nli_forward_mean']
    if len(paired) and set(local_plot_metrics).issubset(set(paired.metric)):
        fig, ax = plt.subplots(figsize=(8, 4))
        for offset, condition, color in [(-.16, 'negation', '#167c80'), (.16, 'entity_substitution', '#d78c3c')]:
            part = paired[(paired.perturbation == condition) & paired.metric.isin(local_plot_metrics)].set_index('metric').reindex(local_plot_metrics)
            values = part.mean_delta_from_same_source_identity
            pos = np.arange(3) + offset
            ax.bar(pos, values, width=.3, color=color, label=condition)
            ax.errorbar(pos, values, yerr=[np.maximum(values-part.ci_low, 0), np.maximum(part.ci_high-values, 0)],
                        fmt='none', ecolor='black', linewidth=.8, capsize=3)
        ax.axhline(0, color='gray', linewidth=.7)
        ax.set_xticks(np.arange(3), ['Coseno MiniLM', 'BERTScore', 'Supporto NLI'])
        ax.set_ylabel('Differenza dalla stessa frase originale contro se stessa')
        ax.set_title('Frase modificata: variazione rispetto al controllo identità\nIC 95% bootstrap per fonte, 200 testi')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / 'local_semantic_changes.png', dpi=180)
        plt.close(fig)
    if {'embedding_cosine', 'bert_chunk_f1', 'nli_forward_mean'}.issubset(control.columns):
        columns = ['lexical_rouge_l_f1', 'lexical_bleu', 'embedding_cosine', 'bert_chunk_f1', 'nli_forward_mean', 'nli_reverse_mean']
        pivot = control.groupby('perturbation')[columns].mean().reindex(names)
        fig, ax = plt.subplots(figsize=(10, 4.4))
        im = ax.imshow(pivot, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
        ax.set_yticks(range(len(pivot)), pivot.index)
        ax.set_xticks(range(len(columns)), ['ROUGE-L', 'BLEU', 'Coseno', 'BERTScore\nchunk', 'NLI→\nmedia', 'NLI←\nmedia'])
        for i in range(len(pivot)):
            for j in range(len(columns)):
                value = pivot.iloc[i, j]
                ax.text(j, i, f'{value:.3f}', ha='center', va='center', color='white' if value > .65 else 'black', fontsize=9)
        ax.set_title('Documenti interi: medie su 200 fonti per condizione')
        fig.colorbar(im, ax=ax, shrink=.8)
        fig.tight_layout()
        fig.savefig(output / 'semantic_perturbations.png', dpi=180)
        plt.close(fig)
    if len(opai_table) and 'forward_minus_reverse' in opai_table:
        adjacent = opai_table[opai_table.comparison == 'adjacent_operation'].copy()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        positions = np.arange(len(adjacent))
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.errorbar(adjacent.forward_minus_reverse, positions,
                    xerr=[np.maximum(adjacent.forward_minus_reverse-adjacent.delta_ci_low, 0),
                          np.maximum(adjacent.delta_ci_high-adjacent.forward_minus_reverse, 0)],
                    fmt='o', capsize=5, color='#167c80')
        ax.set_yticks(positions, adjacent.operation)
        ax.set_xlabel('Supporto NLI originale→modificato meno modificato→originale\n(IC 95% bootstrap per documento sorgente)')
        ax.set_title('OpAI-Bench: asimmetria delle versioni adiacenti')
        fig.tight_layout()
        fig.savefig(output / 'opai_asymmetry.png', dpi=180)
        plt.close(fig)

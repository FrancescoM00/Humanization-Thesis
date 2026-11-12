"""Export continuous semantic scores and per-humanizer distributions.

Consumes already computed ID-keyed score tables. No threshold, verdict,
calibration to a previous editor, model download or neural inference is used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from humanization.evaluation.semantic_fairness import merge_score_tables, metric_quality
from humanization.semantics.metrics import _sentences

def _argument_error(message):
    raise ValueError(message)

def run(args):
    pairs = [json.loads(x) for x in args.pairs.read_text().splitlines() if x.strip()]
    if not pairs:
        raise ValueError('The supplied pair file is empty')
    if len({p['id'] for p in pairs}) != len(pairs):
        raise ValueError('Duplicate input pair IDs')
    metadata = []
    for p in pairs:
        metadata.append({'id': p['id'], 'source_group': p.get('source_group_id') or p.get('source_id') or hashlib.sha256(p['original'].encode()).hexdigest(), 'humanizer': p.get('humanizer') or p.get('generator') or 'unspecified', 'dataset': p.get('dataset', 'humanizer_outputs'), 'operation': p.get('perturbation') or p.get('metadata', {}).get('operation', 'rewrite'), 'original_sentences': len(_sentences(p['original'])), 'rewritten_sentences': len(_sentences(p['modified']))})
    frame = pd.DataFrame(metadata)
    scores = merge_score_tables(args.score)
    if not set(frame.id).issubset(set(scores.id)):
        raise ValueError('Scores do not cover all supplied pairs')
    extras = [c for c in scores if c == 'id' or c not in frame]
    frame = frame.merge(scores[extras], on='id', validate='one_to_one')
    quality, definitions = metric_quality(frame)
    guard_columns = [c for c in frame if c.startswith('guard_') and pd.api.types.is_numeric_dtype(frame[c])]
    exported = pd.concat([frame[list(metadata[0])], quality, frame[guard_columns]], axis=1)
    args.output.mkdir(parents=True, exist_ok=True)
    exported.to_csv(args.output / 'continuous_scores.csv', index=False)
    definitions.to_csv(args.output / 'score_definitions.csv', index=False)
    rows = []
    for keys, part in exported.groupby(['dataset', 'humanizer', 'operation'], dropna=False):
        for metric in list(quality) + guard_columns:
            valid = part.dropna(subset=[metric])
            if valid.empty:
                continue
            values = valid.groupby('source_group')[metric].mean()
            q = values.quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
            rows.append(dict(zip(['dataset', 'humanizer', 'operation'], keys), metric=metric, n_pairs=len(valid), n_sources=len(values), mean=values.mean(), **dict(zip(['q05', 'q10', 'q25', 'median', 'q75', 'q90', 'q95'], q.to_numpy()))))
    pd.DataFrame(rows).to_csv(args.output / 'humanizer_distributions.csv', index=False)
    manifest = {'n_pairs': len(exported), 'humanizers': sorted(exported.humanizer.unique()), 'humanizer_label_source': 'explicit humanizer field, otherwise generator/editor; OpAI editors are not DIPPER evaluations', 'primary_output': 'continuous model support/similarity and lexical changes', 'threshold_applied': False, 'semantic_verdict_created': False, 'distribution_weighting': 'one vote per source within each humanizer/condition', 'length_normalization': 'No previous-editor calibration is applied implicitly; compare lengths and calibrate explicitly on new-domain references.', 'scores_are_not_probabilities_of_document_correctness': True, 'guard_columns_are_descriptive_counts_or_changes_not_calibrated_risks': True, 'pairs_sha256': hashlib.sha256(args.pairs.read_bytes()).hexdigest()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'pairs': None, 'score': None, 'output': None}
    values.update(overrides)
    return SimpleNamespace(**values)

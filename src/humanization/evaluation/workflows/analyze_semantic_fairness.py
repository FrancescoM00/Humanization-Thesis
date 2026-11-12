"""Confronta similarità e NLI aggregati allo stesso modo, errori distribuiti e lunghezza."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
from humanization.evaluation.semantic_fairness import analyze_fairness, merge_score_tables

def _argument_error(message):
    raise ValueError(message)

def run(args):
    pairs = [json.loads(x) for x in args.pairs.read_text().splitlines() if x.strip()]
    if args.distributed_pairs.is_file():
        pairs += [json.loads(x) for x in args.distributed_pairs.read_text().splitlines() if x.strip()]
    for path in args.extra_pairs:
        pairs += [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    defaults = [ROOT / 'results/published/03_semantics/fairness/aligned_old/aligned_cosine.csv', ROOT / 'results/published/03_semantics/fairness/aligned_old/aligned_bertscore.csv']
    tables = list(dict.fromkeys([args.old_scores] + [x for x in defaults if x.is_file()] + args.score))
    scores = merge_score_tables(tables)
    result = analyze_fairness(pairs, scores, args.output, seed=args.seed, n_bootstrap=args.bootstrap, allow_partial=args.allow_partial)
    print(json.dumps(result, indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'pairs': ROOT / 'data/processed/semantic_extension/pairs.jsonl', 'distributed_pairs': ROOT / 'data/processed/semantic_fairness/distributed_pairs.jsonl', 'extra_pairs': [], 'old_scores': ROOT / 'results/published/03_semantics/extension/scores.csv', 'score': [], 'output': ROOT / 'results/new_runs/03_semantics/fairness', 'bootstrap': 1000, 'seed': 42, 'allow_partial': False}
    values.update(overrides)
    return SimpleNamespace(**values)

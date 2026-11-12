"""Evaluate positive semantic controls and select an NLI aggregator on calibration."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import pandas as pd
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
from humanization.evaluation.semantic import analyze_semantic_extension

def _argument_error(message):
    raise ValueError(message)

def run(args):
    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    scores = pd.read_csv(args.scores)
    rescaled = args.rescaled or args.scores.parent / 'bertscore_rescaled.csv'
    if rescaled.is_file():
        extra = pd.read_csv(rescaled)
        names = ['id'] + [name for name in extra if name not in scores and ('rescal' in name or ('baseline' in name and 'distance' in name))]
        if len(names) > 1:
            scores = scores.merge(extra[names], on='id', how='left', validate='one_to_one')
    review_path = args.review or args.scores.parent / 'review40.csv'
    review = pd.read_csv(review_path) if review_path.is_file() else None
    result = analyze_semantic_extension(pairs, scores, args.output, seed=args.seed, n_bootstrap=args.bootstrap, target_fpr=args.target_fpr, review=review)
    print(json.dumps(result, indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'pairs': ROOT / 'data/semantic_extension/pairs.jsonl', 'scores': ROOT / 'results/published/03_semantics/extension/scores.csv', 'output': ROOT / 'results/new_runs/03_semantics/extension', 'rescaled': None, 'review': None, 'seed': 42, 'bootstrap': 1000, 'target_fpr': 0.1}
    values.update(overrides)
    return SimpleNamespace(**values)

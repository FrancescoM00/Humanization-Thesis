#!/usr/bin/env python3
"""Evaluate positive semantic controls and select an NLI aggregator on calibration."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
from semantics.aggregator_analysis import analyze_semantic_extension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, default=ROOT / 'data/semantic_extension/pairs.jsonl')
    parser.add_argument('--scores', type=Path, default=ROOT / 'results/03_semantics/extension/scores.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/03_semantics/extension')
    parser.add_argument('--rescaled', type=Path, help='Optional BERTScore rescaled CSV; otherwise discovered beside scores')
    parser.add_argument('--review', type=Path, help='Optional textual review CSV; otherwise review40.csv beside scores')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--target-fpr', type=float, default=.1)
    args = parser.parse_args()
    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    scores = pd.read_csv(args.scores)
    rescaled = args.rescaled or args.scores.parent / 'bertscore_rescaled.csv'
    if rescaled.is_file():
        extra = pd.read_csv(rescaled)
        names = ['id'] + [name for name in extra if name not in scores and
                         ('rescal' in name or ('baseline' in name and 'distance' in name))]
        if len(names) > 1:
            scores = scores.merge(extra[names], on='id', how='left', validate='one_to_one')
    review_path = args.review or args.scores.parent / 'review40.csv'
    review = pd.read_csv(review_path) if review_path.is_file() else None
    result = analyze_semantic_extension(pairs, scores, args.output, seed=args.seed,
                                        n_bootstrap=args.bootstrap, target_fpr=args.target_fpr, review=review)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

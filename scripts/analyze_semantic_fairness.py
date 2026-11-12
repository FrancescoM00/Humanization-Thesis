#!/usr/bin/env python3
"""Compare equally aggregated similarities/NLI, distributed errors and length."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
from semantics.fairness_analysis import analyze_fairness, merge_score_tables


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,default=ROOT/'data/semantic_extension/pairs.jsonl')
    p.add_argument('--distributed-pairs',type=Path,default=ROOT/'data/semantic_fairness/distributed_pairs.jsonl')
    p.add_argument('--extra-pairs',type=Path,action='append',default=[],help='Additional frozen pair JSONL, e.g. expanded positive and distributed source panels.')
    p.add_argument('--old-scores',type=Path,default=ROOT/'results/03_semantics/extension/scores.csv')
    p.add_argument('--score',type=Path,action='append',default=[],help='Extra ID-keyed score CSV; may repeat, including distributed NLI/embedding and aligned similarities.')
    p.add_argument('--output',type=Path,default=ROOT/'runs/03_semantics/fairness')
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--allow-partial',action='store_true',help='Diagnostic only: permit incomplete metrics; final runs must omit this flag.')
    args=p.parse_args()
    pairs=[json.loads(x) for x in args.pairs.read_text().splitlines() if x.strip()]
    if args.distributed_pairs.is_file():
        pairs += [json.loads(x) for x in args.distributed_pairs.read_text().splitlines() if x.strip()]
    for path in args.extra_pairs:
        pairs += [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    defaults=[ROOT/'results/03_semantics/fairness/aligned_old/aligned_cosine.csv',
              ROOT/'results/03_semantics/fairness/aligned_old/aligned_bertscore.csv']
    tables=list(dict.fromkeys([args.old_scores]+[x for x in defaults if x.is_file()]+args.score))
    scores=merge_score_tables(tables)
    result=analyze_fairness(pairs,scores,args.output,seed=args.seed,n_bootstrap=args.bootstrap,allow_partial=args.allow_partial)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

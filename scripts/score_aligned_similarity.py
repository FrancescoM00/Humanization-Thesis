#!/usr/bin/env python3
"""Compute aligned cosine/BERTScore maxima and mean/q10/min per direction."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'.cache'))

from semantics.aligned_similarity import aligned_cosine_scores, aligned_bert_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, default=ROOT/'data/semantic_extension/pairs.jsonl')
    parser.add_argument('--alignment', type=Path, default=ROOT/'data/semantic_extension/alignment.json')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/03_semantics/fairness/aligned_old')
    parser.add_argument('--stage', choices=['cosine', 'bertscore'], required=True)
    parser.add_argument('--model', type=Path, default=ROOT/'data/models/bertscore')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cpu', help='Explicit PyTorch device for BERTScore: cpu, cuda or cuda:0')
    args = parser.parse_args()
    pairs = [json.loads(x) for x in args.pairs.read_text().splitlines() if x.strip()]
    alignment = json.loads(args.alignment.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    stem = 'aligned_cosine' if args.stage == 'cosine' else 'aligned_bertscore'
    def progress(done, total):
        (args.output/(stem+'_progress.json')).write_text(json.dumps({
            'completed_pairs':done,'total_pairs':total,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
    if args.stage == 'cosine':
        frame, units = aligned_cosine_scores(pairs, alignment, progress_callback=progress)
    else:
        frame, units = aligned_bert_scores(pairs, alignment, args.model, n_threads=args.threads,
                                           batch_size=args.batch_size, progress_callback=progress, device=args.device)
    frame.to_csv(args.output/(stem+'.csv'), index=False)
    (args.output/(stem+'_units.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in units))
    metadata = {'stage':stem,'status':'executed','n_pairs':len(frame),
                'pairs_sha256':hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
                'alignment_sha256':hashlib.sha256(args.alignment.read_bytes()).hexdigest(),
                'elapsed_seconds':time.monotonic()-start,
                'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                'threads':args.threads,'batch_size':args.batch_size,
                'device':args.device if args.stage=='bertscore' else 'cpu',
                'model':str(args.model) if args.stage=='bertscore' else 'cached MiniLM sentence similarities',
                'unit_protocol':'exact existing NLI alignment chunks; no silent truncation',
                'alignment':'maximum across all premise units; both directions',
                'aggregation':'unweighted mean/q10/min; worst direction for bidirectional',
                'code_sha256':hashlib.sha256((ROOT/'src/semantics/aligned_similarity.py').read_bytes()).hexdigest()}
    (args.output/(stem+'_run.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__ == '__main__':
    main()

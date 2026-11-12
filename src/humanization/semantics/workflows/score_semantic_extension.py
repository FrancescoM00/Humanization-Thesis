"""Calcola gli score delle nuove coppie semantiche conservando quelli delle coppie invariate già calcolate."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from humanization import resources as resource
import sys
import time
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

def _argument_error(message):
    raise ValueError(message)

def run(args):
    import pandas as pd
    from humanization.semantics.metrics import lexical_scores, embedding_scores, bert_scores
    start = time.monotonic()
    pairs_path = args.data_dir / 'pairs.jsonl'
    pairs = [json.loads(x) for x in pairs_path.read_text().splitlines() if x.strip()]
    previous_pairs = {p['id']: p for p in map(json.loads, (args.previous_data / 'pairs.jsonl').read_text().splitlines())}
    previous = pd.read_csv(args.previous_output / f'{args.stage}.csv').set_index('id', drop=False)
    reused, fresh = ([], [])
    for pair in pairs:
        old = previous_pairs.get(pair['id'])
        if old is not None and pair['id'] in previous.index:
            for key in ['original', 'modified', 'target_original', 'target_modified']:
                if pair.get(key) != old.get(key):
                    raise ValueError(f"Changed text behind reused ID: {pair['id']}, {key}")
            reused.append(pair['id'])
        else:
            fresh.append(pair)
    args.output.mkdir(parents=True, exist_ok=True)

    def progress(completed, total):
        (args.output / f'{args.stage}_progress.json').write_text(json.dumps({'reused_pairs': len(reused), 'completed_new_pairs': completed, 'total_new_pairs': total, 'elapsed_seconds': time.monotonic() - start}, indent=2) + '\n')
    if args.stage == 'lexical':
        scored = lexical_scores(fresh)
    elif args.stage == 'embedding':
        scored, new_alignment = embedding_scores(fresh, ROOT / 'data/models/embedding', n_threads=args.threads, progress_callback=progress)
        old_alignment = json.loads((args.previous_data / 'alignment.json').read_text())
        alignment = {key: old_alignment[key] for key in reused}
        alignment.update(new_alignment)
        (args.data_dir / 'alignment.json').write_text(json.dumps(alignment, ensure_ascii=False))
    else:
        scored = bert_scores(fresh, ROOT / 'data/models/bertscore', batch_size=4, n_threads=args.threads, progress_callback=progress)
    combined = pd.concat([previous.loc[reused].reset_index(drop=True), scored], ignore_index=True)
    combined = combined.set_index('id', drop=False).loc[[p['id'] for p in pairs]].reset_index(drop=True)
    combined.to_csv(args.output / f'{args.stage}.csv', index=False)
    versions = {}
    for name in ['torch', 'transformers', 'sentence-transformers', 'bert-score', 'rouge-score', 'sacrebleu']:
        versions[name] = importlib.metadata.version(name)
    metadata = {'stage': args.stage, 'status': 'executed', 'n_pairs': len(combined), 'n_reused_pairs': len(reused), 'n_newly_scored_pairs': len(fresh), 'pairs_sha256': hashlib.sha256(pairs_path.read_bytes()).hexdigest(), 'elapsed_seconds': time.monotonic() - start, 'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'device': 'cpu', 'threads': args.threads, 'versions': versions, 'reused_results': str(args.previous_output), 'previous_score_sha256': hashlib.sha256((args.previous_output / f'{args.stage}.csv').read_bytes()).hexdigest(), 'metrics_code_sha256': hashlib.sha256((ROOT / 'src/humanization/semantics/metrics.py').read_bytes()).hexdigest()}
    (args.output / f'{args.stage}_run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'stage': None, 'data_dir': ROOT / 'data/processed/semantic_extension', 'output': ROOT / 'results/new_runs/03_semantics/extension', 'previous_data': ROOT / 'data/processed/semantic_baseline', 'previous_output': ROOT / 'results/published/03_semantics/baseline', 'threads': 4}
    values.update(overrides)
    return SimpleNamespace(**values)

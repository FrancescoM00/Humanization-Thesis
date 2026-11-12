"""Prepare subtle semantic alterations from existing MGTBench machine texts."""
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
from humanization.semantics.subtle_perturbations import generate_subtle_pairs

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def _argument_error(message):
    raise ValueError(message)

def run(args):
    originals = [pair for pair in read_jsonl(args.original_pairs) if pair['dataset'] == 'controlled' and pair['perturbation'] == 'identity']
    preferred = {pair['source_id'] for pair in originals}
    records = read_jsonl(args.input)
    pairs, manifest = generate_subtle_pairs(records, n_per_type=args.n_per_type, seed=args.seed, preferred_source_ids=preferred)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join((json.dumps(pair, ensure_ascii=False) + '\n' for pair in pairs)))
    manifest.update({'source_file': str(args.input), 'source_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(), 'pairs_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(), 'code_sha256': hashlib.sha256((ROOT / 'src/humanization/semantics/subtle_perturbations.py').read_bytes()).hexdigest()})
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'screening'}, indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'input': ROOT / 'data/processed/generated/baseline/samples.jsonl', 'original_pairs': ROOT / 'data/processed/semantic_baseline/pairs.jsonl', 'output': ROOT / 'data/processed/semantic_extension/subtle_pairs.jsonl', 'manifest': ROOT / 'results/new_runs/03_semantics/extension/subtle_manifest.json', 'n_per_type': 50, 'seed': 42}
    values.update(overrides)
    return SimpleNamespace(**values)

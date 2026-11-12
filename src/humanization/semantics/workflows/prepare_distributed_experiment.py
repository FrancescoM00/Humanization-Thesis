"""Prepare nested distributed semantic corruptions without neural inference."""
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
from humanization.semantics.distributed_perturbations import generate_distributed_pairs

def _argument_error(message):
    raise ValueError(message)

def run(args):
    if args.output.exists() and (not args.overwrite):
        raise FileExistsError(f'Preserving frozen inputs {args.output}; use --overwrite intentionally')
    positives = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]

    def progress(n, total):
        print(json.dumps({'phase': 'spacy_applicability', 'completed': n, 'total': total}), flush=True)
    pairs, manifest = generate_distributed_pairs(positives, n_per_family=args.n_per_family, seed=args.seed, progress_callback=progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join((json.dumps(pair, ensure_ascii=False) + '\n' for pair in pairs)))
    manifest.update({'input_file': str(args.input), 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(), 'pairs_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(), 'code_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [ROOT / 'src/humanization/semantics/distributed_perturbations.py', Path(__file__).resolve()]}})
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: value for key, value in manifest.items() if key != 'screening'}, indent=2), flush=True)

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'input': ROOT / 'data/processed/semantic_extension/positive_pairs.jsonl', 'output': ROOT / 'data/processed/semantic_fairness/distributed_pairs.jsonl', 'manifest': ROOT / 'results/new_runs/03_semantics/fairness/distributed_manifest.json', 'n_per_family': 60, 'seed': 20261105, 'overwrite': False}
    values.update(overrides)
    return SimpleNamespace(**values)

#!/usr/bin/env python3
"""Prepare subtle semantic alterations from existing MGTBench machine texts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from semantics.subtle_perturbations import generate_subtle_pairs


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/clean/samples.jsonl')
    parser.add_argument('--original-pairs', type=Path, default=ROOT/'data/semantic_full/pairs.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'data/semantic_extension/subtle_pairs.jsonl')
    parser.add_argument('--manifest', type=Path, default=ROOT/'runs/03_semantics/extension/subtle_manifest.json')
    parser.add_argument('--n-per-type', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    originals = [pair for pair in read_jsonl(args.original_pairs)
                 if pair['dataset'] == 'controlled' and pair['perturbation'] == 'identity']
    preferred = {pair['source_id'] for pair in originals}
    records = read_jsonl(args.input)
    pairs, manifest = generate_subtle_pairs(records, n_per_type=args.n_per_type,
                                          seed=args.seed, preferred_source_ids=preferred)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(pair, ensure_ascii=False)+'\n' for pair in pairs))
    manifest.update({'source_file': str(args.input), 'source_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                     'pairs_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                     'code_sha256': hashlib.sha256((ROOT/'src/semantics/subtle_perturbations.py').read_bytes()).hexdigest()})
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'screening'}, indent=2))


if __name__ == '__main__':
    main()

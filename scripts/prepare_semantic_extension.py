#!/usr/bin/env python3
"""Prepare OpAI positive controls, matched corruptions and subtle challenges."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from semantics.positive_controls import (
    assign_source_splits, load_local_opai_trajectories,
    make_matched_negative_controls, select_positive_controls,
)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opai-dir', type=Path, default=ROOT / 'data/opai')
    parser.add_argument('--previous-pairs', type=Path, default=ROOT / 'data/semantic_full/pairs.jsonl')
    parser.add_argument('--subtle-pairs', type=Path, default=ROOT / 'data/semantic_extension/subtle_pairs.jsonl')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/semantic_extension')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/03_semantics/extension')
    parser.add_argument('--n', type=int, default=100, help='Common OpAI trajectories; three positive pairs each')
    parser.add_argument('--matched-subtle-per-type', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--without-subtle', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    pair_path = args.data_dir / 'pairs.jsonl'
    if pair_path.exists() and not args.overwrite:
        raise FileExistsError(f'Preserving completed input {pair_path}; use --overwrite for intentional regeneration')
    if not args.without_subtle and not args.subtle_pairs.is_file():
        raise FileNotFoundError(f'Prepare subtle perturbations first: {args.subtle_pairs}')
    trajectories, provenance = load_local_opai_trajectories(args.opai_dir)
    positives, positive_manifest = select_positive_controls(trajectories, args.n, args.seed)
    negatives, negative_manifest = make_matched_negative_controls(positives, args.seed, n_per_subtle=args.matched_subtle_per_type)
    previous_all = read_jsonl(args.previous_pairs)
    previous = [pair for pair in previous_all if pair['dataset'] in {'controlled', 'local_control', 'manual_sanity'}]
    subtle = [] if args.without_subtle else read_jsonl(args.subtle_pairs)
    pairs = previous + positives + negatives + subtle
    # Reassign source splits over the complete union, so every corruption,
    # control and editor of one source has a single immutable split.
    assign_source_splits(pairs, args.seed)
    assert len({pair['id'] for pair in pairs}) == len(pairs)
    assert len(previous) == 1607, 'Unexpected previous cohort; inspect before rerunning this registered extension'
    write_jsonl(pair_path, pairs)
    write_jsonl(args.data_dir / 'positive_pairs.jsonl', positives)
    write_jsonl(args.data_dir / 'matched_negative_pairs.jsonl', negatives)
    write_json(args.output / 'positive_controls_manifest.json', {**positive_manifest, 'source_metadata': provenance})
    write_json(args.output / 'matched_negative_manifest.json', negative_manifest)
    old_manifest = json.loads((ROOT / 'results/03_semantics/baseline/preparation_manifest.json').read_text())
    manifest = {
        **old_manifest,
        'source_file': str(args.previous_pairs),
        'source_sha256': hashlib.sha256(args.previous_pairs.read_bytes()).hexdigest(),
        'pairs_sha256': hashlib.sha256(pair_path.read_bytes()).hexdigest(),
        'seed': args.seed, 'n_pairs': len(pairs),
        'opai': {'status': 'prepared', 'n_pairs': len(positives) + len(negatives),
                 'n_positive_pairs': len(positives), 'n_matched_negatives': len(negatives)},
        'datasets': dict(Counter(pair['dataset'] for pair in pairs)),
        'condition_counts': dict(Counter(pair['dataset'] + ':' + pair['perturbation'] for pair in pairs)),
        'evaluation_split_pair_counts': dict(Counter(pair['evaluation_split'] for pair in pairs)),
        'evaluation_split_source_counts': dict(Counter({pair['source_group_id']: pair['evaluation_split'] for pair in pairs}.values())),
        'positive_labels_are_presumed_not_gold': True,
        'negative_labels_are_intended_not_gold': True,
        'metric_score_independent_selection': True,
        'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [ROOT / 'src/semantics/positive_controls.py', Path(__file__).resolve()]},
    }
    write_json(args.output / 'preparation_manifest.json', manifest)
    # Existing generic reporting expects this description of original eligibility.
    old_eligibility = ROOT / 'results/03_semantics/baseline/eligibility_manifest.json'
    if old_eligibility.is_file():
        write_json(args.output / 'eligibility_manifest.json', json.loads(old_eligibility.read_text()))
    print(json.dumps({key: manifest[key] for key in ['n_pairs', 'datasets', 'opai', 'evaluation_split_pair_counts', 'evaluation_split_source_counts', 'pairs_sha256']}, indent=2))


if __name__ == '__main__':
    main()

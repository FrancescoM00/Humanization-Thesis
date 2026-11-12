#!/usr/bin/env python3
"""Prepare frozen Kaggle panels, without running generators or detectors."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from humanizers.cohort import (fingerprint, paired_samples, read_jsonl,
                               reconstruct_samples, save_cohorts, select_cohorts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', type=Path, help='Existing length-matched samples; otherwise reconstruct from raw CSVs')
    p.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-cell', type=int, default=2)
    p.add_argument('--calibration-per-domain', type=int, default=100)
    p.add_argument('--reserve-per-cell', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    metadata = ROOT / 'results/02_stylometry/length_matched/features.csv'
    historic = ROOT / 'results/02_stylometry/length_matched/data_manifest.json'
    samples = read_jsonl(args.samples) if args.samples else reconstruct_samples(args.raw_dir, metadata, historic)
    panels = select_cohorts(paired_samples(samples), per_cell=args.per_cell,
                           calibration_per_domain=args.calibration_per_domain,
                           reserve_per_cell=args.reserve_per_cell, seed=args.seed)
    provenance = {'seed': args.seed, 'per_cell': args.per_cell,
                  'reserve_per_cell': args.reserve_per_cell,
                  'calibration_per_domain': args.calibration_per_domain,
                  'metadata_sha256': fingerprint(metadata),
                  'historical_manifest_sha256': fingerprint(historic),
                  'reconstructed_text_ids_verified': args.samples is None}
    if args.samples:
        provenance['samples_sha256'] = fingerprint(args.samples)
    manifest = save_cohorts(panels, args.output, provenance)
    print({k: v['test_sources'] for k, v in manifest['panels'].items()}, flush=True)


if __name__ == '__main__':
    main()

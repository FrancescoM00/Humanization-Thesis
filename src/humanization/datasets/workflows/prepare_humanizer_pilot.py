"""Prepara gruppi di testi preselezionati, senza eseguire generatori o detector."""
import argparse
from pathlib import Path
import sys
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from humanization.datasets.cohort import fingerprint, paired_samples, read_jsonl, reconstruct_samples, save_cohorts, select_cohorts

def _argument_error(message):
    raise ValueError(message)

def run(args):
    metadata = ROOT / 'results/published/02_stylometry/length_matched/features.csv'
    historic = ROOT / 'results/published/02_stylometry/length_matched/data_manifest.json'
    samples = read_jsonl(args.samples) if args.samples else reconstruct_samples(args.raw_dir, metadata, historic)
    panels = select_cohorts(paired_samples(samples), per_cell=args.per_cell, calibration_per_domain=args.calibration_per_domain, reserve_per_cell=args.reserve_per_cell, seed=args.seed)
    provenance = {'seed': args.seed, 'per_cell': args.per_cell, 'reserve_per_cell': args.reserve_per_cell, 'calibration_per_domain': args.calibration_per_domain, 'metadata_sha256': fingerprint(metadata), 'historical_manifest_sha256': fingerprint(historic), 'reconstructed_text_ids_verified': args.samples is None}
    if args.samples:
        provenance['samples_sha256'] = fingerprint(args.samples)
    manifest = save_cohorts(panels, args.output, provenance)
    print({k: v['test_sources'] for k, v in manifest['panels'].items()}, flush=True)

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'samples': None, 'raw_dir': ROOT / 'data/raw/mgtbench', 'output': None, 'per_cell': 2, 'calibration_per_domain': 100, 'reserve_per_cell': 6, 'seed': 42}
    values.update(overrides)
    return SimpleNamespace(**values)

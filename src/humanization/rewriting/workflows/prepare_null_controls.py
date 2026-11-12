"""Prepare detector null controls; optionally compare already saved scores."""
import argparse
import csv
import json
from pathlib import Path
import sys
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from humanization.rewriting.controls import prepare, read_jsonl, summarize_null_controls

def _argument_error(message):
    raise ValueError(message)

def run(args):
    if args.detector_scores and (not args.thresholds):
        _argument_error('--thresholds is required with --detector-scores')
    manifest = prepare(args.cohort, args.rewrites, args.output, seed=args.seed, noise_max_edits=args.noise_max_edits, include_rewrites=args.include_rewrites, only=args.only)
    if args.detector_scores:
        references = [r for path in args.rewrites for r in read_jsonl(path) if not args.only or r['humanizer'] in args.only]
        scores = []
        for path in args.detector_scores:
            with path.open(newline='') as handle:
                scores.extend(csv.DictReader(handle))
        summary = summarize_null_controls(read_jsonl(args.cohort), references, read_jsonl(args.output / 'rewrites.jsonl'), scores, json.loads(args.thresholds.read_text()), lower_is_ai=set(args.lower_is_ai))
        with (args.output / 'null_control_comparison.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(summary)
    print(json.dumps(dict(n_rows=manifest['n_rows'], conditions=manifest['conditions'], output=str(args.output)), indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'cohort': None, 'rewrites': None, 'output': None, 'only': None, 'seed': 42, 'noise_max_edits': 1, 'include_rewrites': False, 'detector_scores': None, 'thresholds': None, 'lower_is_ai': []}
    values.update(overrides)
    return SimpleNamespace(**values)

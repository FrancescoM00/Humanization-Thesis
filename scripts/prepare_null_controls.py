#!/usr/bin/env python3
"""Prepare detector null controls; optionally compare already saved scores."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from humanizers.null_controls import prepare, read_jsonl, summarize_null_controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--rewrites', type=Path, action='append', required=True,
                        help='Repeat for different humanizers; identical identity duplicates are ignored')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', action='append', help='Only these reference humanizer names')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise-max-edits', type=int, default=1)
    parser.add_argument('--include-rewrites', action='store_true',
                        help='Include reference rewrites so one detector run covers both references and controls')
    parser.add_argument('--detector-scores', type=Path, action='append',
                        help='Optional existing score CSV(s); combine control and reference scores')
    parser.add_argument('--thresholds', type=Path, help='Frozen threshold JSON; required with scores')
    parser.add_argument('--lower-is-ai', action='append', default=[],
                        help='Detector name whose raw scores use strict score < threshold (e.g. a raw ratio)')
    args = parser.parse_args()
    if args.detector_scores and not args.thresholds:
        parser.error('--thresholds is required with --detector-scores')
    manifest = prepare(args.cohort, args.rewrites, args.output, seed=args.seed,
        noise_max_edits=args.noise_max_edits, include_rewrites=args.include_rewrites, only=args.only)
    if args.detector_scores:
        references = [r for path in args.rewrites for r in read_jsonl(path)
                      if not args.only or r['humanizer'] in args.only]
        scores = []
        for path in args.detector_scores:
            with path.open(newline='') as handle:
                scores.extend(csv.DictReader(handle))
        summary = summarize_null_controls(read_jsonl(args.cohort), references,
            read_jsonl(args.output/'rewrites.jsonl'), scores,
            json.loads(args.thresholds.read_text()), lower_is_ai=set(args.lower_is_ai))
        with (args.output/'null_control_comparison.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(summary)
    print(json.dumps(dict(n_rows=manifest['n_rows'], conditions=manifest['conditions'],
                          output=str(args.output)), indent=2))


if __name__ == '__main__':
    main()

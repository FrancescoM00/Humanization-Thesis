#!/usr/bin/env python3
"""Regenerate pilot/study thesis summaries from saved per-text scores, without models."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from humanizers.study_analysis import analyze_study, compare_reference, pilot_pair_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    study = commands.add_parser('study', help='Rebuild tables for the 126-source study')
    study.add_argument('--run', type=Path, default=ROOT/'inputs/study_126')
    study.add_argument('--pilot-cohort', type=Path, default=ROOT/'inputs/study_126/cohorts/pilot.jsonl')
    study.add_argument('--output', type=Path, default=ROOT/'runs/reproduced_study_126')
    study.add_argument('--reference', type=Path, help='Optional frozen published tables, read-only comparison')
    pilot = commands.add_parser('pilot', help='Paired ASR comparison of Qwen 3B and DIPPER on 42 sources')
    pilot.add_argument('--qwen-run', type=Path, default=ROOT/'inputs/pilot_42/qwen')
    pilot.add_argument('--dipper-run', type=Path, default=ROOT/'inputs/pilot_42/dipper')
    pilot.add_argument('--output', type=Path, default=ROOT/'runs/reproduced_pilot_42')
    pilot.add_argument('--reference', type=Path)
    for sub in [study, pilot]:
        sub.add_argument('--bootstrap', type=int, default=5000)
        sub.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('--bootstrap must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'study':
        result = analyze_study(args.run, args.pilot_cohort, args.output, args.bootstrap, args.seed, args.reference)
    else:
        frame = pilot_pair_comparison(args.qwen_run, args.dipper_run, args.output, args.bootstrap, args.seed)
        checks = compare_reference(args.output, args.reference, ['paired_asr_comparison.csv']) if args.reference else None
        result = {'n_detectors': len(frame), 'models_executed': False,
                  'reference_checks': len(checks) if checks is not None else 0,
                  'reference_mismatched_columns': int((~checks.matches_at_1e_12).sum()) if checks is not None and len(checks) else 0,
                  'bootstrap_draws': args.bootstrap, 'bootstrap_seed': args.seed,
                  'output': str(args.output)}
        (args.output/'reproduction_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get('reference_mismatched_columns', 0):
        raise SystemExit('Generated numerical results differ from the reference; inspect reproduction_check.csv. Frozen tables have not been overwritten.')


if __name__ == '__main__':
    main()

"""Regenerate pilot/study thesis summaries from saved per-text scores, without models."""
from pathlib import Path
import argparse
import json
import sys
from humanization.paths import project_root
ROOT = project_root()
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from humanization.evaluation.study import analyze_study, compare_reference, pilot_pair_comparison

def _argument_error(message):
    raise ValueError(message)

def run(args):
    if args.bootstrap < 1:
        _argument_error('--bootstrap must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'study':
        result = analyze_study(args.run, args.pilot_cohort, args.output, args.bootstrap, args.seed, args.reference)
    else:
        frame = pilot_pair_comparison(args.qwen_run, args.dipper_run, args.output, args.bootstrap, args.seed)
        checks = compare_reference(args.output, args.reference, ['paired_asr_comparison.csv']) if args.reference else None
        result = {'n_detectors': len(frame), 'models_executed': False, 'reference_checks': len(checks) if checks is not None else 0, 'reference_mismatched_columns': int((~checks.matches_at_1e_12).sum()) if checks is not None and len(checks) else 0, 'bootstrap_draws': args.bootstrap, 'bootstrap_seed': args.seed, 'output': str(args.output)}
        (args.output / 'reproduction_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get('reference_mismatched_columns', 0):
        raise SystemExit('Generated numerical results differ from the reference; inspect reproduction_check.csv. Frozen tables have not been overwritten.')

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'run': ROOT / 'data/processed/study_126', 'pilot_cohort': ROOT / 'data/processed/study_126/cohorts/pilot.jsonl', 'output': ROOT / 'results/new_runs/reproduced_pilot_42', 'reference': None, 'qwen_run': ROOT / 'data/processed/pilot_42/qwen', 'dipper_run': ROOT / 'data/processed/pilot_42/dipper', 'bootstrap': 5000, 'seed': 42}
    values.update(overrides)
    return SimpleNamespace(**values)

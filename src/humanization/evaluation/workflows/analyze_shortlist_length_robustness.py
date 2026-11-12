"""Analyze the candidate-pool robustness check without fitting any model."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
from humanization.evaluation.shortlist_robustness_analysis import analyze_results, write_report

def _argument_error(message):
    raise ValueError(message)

def run(args):
    if args.results.resolve() != args.output.resolve():
        shutil.copytree(args.results, args.output, dirs_exist_ok=True)
    summary = analyze_results(args.output, args.protocol, ROOT)
    report = write_report(args.output, args.protocol)
    print(json.dumps({'primary_metrics': summary['primary_metrics'], 'macro_differences': summary['macro_differences'], 'parent_C3_reproduced': summary['parent_C3_reproduced'], 'report': str(report)}, ensure_ascii=False, indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'results': ROOT / 'results/published/02_stylometry/shortlist_length_robustness', 'output': ROOT / 'results/new_runs/shortlist_100_analysis', 'protocol': None}
    values.update(overrides)
    return SimpleNamespace(**values)

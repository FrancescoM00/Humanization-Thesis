"""Analizza il controllo di robustezza delle feature candidate senza addestrare modelli."""
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
from humanization.evaluation.shortlist_robustness_analysis import analyze_results
from humanization.evaluation.shortlist_plots import plot_robustness

def _argument_error(message):
    raise ValueError(message)

def run(args):
    if args.results.resolve() != args.output.resolve():
        shutil.copytree(args.results, args.output, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('*.md', '*.MD', '*.txt', '*.pdf', '*.tex'))
    if args.protocol is not None and args.protocol.resolve() != (args.output / 'protocol.json').resolve():
        shutil.copyfile(args.protocol, args.output / 'protocol.json')
    summary = analyze_results(args.output, args.protocol, ROOT)
    figures = plot_robustness(args.output)
    print(json.dumps({'primary_metrics': summary['primary_metrics'], 'macro_differences': summary['macro_differences'], 'parent_C3_reproduced': summary['parent_C3_reproduced'], 'figures': [str(path) for path in figures]}, ensure_ascii=False, indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'results': ROOT / 'results/published/02_stylometry/shortlist_length_robustness', 'output': ROOT / 'results/new_runs/shortlist_100_analysis', 'protocol': ROOT / 'configs/protocols/shortlist_robustness.json'}
    values.update(overrides)
    return SimpleNamespace(**values)

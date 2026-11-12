"""Analizza le predizioni salvate delle feature compatte con bootstrap appaiato per gruppi."""
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
from humanization.evaluation.shortlist_statistics import analyze_results
from humanization.evaluation.shortlist_reporting import write_report

def _argument_error(message):
    raise ValueError(message)

def run(args):
    if args.results.resolve() != args.output.resolve():
        shutil.copytree(args.results, args.output, dirs_exist_ok=True)
    if args.protocol is not None and args.protocol.resolve() != (args.output / 'protocol.json').resolve():
        shutil.copyfile(args.protocol, args.output / 'protocol.json')
    summary = analyze_results(args.output, args.protocol)
    report = write_report(args.output, args.protocol)
    print(json.dumps({'primary_metrics': summary['primary_metrics'], 'report': str(report), 'elapsed_seconds': summary['elapsed_seconds']}, indent=2, ensure_ascii=False))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'results': ROOT / 'results/published/02_stylometry/shortlist', 'output': ROOT / 'results/new_runs/shortlist_analysis', 'protocol': ROOT / 'configs/protocols/shortlist.json'}
    values.update(overrides)
    return SimpleNamespace(**values)

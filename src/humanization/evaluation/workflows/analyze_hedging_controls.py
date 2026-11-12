"""Evaluate the dedicated hedging panel at its frozen historical threshold."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
from humanization.evaluation.hedging import analyze, write_report

def _argument_error(message):
    raise ValueError(message)

def run(args):
    summary = analyze(args.pairs, args.scores, args.output, args.protocol, args.review)
    report = write_report(args.output)
    print(json.dumps({'primary_results': summary['primary_results'], 'report': str(report), 'independent_human_validation_available': summary['independent_human_validation_available']}, indent=2, ensure_ascii=False))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'pairs': None, 'scores': None, 'protocol': None, 'review': None, 'output': None}
    values.update(overrides)
    return SimpleNamespace(**values)

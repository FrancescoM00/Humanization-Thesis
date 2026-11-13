"""Valuta il pannello dedicato alle cautele con la soglia precedente congelata."""
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
from humanization.evaluation.hedging import analyze, plot_support_distributions

def _argument_error(message):
    raise ValueError(message)

def run(args):
    summary = analyze(args.pairs, args.scores, args.output, args.protocol, args.review)
    figure = plot_support_distributions(args.output)
    print(json.dumps({'primary_results': summary['primary_results'], 'figure': str(figure), 'independent_human_validation_available': summary['independent_human_validation_available']}, indent=2, ensure_ascii=False))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'pairs': None, 'scores': None, 'protocol': ROOT / 'configs/protocols/hedging.json', 'review': None, 'output': None}
    values.update(overrides)
    return SimpleNamespace(**values)

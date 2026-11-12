"""Esegue l'esperimento congelato della shortlist per gruppi sulle feature numeriche in cache."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from humanization.evaluation.shortlist import run_validation

def _argument_error(message):
    raise ValueError(message)

def run(args):
    print(json.dumps(run_validation(args.protocol, args.output, ROOT), indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'protocol': ROOT / 'configs/protocols/shortlist.json', 'output': ROOT / 'results/new_runs/02_stylometry/shortlist'}
    values.update(overrides)
    return SimpleNamespace(**values)

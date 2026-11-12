"""Refit C3 on 100 candidates, keeping the exact previous train/test splits."""
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
from humanization.evaluation.shortlist_robustness import run_robustness

def _argument_error(message):
    raise ValueError(message)

def run(args):
    print(json.dumps(run_robustness(args.protocol, args.output, ROOT), indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'protocol': ROOT / 'results/published/02_stylometry/shortlist_length_robustness/protocol.json', 'output': ROOT / 'results/new_runs/02_stylometry/shortlist_length_robustness'}
    values.update(overrides)
    return SimpleNamespace(**values)

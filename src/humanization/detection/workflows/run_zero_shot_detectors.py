"""Esegue Binoculars sul campione congelata e sulle riscritture, calibrando sui soli umani."""
import argparse
import os
from pathlib import Path
import sys
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from humanization.detection.binoculars import run

def _argument_error(message):
    raise ValueError(message)

def run(args):
    run(args.cohort, args.rewrites, args.config, args.output, observer_device=args.observer_device, performer_device=args.performer_device, analysis_only=args.analysis_only, only=args.only)

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'cohort': None, 'rewrites': None, 'config': None, 'output': None, 'observer_device': 'cuda:0', 'performer_device': 'cuda:1', 'analysis_only': False, 'only': None}
    values.update(overrides)
    return SimpleNamespace(**values)

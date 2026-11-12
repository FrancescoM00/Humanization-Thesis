"""Calcola i punteggi di un insieme di detector aperti e valuta l’evasione condizionata con soglie calibrate."""
import argparse
from pathlib import Path
import sys
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from humanization.detection.pipeline import run

def _argument_error(message):
    raise ValueError(message)

def run(args):
    run(args.cohort, args.rewrites, args.config, args.output, args.device, args.analysis_only)

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'cohort': None, 'rewrites': None, 'config': None, 'output': None, 'device': 'cpu', 'analysis_only': False}
    values.update(overrides)
    return SimpleNamespace(**values)

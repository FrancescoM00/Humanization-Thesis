"""Verify the frozen panel and record source-only input handling before inference."""
import argparse
import json
from pathlib import Path
import sys
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from humanization.datasets.study_protocol import prepare_study_protocol

def _argument_error(message):
    raise ValueError(message)

def run(args):
    print(json.dumps(prepare_study_protocol(a.cohort, a.output, panel=a.panel), indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'cohort': None, 'output': None, 'panel': 'study'}
    values.update(overrides)
    return SimpleNamespace(**values)

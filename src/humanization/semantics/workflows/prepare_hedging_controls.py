"""Prepara il pannello congelato delle cautele già presenti senza modelli NLI o generativi."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from humanization.semantics.hedging_controls import prepare_panel

def _argument_error(message):
    raise ValueError(message)

def run(args):
    print(json.dumps(prepare_panel(args.protocol, args.output, ROOT), ensure_ascii=False, indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'protocol': ROOT / 'configs/protocols/hedging.json', 'output': ROOT / 'results/new_runs/hedging'}
    values.update(overrides)
    return SimpleNamespace(**values)

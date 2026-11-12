"""Run fixed rewriting systems on a frozen desktop cohort, with safe resume."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from humanization.rewriting.pipeline import run_humanizers

def _argument_error(message):
    raise ValueError(message)

def run(args):
    manifest = run_humanizers(args.cohort, args.config, args.output, device=args.device, only=args.only)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'cohort': None, 'config': None, 'output': None, 'device': 'cuda', 'only': None}
    values.update(overrides)
    return SimpleNamespace(**values)

#!/usr/bin/env python3
"""Run fixed rewriting systems on a frozen Kaggle cohort, with safe resume."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from humanizers.rewriting import run_humanizers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda",
                        help="cpu, cuda or cuda:N; no implicit CPU fallback")
    parser.add_argument("--only", action="append",
                        help="Select a named preset; explicitly opts in even when enabled=false")
    args = parser.parse_args()
    manifest = run_humanizers(args.cohort, args.config, args.output,
                              device=args.device, only=args.only)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

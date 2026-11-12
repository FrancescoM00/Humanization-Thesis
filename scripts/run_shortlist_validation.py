#!/usr/bin/env python3
"""Run the frozen grouped shortlist experiment on cached numeric features."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from stylometry.shortlist_validation import run_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "results/02_stylometry/shortlist/protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/02_stylometry/shortlist")
    args = parser.parse_args()
    print(json.dumps(run_validation(args.protocol, args.output, ROOT), indent=2))


if __name__ == "__main__":
    main()

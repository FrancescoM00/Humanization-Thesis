#!/usr/bin/env python3
"""Score an open detector pool and evaluate calibrated conditional evasion."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from humanizers.detection import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--rewrites', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--analysis-only', action='store_true', help='Use existing detector_scores.csv; do not load or download any checkpoint')
    args = parser.parse_args()
    run(args.cohort, args.rewrites, args.config, args.output, args.device, args.analysis_only)


if __name__ == '__main__':
    main()

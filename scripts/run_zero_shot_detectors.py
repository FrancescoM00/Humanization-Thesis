#!/usr/bin/env python3
"""Run Binoculars on the frozen cohort and rewrites, calibrating on humans only."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

from humanizers.zero_shot import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--rewrites', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observer-device', default='cuda:0')
    parser.add_argument('--performer-device', default='cuda:1')
    parser.add_argument('--analysis-only', action='store_true')
    parser.add_argument('--only', action='append', help='Explicit zero-shot detector name; repeat to select several')
    args = parser.parse_args()
    run(args.cohort, args.rewrites, args.config, args.output,
        observer_device=args.observer_device, performer_device=args.performer_device,
        analysis_only=args.analysis_only, only=args.only)


if __name__ == '__main__':
    main()

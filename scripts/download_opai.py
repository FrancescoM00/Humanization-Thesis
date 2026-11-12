#!/usr/bin/env python3
"""Resolve and save the exact public OpAI-Bench release revision before download."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')


def main():
    import requests
    from semantics.opai import OFFICIAL_DATASET, download_official_records
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/opai/records.jsonl')
    parser.add_argument('--revision', help='Exact 40-character dataset SHA; defaults to resolving main once')
    args = parser.parse_args()
    try:
        if args.revision:
            revision = args.revision
        else:
            response = requests.get(f'https://huggingface.co/api/datasets/{OFFICIAL_DATASET}/revision/main', timeout=30)
            if not response.ok:
                raise RuntimeError(f'Official dataset metadata returned HTTP {response.status_code}')
            revision = response.json()['sha']
        manifest = download_official_records(args.output, revision=revision)
    except (requests.RequestException, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'Official OpAI download unavailable ({type(exc).__name__}); check authorized huggingface.co access or provide local official files.', file=sys.stderr)
        return 1
    args.output.with_suffix('.provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

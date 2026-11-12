"""Salva la revisione pubblica esatta di OpAI-Bench prima del download."""
import argparse
import json
import os
from pathlib import Path
import sys
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

def _argument_error(message):
    raise ValueError(message)

def run(args):
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

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'output': ROOT / 'data/raw/opai/records.jsonl', 'revision': None}
    values.update(overrides)
    return SimpleNamespace(**values)

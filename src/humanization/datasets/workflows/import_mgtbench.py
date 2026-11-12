
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from humanization.paths import project_root
ROOT = project_root()
EXPECTED = {'Essay_LLMs.csv': 'f7ae324296b3d81b2134e873406f70a457c0383168aa7a9de83b70051ca41707', 'Reuters_LLMs.csv': 'd3a31dd8fd3b04ec333ec29ad3a929618b5a88beaf2e8338b50594957a6fa32b', 'WP_LLMs.csv': 'ee787f528f2b5d4440cf9cd714ea4f81f9cf7b59086c37d90a376ba3db3a7278'}
REQUIRED = {'id', 'prompt', 'human', 'ChatGPT-turbo', 'Claude', 'ChatGLM', 'Dolly', 'ChatGPT', 'GPT4All', 'StableLM'}

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def inspect(directory):
    result = {}
    for name, expected in EXPECTED.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f'Manca il CSV originale: {path}')
        actual = sha(path)
        if actual != expected:
            raise ValueError(f'{name}: SHA-256 diverso dai dati degli esperimenti. Non cambiare ordine delle righe, celle o codifica dei CSV originali.')
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            if REQUIRED - set(columns):
                raise ValueError(f'{name}: colonne mancanti {sorted(REQUIRED - set(columns))}')
            n_rows = sum((1 for _ in reader))
        if n_rows != 1000:
            raise ValueError(f'{name}: attese 1.000 righe, trovate {n_rows}')
        result[name] = {'sha256': actual, 'bytes': path.stat().st_size, 'rows': n_rows, 'columns': columns, 'matches_published_experiments': True}
    return result

def _argument_error(message):
    raise ValueError(message)

def run(args):
    source, output = (args.input_dir.resolve(), args.output.resolve())
    files = inspect(source)
    if not args.verify_only:
        for name, info in files.items():
            target = output / name
            if target.exists() and sha(target) != info['sha256']:
                raise FileExistsError(f'{target} contiene dati diversi: usare una cartella nuova')
        output.mkdir(parents=True, exist_ok=True)
        for name in files:
            target = output / name
            if not target.exists():
                shutil.copy2(source / name, target)
        if not (output / 'SOURCE.json').exists():
            manifest = {'source': 'User-supplied official MGTBench CSV copies', 'imported_at_utc': datetime.now(timezone.utc).isoformat(), 'official_source': 'https://drive.google.com/drive/folders/1p4iBeM4r-sUKe8TnS4DcYlxvQagcmola', 'files': files, 'transformations_on_import': []}
            (output / 'SOURCE.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    for name, info in files.items():
        print(f"{name}: {info['rows']} righe, SHA-256 {info['sha256']} — OK")
    print('Le copie coincidono con i dati degli esperimenti. Nessuna pulizia applicata ai CSV.')

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'input_dir': ROOT / 'data/raw/mgtbench', 'output': ROOT / 'data/raw/mgtbench', 'verify_only': False}
    values.update(overrides)
    return SimpleNamespace(**values)

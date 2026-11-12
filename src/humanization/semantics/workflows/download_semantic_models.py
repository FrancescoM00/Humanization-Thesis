"""Scarica file pubblici di modelli con revisione fissata tramite HTTPS verificato.

Non modifica mai le regole di rete, non registra URL di redirect firmati e
non abilita codice remoto.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit
MODELS = {'embedding': 'sentence-transformers/all-MiniLM-L6-v2', 'bertscore': 'FacebookAI/roberta-base', 'nli': 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'}
CONFIG_FILES = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'vocab.json', 'merges.txt', 'spm.model', 'sentencepiece.bpe.model', 'modules.json', 'sentence_bert_config.json', '1_Pooling/config.json', 'README.md'}

def download_one(kind, destination, revision='main'):
    import requests
    repo = MODELS[kind]
    destination = Path(destination) / kind
    destination.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    def get(url, **kwargs):
        try:
            response = session.get(url, timeout=45, **kwargs)
        except requests.RequestException as exc:
            raise RuntimeError(f'HTTPS request to {urlsplit(url).hostname} failed ({type(exc).__name__}); check network policy') from None
        if not response.ok:
            raise RuntimeError(f'HTTPS {response.status_code} from {urlsplit(response.url).hostname}; no model downloaded')
        return response
    spec = get(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true').json()
    revision = spec['sha']
    files = {s['rfilename']: s for s in spec['siblings']}
    weights = next((name for name in ['model.safetensors', 'pytorch_model.bin'] if name in files), None)
    if weights is None:
        raise RuntimeError(f'{repo}: no supported unsharded model weights; inspect upstream layout')
    selected = sorted(set(files) & CONFIG_FILES | {weights})
    manifest = {'repository': repo, 'revision': revision, 'kind': kind, 'files': [], 'tls_verification': True, 'trust_remote_code': False}
    for name in selected:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = files[name].get('lfs', {}).get('sha256')
        if path.is_file() and expected and (hashlib.sha256(path.read_bytes()).hexdigest() == expected):
            digest = expected
        else:
            response = get(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', stream=True)
            tmp = path.with_suffix(path.suffix + '.download')
            digest_object = hashlib.sha256()
            try:
                with tmp.open('wb') as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            digest_object.update(block)
                            handle.write(block)
                digest = digest_object.hexdigest()
                if expected and digest != expected:
                    raise RuntimeError(f'{kind}/{name}: upstream LFS SHA-256 verification failed')
                tmp.replace(path)
            finally:
                if tmp.exists():
                    tmp.unlink()
        manifest['files'].append({'name': name, 'sha256': digest, 'upstream_lfs_sha256': expected, 'bytes': path.stat().st_size})
        print(f'{kind}: {name} ({path.stat().st_size} bytes)', flush=True)
    (destination / 'DOWNLOAD_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest

def _argument_error(message):
    raise ValueError(message)

def run(args):
    if args.revision and args.model == 'all':
        _argument_error('--revision requires selecting one --model')
    try:
        for kind in MODELS if args.model == 'all' else [args.model]:
            download_one(kind, args.output, args.revision or 'main')
    except (RuntimeError, OSError, KeyError, ValueError) as exc:
        print(f'Model download blocked: {exc}', file=sys.stderr)
        return 1
    return 0

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'output': Path('data/models'), 'model': 'all', 'revision': None}
    values.update(overrides)
    return SimpleNamespace(**values)

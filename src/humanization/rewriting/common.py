from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import time
PROTOCOL = 'humanizer_fixed_single_attempt_v1'
DIPPER_TOKENIZER = 'google/t5-v1_1-xxl'
DIPPER_TOKENIZER_REVISION = '3db67ab1af984cf10548a73467f0e5bca2aaaeb2'
DIPPER_WINDOW_PROTOCOLS = {'entire_passage_no_prefix', 'three_sentence_context_v2'}

def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def _atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)

def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'

def _jsonl(rows) -> str:
    return ''.join((json.dumps(row, ensure_ascii=False) + '\n' for row in rows))

def pair_id(system: str, source_id: str) -> str:
    payload = json.dumps([system, source_id], ensure_ascii=False).encode()
    return 'humanizer_' + _digest(payload)[:24]

def generation_status(text: str, generated_tokens: int, max_new_tokens: int, finished_eos: bool) -> str:
    """Una generazione che raggiunge il limite di token senza EOS non è mai una riscrittura valida."""
    if not text.strip():
        return 'generation_error'
    if generated_tokens >= max_new_tokens and (not finished_eos):
        return 'truncated'
    return 'ok'

def _base_row(source: dict, system: dict, seed: int) -> dict:
    original = source['machine']
    return {'id': pair_id(system['name'], source['id']), 'source_id': source['id'], 'source_group_id': source['source_group_id'], 'domain': source['domain'], 'generator': source['generator'], 'humanizer': system['name'], 'dataset': 'MGTBench_humanizer_pilot', 'perturbation': 'identity' if system['kind'] == 'identity' else 'rewrite', 'original': original, 'modified': '', 'status': 'pending', 'metadata': {'protocol': PROTOCOL, 'model_id': system.get('model_id'), 'revision': system.get('revision'), 'seed': seed, 'original_sha256': _digest(original.encode()), 'input_tokens': None, 'output_tokens': None, 'finished_eos': None, 'unchanged': False, 'do_sample': False, 'num_return_sequences': 1, 'detector_feedback_used': False}}

def _validate_system(system: dict) -> None:
    if system.get('kind') not in {'identity', 'causal', 'dipper'}:
        raise ValueError(f"Unsupported humanizer kind: {system.get('kind')}")
    if not isinstance(system.get('name'), str) or not system['name']:
        raise ValueError('Every humanizer needs a nonempty name')
    if system['kind'] == 'identity':
        return
    if not system.get('model_id') or not re.fullmatch('[0-9a-f]{40}', system.get('revision', '')):
        raise ValueError(f"{system['name']}: specify model_id and an immutable 40-character revision")
    if int(system.get('max_input_tokens', 0)) < 1 or int(system.get('max_new_tokens', 0)) < 1:
        raise ValueError(f"{system['name']}: positive input and generation budgets are required")
    if system['kind'] == 'causal' and (not str(system.get('prompt', '')).strip()):
        raise ValueError('The causal humanizer requires a fixed instruction prompt')
    if system.get('tokenizer_id') and (not re.fullmatch('[0-9a-f]{40}', system.get('tokenizer_revision', ''))):
        raise ValueError('An external tokenizer must have an immutable tokenizer_revision')
    if system['kind'] == 'dipper':
        if system.get('window_protocol', 'entire_passage_no_prefix') not in DIPPER_WINDOW_PROTOCOLS:
            raise ValueError('Unsupported DIPPER window_protocol')
        dipper_prompt('Budget check.', system.get('lexical_diversity', 60), system.get('order_diversity', 0))

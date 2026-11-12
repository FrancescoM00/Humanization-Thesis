"""Esegue sistemi di riscrittura fissi e non adattivi, conservando i tentativi falliti.

Il sistema identità richiede soltanto la libreria standard di Python. I modelli
neurali sono importati quando servono, caricati uno alla volta e mai interrogati
attraverso un detector.
"""
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

def _rewrite_one(source: dict, system: dict, seed: int, tokenizer, model, device) -> dict:
    import torch
    if system['kind'] == 'dipper' and system.get('window_protocol', 'entire_passage_no_prefix') == 'three_sentence_context_v2':
        return _rewrite_dipper_context(source, system, seed, tokenizer, model, device)
    row = _base_row(source, system, seed)
    text = source['machine']
    if system['kind'] == 'dipper':
        model_input = dipper_prompt(text, system.get('lexical_diversity', 60), system.get('order_diversity', 0))
        row['metadata']['dipper_window_protocol'] = 'entire_passage_no_prefix'
        row['metadata']['lexical_diversity'] = system.get('lexical_diversity', 60)
        row['metadata']['order_diversity'] = system.get('order_diversity', 0)
        add_special_tokens = True
    else:
        model_input = qwen.format_input(tokenizer, system, text)
        row['metadata']['prompt_sha256'] = _digest(system['prompt'].encode())
        add_special_tokens = False
    encoded = tokenizer(model_input, return_tensors='pt', truncation=False, add_special_tokens=add_special_tokens)
    input_tokens = int(encoded['input_ids'].shape[-1])
    row['metadata']['input_tokens'] = input_tokens
    budget = int(system['max_input_tokens'])
    maximum_output = int(system['max_new_tokens'])
    model_limit = getattr(model.config, 'max_position_embeddings', None)
    total_exceeds = system['kind'] == 'causal' and isinstance(model_limit, int) and (input_tokens + maximum_output > model_limit)
    if input_tokens > budget or total_exceeds:
        row['status'] = 'input_too_long'
        row['metadata']['error'] = 'Full input exceeds the declared/model token budget; nothing was truncated.'
        return row
    encoded = {key: value.to(device) for key, value in encoded.items()}
    started = time.monotonic()
    torch.manual_seed(seed)
    with torch.inference_mode():
        result = model.generate(**encoded, do_sample=False, num_beams=1, num_return_sequences=1, max_new_tokens=maximum_output, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    sequence = result[0].tolist()
    if system['kind'] == 'causal':
        generated = sequence[input_tokens:]
    else:
        start_id = getattr(model.config, 'decoder_start_token_id', None)
        generated = sequence[1:] if sequence and sequence[0] == start_id else sequence
    eos = getattr(model.generation_config, 'eos_token_id', None)
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    eos_ids.add(tokenizer.eos_token_id)
    finished = bool(generated and generated[-1] in eos_ids)
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    row['modified'] = decoded
    row['status'] = generation_status(decoded, len(generated), maximum_output, finished)
    row['metadata'].update(output_tokens=len(generated), finished_eos=finished, elapsed_seconds=time.monotonic() - started, unchanged=' '.join(decoded.split()) == ' '.join(text.split()))
    return row

def run_humanizers(cohort_path: Path, config_path: Path, output: Path, *, device: str='cuda', only: list[str] | None=None) -> dict:
    cohort_path, config_path, output = (Path(cohort_path), Path(config_path), Path(output))
    cohort = read_jsonl(cohort_path)
    sources = [row for row in cohort if row.get('split') == 'test']
    if not sources or len({row['id'] for row in sources}) != len(sources):
        raise ValueError('Need nonempty test rows with unique ids')
    for row in sources:
        if not isinstance(row.get('machine'), str) or not row['machine'].strip():
            raise ValueError('Every test source needs a nonempty machine text')
        for field in ('source_group_id', 'domain', 'generator'):
            if not row.get(field):
                raise ValueError(f'Missing cohort field: {field}')
    config = json.loads(config_path.read_text())
    configured = config['humanizers']
    names = [system.get('name') for system in configured]
    if len(set(names)) != len(names):
        raise ValueError('Humanizer names must be unique')
    if only and set(only) - set(names):
        raise ValueError(f'Unknown --only humanizer: {set(only) - set(names)}')
    systems = [s for s in configured if (s['name'] in only if only else s.get('enabled', True))]
    if not systems:
        raise ValueError('No humanizers selected')
    for system in systems:
        _validate_system(system)
    seed = int(config.get('seed', 42))
    fingerprint = {'protocol': PROTOCOL, 'cohort_sha256': _digest(cohort_path.read_bytes()), 'config_sha256': _digest(config_path.read_bytes()), 'code_sha256': _digest(Path(__file__).read_bytes()), 'selected_humanizers': [s['name'] for s in systems], 'device': device}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, rows_path = (output / 'humanizer_run.json', output / 'rewrites.jsonl')
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any((previous.get(key) != value for key, value in fingerprint.items())):
            raise ValueError('Resume configuration/cohort/device differs; choose a new output directory')
    elif rows_path.exists():
        raise ValueError('Rewrites exist without a matching run manifest; choose a new output directory')
    rows = read_jsonl(rows_path) if rows_path.exists() else []
    ids = [row['id'] for row in rows]
    expected = {pair_id(s['name'], source['id']) for s in systems for source in sources}
    if len(set(ids)) != len(ids) or not set(ids).issubset(expected):
        raise ValueError('Existing rewrite rows contain duplicate or foreign ids')
    known = set(ids)
    created = previous['created_at'] if manifest_path.exists() else datetime.now(timezone.utc).isoformat()
    versions = {}
    for package in ('torch', 'transformers', 'accelerate', 'bitsandbytes', 'sentencepiece'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = dict(fingerprint, created_at=created, expected_rewrites=len(expected), n_sources=len(sources), seed=seed, systems=systems, versions=versions, generation_attempts_per_system_source=1, resumed_completed_rewrites=len(rows))

    def save(status: str) -> None:
        manifest.update(status=status, completed_rewrites=len(rows), status_counts=dict(Counter((row['status'] for row in rows))))
        _atomic(rows_path, _jsonl(rows))
        _atomic(output / 'pairs.jsonl', _jsonl((row for row in rows if row['status'] == 'ok')))
        _atomic(manifest_path, _json(manifest))
    save('running')
    for system in systems:
        pending = [source for source in sources if pair_id(system['name'], source['id']) not in known]
        if not pending:
            continue
        tokenizer = model = selected_device = None
        try:
            if system['kind'] != 'identity':
                tokenizer, model, selected_device = _load_model(system, device)
            for source in pending:
                if system['kind'] == 'identity':
                    row = _base_row(source, system, seed)
                    row.update(modified=source['machine'], status='ok')
                    row['metadata']['unchanged'] = True
                else:
                    try:
                        row = _rewrite_one(source, system, seed, tokenizer, model, selected_device)
                    except Exception as error:
                        row = error.rewrite_row if isinstance(error, _DipperWindowGenerationError) else _base_row(source, system, seed)
                        row['status'] = 'generation_error'
                        row['metadata'].setdefault('error_type', type(error).__name__)
                        row['metadata'].setdefault('error', 'Generation failed; inspect the notebook error output.')
                        rows.append(row)
                        known.add(row['id'])
                        save('interrupted')
                        raise
                rows.append(row)
                known.add(row['id'])
                save('running')
                print(f"{system['name']}: {len(rows)}/{len(expected)} total; {row['status']}", flush=True)
        except BaseException:
            save('interrupted')
            raise
        finally:
            del model, tokenizer
            gc.collect()
            if system['kind'] != 'identity':
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    save('completed')
    return manifest
from .common import _digest, read_jsonl, _atomic, _json, _jsonl, pair_id, generation_status, _base_row, _validate_system
from .loading import _cached_hf_roots, _load_cached_tokenizer, _load_model
from .dipper import dipper_prompt, dipper_context_prompt, _dipper_sentence_nlp, _dipper_sentences, _DipperWindowGenerationError, _rewrite_dipper_context
from . import qwen

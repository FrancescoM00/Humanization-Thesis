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
import tempfile
import time
from humanization.paths import project_root
PROTOCOL = 'humanizer_fixed_single_attempt_v1'
DIPPER_TOKENIZER = 'google/t5-v1_1-xxl'
DIPPER_TOKENIZER_REVISION = '3db67ab1af984cf10548a73467f0e5bca2aaaeb2'
DIPPER_WINDOW_PROTOCOLS = {'entire_passage_no_prefix', 'three_sentence_context_v2'}

def _cached_hf_roots() -> list[Path]:
    from huggingface_hub import constants
    candidates = [os.environ.get('HF_HUB_CACHE'), os.environ.get('HUMANIZATION_HF_READ_CACHE'), constants.HF_HUB_CACHE, Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))) / 'hub', project_root() / '.cache/huggingface/hub', Path(tempfile.gettempdir()) / 'humanization/huggingface/hub', Path.home() / '.cache/huggingface/hub']
    return list(dict.fromkeys((Path(root) for root in candidates if root)))

def _load_cached_tokenizer(system: dict, roots: list[Path], tokenizer_cls):
    dipper = system['kind'] == 'dipper'
    repo_id = system.get('tokenizer_id', DIPPER_TOKENIZER if dipper else system['model_id'])
    revision = system.get('tokenizer_revision', DIPPER_TOKENIZER_REVISION if dipper else system['revision'])
    kwargs = {} if dipper else {'trust_remote_code': False}
    for root in roots:
        try:
            return tokenizer_cls.from_pretrained(repo_id, revision=revision, cache_dir=str(root), local_files_only=True, **kwargs)
        except OSError:
            continue
    raise RuntimeError(f'HUMANIZATION_REUSE_CACHED_MODELS=1: tokenizer {repo_id}@{revision} is not available in the existing caches. Populate its pinned cache in a writable session before enabling strictly local model reuse.')

def _load_model(system: dict, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, T5ForConditionalGeneration, T5Tokenizer
    selected_device = torch.device(device)
    if selected_device.type not in {'cpu', 'cuda'}:
        raise ValueError('Use cpu, cuda or cuda:N')
    reuse_cached = os.environ.get('HUMANIZATION_REUSE_CACHED_MODELS') == '1'
    snapshot = None
    if reuse_cached:
        from humanization.detection.binoculars import cached_weight_snapshot
        roots = _cached_hf_roots()
        snapshot = cached_weight_snapshot(system['model_id'], system['revision'], roots)
        if snapshot is None:
            raise RuntimeError(f"HUMANIZATION_REUSE_CACHED_MODELS=1: no complete pinned checkpoint for {system['model_id']}@{system['revision']} in existing caches. All safetensors shards and config.json are required; no download was attempted.")
        tokenizer = _load_cached_tokenizer(system, roots, T5Tokenizer if system['kind'] == 'dipper' else AutoTokenizer)
    if selected_device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Select a CUDA-capable GPU before this stage.')
        if selected_device.index is None:
            selected_device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(selected_device.index)
    quantized = bool(system.get('load_in_4bit', False))
    if quantized and selected_device.type != 'cuda':
        raise ValueError('The 4-bit rewriting configuration requires CUDA')
    if system['kind'] == 'dipper':
        if not reuse_cached:
            from huggingface_hub import constants
            cache = Path(constants.HF_HUB_CACHE)
            cache.mkdir(parents=True, exist_ok=True)
            model_cache = cache / ('models--' + system['model_id'].replace('/', '--'))
            cached = sum((p.stat().st_size for p in model_cache.rglob('*') if p.is_file() and (not p.is_symlink()))) if model_cache.exists() else 0
            minimum = max(0, 55 * 1024 ** 3 - cached)
            if shutil.disk_usage(cache).free < minimum:
                raise RuntimeError('DIPPER needs approximately 45 GB of checkpoint files. Provide at least 55 GiB of free model-cache space before its first download.')
        if selected_device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(selected_device)
            if free < 8 * 1024 ** 3:
                raise RuntimeError('DIPPER preflight requires at least 8 GiB of free GPU memory; this is a preliminary check, not a guarantee against OOM.')
        if not quantized:
            raise ValueError('This desktop DIPPER preset requires load_in_4bit=true')
        if not reuse_cached:
            tokenizer = T5Tokenizer.from_pretrained(system.get('tokenizer_id', DIPPER_TOKENIZER), revision=system.get('tokenizer_revision', DIPPER_TOKENIZER_REVISION))
        model_cls = T5ForConditionalGeneration
    else:
        if not reuse_cached:
            tokenizer = AutoTokenizer.from_pretrained(system.get('tokenizer_id', system['model_id']), revision=system.get('tokenizer_revision', system['revision']), trust_remote_code=False)
        model_cls = AutoModelForCausalLM
    kwargs = {'revision': system['revision'], 'trust_remote_code': False, 'torch_dtype': torch.float16 if selected_device.type == 'cuda' else torch.float32, 'low_cpu_mem_usage': True}
    if quantized:
        kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16), device_map={'': selected_device.index if selected_device.index is not None else 0})
    if reuse_cached:
        kwargs['local_files_only'] = True
    model = model_cls.from_pretrained(str(snapshot) if reuse_cached else system['model_id'], **kwargs)
    if not quantized:
        model.to(selected_device)
    model.eval()
    return (tokenizer, model, selected_device)

"""Run fixed, non-adaptive rewriting systems and preserve unsuccessful attempts.

The identity system requires only the Python standard library. Neural models
are imported lazily, loaded one at a time, and never queried through a detector.
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
from .common import _base_row, generation_status, _digest


def dipper_prompt(text: str, lexical_diversity: int, order_diversity: int) -> str:
    allowed = {0, 20, 40, 60, 80, 100}
    if lexical_diversity not in allowed or order_diversity not in allowed:
        raise ValueError('DIPPER diversity must be one of 0, 20, 40, 60, 80, 100')
    normalized = ' '.join(text.split())
    return f'lexical = {100 - lexical_diversity}, order = {100 - order_diversity} <sent> {normalized} </sent>'

def dipper_context_prompt(text: str, prefix: str, lexical_diversity: int, order_diversity: int) -> str:
    """Model-card format: all previous rewritten context precedes <sent>."""
    prompt = dipper_prompt(text, lexical_diversity, order_diversity)
    prefix = ' '.join(prefix.split())
    if not prefix:
        return prompt
    head, marker, tail = prompt.partition(' <sent> ')
    return f'{head} {prefix}{marker}{tail}'

@lru_cache(maxsize=1)
def _dipper_sentence_nlp():
    from humanization.stylometry.features import load_nlp
    return load_nlp()

def _dipper_sentences(text: str) -> list[str]:
    normalized = ' '.join(text.split())
    return [sentence.text.strip() for sentence in _dipper_sentence_nlp()(normalized).sents if sentence.text.strip()]

class _DipperWindowGenerationError(RuntimeError):
    """Retain completed windows if a later model call raises an exception."""

    def __init__(self, rewrite_row: dict):
        super().__init__('DIPPER window generation failed; inspect the notebook error output.')
        self.rewrite_row = rewrite_row

def _rewrite_dipper_context(source: dict, system: dict, seed: int, tokenizer, model, device) -> dict:
    """Three-sentence windows with cumulative rewritten prefix, no truncation.

    The public recipe's context placement/inverse codes are retained. Sentence
    splitting (spaCy), NF4 loading, greedy decoding and max_new_tokens per window
    are explicit experimental choices, not a claim of full paper replication.
    A partial output remains auditable but never enters successful pairs.
    """
    import torch
    row = _base_row(source, system, seed)
    metadata = row['metadata']
    started = time.monotonic()
    sentences = _dipper_sentences(source['machine'])
    windows = [' '.join(sentences[i:i + 3]) for i in range(0, len(sentences), 3)]
    details = []
    outputs = []
    metadata.update(dipper_window_protocol='three_sentence_context_v2', dipper_sentence_segmentation='spaCy en_core_web_sm parser after whitespace normalization', dipper_context_policy='all prior rewritten windows; no rolling or dropped context', dipper_recipe_source='https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl/blob/main/README.md', lexical_diversity=system.get('lexical_diversity', 60), order_diversity=system.get('order_diversity', 0), window_size_sentences=3, n_input_sentences=len(sentences), n_windows_expected=len(windows), n_windows_generated=0, n_windows_completed=0, input_tokens=0, input_token_aggregation='sum of full window prompts including cumulative prefix', output_tokens=0, finished_eos=False, max_new_tokens_per_window=int(system['max_new_tokens']), generation_budget_scope='per window; total generation budget grows with window count', window_generation=details, elapsed_seconds=0.0)
    maximum_output = int(system['max_new_tokens'])
    budget = int(system['max_input_tokens'])
    torch.manual_seed(seed)

    def finish(status):
        row['status'] = status
        row['modified'] = ' '.join(outputs).strip()
        metadata['unchanged'] = ' '.join(row['modified'].split()) == ' '.join(source['machine'].split())
        metadata['finished_eos'] = bool(windows and metadata['n_windows_completed'] == len(windows))
        metadata['window_processing_elapsed_seconds'] = time.monotonic() - started
        return row
    if not windows:
        metadata['error'] = 'Sentence segmentation produced no nonempty window.'
        return finish('generation_error')
    for index, window in enumerate(windows):
        prefix = ' '.join(outputs)
        prompt = dipper_context_prompt(window, prefix, metadata['lexical_diversity'], metadata['order_diversity'])
        detail = dict(window_index=index, sentence_start=index * 3, sentence_end_exclusive=min(index * 3 + 3, len(sentences)), original_window_sha256=_digest(window.encode()), full_prompt_sha256=_digest(prompt.encode()), prefix_characters=len(prefix), input_tokens=None, output_tokens=0, finished_eos=None, status='pending')
        details.append(detail)
        try:
            encoded = tokenizer(prompt, return_tensors='pt', truncation=False, add_special_tokens=True)
            input_tokens = int(encoded['input_ids'].shape[-1])
            detail['input_tokens'] = input_tokens
            metadata['input_tokens'] += input_tokens
            if input_tokens > budget:
                detail['status'] = 'input_too_long'
                metadata['error'] = 'Full window with cumulative rewritten prefix exceeds the declared input budget; no text or context was truncated.'
                metadata['failed_window_index'] = index
                return finish('input_too_long')
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated_started = time.monotonic()
            with torch.inference_mode():
                result = model.generate(**encoded, do_sample=False, num_beams=1, num_return_sequences=1, max_new_tokens=maximum_output, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
            sequence = result[0].tolist()
            start_id = getattr(model.config, 'decoder_start_token_id', None)
            generated = sequence[1:] if sequence and sequence[0] == start_id else sequence
            eos = getattr(model.generation_config, 'eos_token_id', None)
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
            eos_ids.add(tokenizer.eos_token_id)
            finished = bool(generated and generated[-1] in eos_ids)
            decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
            elapsed = time.monotonic() - generated_started
            status = generation_status(decoded, len(generated), maximum_output, finished)
            detail.update(output_tokens=len(generated), finished_eos=finished, status=status, elapsed_seconds=elapsed)
            metadata['elapsed_seconds'] += elapsed
            metadata['output_tokens'] += len(generated)
            metadata['n_windows_generated'] += 1
            if decoded:
                outputs.append(decoded)
            if status != 'ok':
                metadata['failed_window_index'] = index
                return finish(status)
            metadata['n_windows_completed'] += 1
        except Exception as error:
            detail['status'] = 'generation_error'
            metadata['error_type'] = type(error).__name__
            metadata['error'] = 'Window generation failed; partial outputs are retained for audit.'
            metadata['failed_window_index'] = index
            finish('generation_error')
            raise _DipperWindowGenerationError(row) from error
    return finish('ok')

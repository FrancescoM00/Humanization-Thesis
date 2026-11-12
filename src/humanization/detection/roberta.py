"""Open detector evaluation with human-only calibration and conditional ASR.

Detector scores are token-weighted means over non-overlapping, complete chunks.
They are model scores, not calibrated probabilities of human authorship. Each
detector receives its own frozen threshold; a three-model majority vote combines
binary decisions, never raw scores. Failed generations remain in ASR denominators.
"""
from __future__ import annotations
from collections import defaultdict
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import time
import numpy as np
import pandas as pd
from humanization.datasets.records import digest

def validate_ai_label(model_config, detector):
    """Require the explicit model-card class direction, checking actual config."""
    index = detector.get('ai_label_index')
    if isinstance(index, bool) or not isinstance(index, int) or index not in (0, 1):
        raise ValueError('Every binary detector requires explicit ai_label_index 0 or 1')
    if int(model_config.num_labels) != 2:
        raise ValueError('Only binary sequence-classification detectors are supported')
    labels = {int(key): str(value) for key, value in model_config.id2label.items()}
    if set(labels) != {0, 1}:
        raise ValueError('Detector config must expose exactly two labels, 0 and 1')
    expected = detector.get('ai_label')
    if expected is not None and labels[index].casefold() != str(expected).casefold():
        raise ValueError(f'AI label mismatch: config index {index} is {labels[index]!r}, expected {expected!r}')
    return (index, labels)

def _require_device(device):
    import torch
    selected = torch.device(device)
    if selected.type not in {'cpu', 'cuda'}:
        raise ValueError('Choose cpu or cuda')
    if selected.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA unavailable: enable a desktop GPU accelerator or select --device cpu')
    return selected

def score_detector(records, detector, output, device='cpu'):
    """Run one checkpoint, persisting text-keyed scores after every document."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from humanization.semantics.metrics import token_chunks
    selected = _require_device(device)
    revision = detector.get('revision')
    if not revision:
        raise ValueError('Pin a detector checkpoint revision before inference')
    if detector.get('preprocessing', 'none') != 'none':
        raise ValueError('This experiment implements the documented raw-input variant only (preprocessing=none)')
    if detector['model_id'] == 'TrustSafeAI/RADAR-Vicuna-7B':
        version = tuple((int(x) for x in re.match('(\\d+)\\.(\\d+)', torch.__version__).groups()))
        if version < (2, 6):
            raise RuntimeError('RADAR uses PyTorch .bin weights: use torch >= 2.6 with the desktop CUDA build')
    fingerprint = digest({'detector': detector, 'device': str(selected), 'aggregation': 'full-text-nonoverlap-token-weighted-probability-v1', 'implementation': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers')})
    cache_path = Path(output) / 'detector_cache' / (fingerprint + '.jsonl')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if cache_path.exists():
        for item in read_jsonl(cache_path):
            cached[item['text_sha256']] = item
    tokenizer = AutoTokenizer.from_pretrained(detector['model_id'], revision=revision, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(detector['model_id'], revision=revision, trust_remote_code=False, torch_dtype=torch.float32)
    index, labels = validate_ai_label(model.config, detector)
    model.to(selected).eval()
    context_limit = getattr(model.config, 'max_position_embeddings', 512)
    if getattr(model.config, 'model_type', '') == 'roberta':
        context_limit -= int(getattr(model.config, 'pad_token_id', 1)) + 1
    total_budget = min(int(detector.get('max_tokens', 512)), int(tokenizer.model_max_length), context_limit)
    budget = total_budget - tokenizer.num_special_tokens_to_add(pair=False)
    if budget < 1:
        raise ValueError('Detector context budget is too small')
    batch_size = int(detector.get('batch_size', 8))
    results = []
    started = time.monotonic()
    try:
        for position, row in enumerate(records):
            text = row['text']
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            if text_hash not in cached:
                chunks = token_chunks(text, tokenizer, max_tokens=budget)
                weights = [len(tokenizer.encode(chunk, add_special_tokens=False)) for chunk in chunks]
                predictions = []
                for start in range(0, len(chunks), batch_size):
                    batch = tokenizer(chunks[start:start + batch_size], padding=True, truncation=False, return_tensors='pt')
                    if batch['input_ids'].shape[1] > total_budget:
                        raise RuntimeError('Detector tokenizer exceeded the verified context; input was not truncated')
                    batch = {key: value.to(selected) for key, value in batch.items()}
                    with torch.inference_mode():
                        predictions.extend(model(**batch).logits.softmax(-1)[:, index].cpu().tolist())
                if not weights or not all(weights):
                    raise ValueError('A detector input is empty after tokenization')
                item = {'text_sha256': text_hash, 'score': float(np.average(predictions, weights=weights)), 'n_chunks': len(chunks), 'tokens': sum(weights)}
                cached[text_hash] = item
                with cache_path.open('a') as handle:
                    handle.write(json.dumps(item) + '\n')
            item = cached[text_hash]
            results.append({key: value for key, value in row.items() if key != 'text'} | {'detector': detector['name'], **item})
            if position % 25 == 0 or position + 1 == len(records):
                print(f"{detector['name']}: {position + 1}/{len(records)} texts", flush=True)
    finally:
        del model, tokenizer
        gc.collect()
        if selected.type == 'cuda':
            torch.cuda.empty_cache()
    metadata = {'name': detector['name'], 'model_id': detector['model_id'], 'revision': revision, 'id2label': labels, 'ai_label_index': index, 'device': str(selected), 'preprocessing': 'none', 'full_text_coverage': True, 'aggregation': 'token-weighted mean of non-overlapping chunk AI scores', 'max_tokens_including_specials': total_budget, 'cache_fingerprint': fingerprint, 'elapsed_seconds': time.monotonic() - started}
    return (pd.DataFrame(results), metadata)

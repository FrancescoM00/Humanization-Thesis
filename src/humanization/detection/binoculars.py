"""Binoculars with human-only calibration and complete, declared token coverage.

The score is a ratio, not a probability: lower means more AI-like. The numerator
is performer observed-token NLL; the denominator is observer-to-performer
cross-entropy. No classifier threshold or majority vote is silently reused.
"""
from __future__ import annotations
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import numpy as np
import pandas as pd
from humanization.evaluation.detection import _asr_interval, calibrate_low_threshold, summarize_zero_shot
from humanization.datasets.records import digest, read_jsonl, text_records
from humanization.paths import project_root
PROTOCOL = 'binoculars_fullcoverage_fp32_statistics_v1'
UPSTREAM_COMMIT = 'c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8'
DEFAULT_MODELS = {'observer_model_id': 'tiiuae/falcon-7b', 'observer_revision': 'ec89142b67d748a1865ea4451372db8313ada0d8', 'performer_model_id': 'tiiuae/falcon-7b-instruct', 'performer_revision': '8782b5c5d8c9290412416618f36a133653e85285'}

def token_windows(ids, max_tokens=512):
    """One boundary token overlaps; every next-token target is scored once.

    The denominator mask excludes the repeated first token of later windows.
    Its last-position cross-entropy is retained, matching upstream Binoculars
    for a single window. The first token is not an observed-token NLL target.
    """
    if len(ids) < 2 or max_tokens < 2:
        raise ValueError('Binoculars needs at least two tokens and context size >= 2')
    start = 0
    while True:
        end = min(start + int(max_tokens), len(ids))
        yield (ids[start:end], 0 if start == 0 else 1)
        if end == len(ids):
            break
        start = end - 1

def binoculars_components(observer_logits, performer_logits, input_ids, attention_mask, *, cross_entropy_mask=None, positions_per_batch=32):
    """Sufficient statistics, accumulating in FP32 even with FP16 model weights.

    Masks refer to real tokens, including genuine EOS tokens. The NLL mask
    requires both the predictor and target positions to be real. As upstream,
    cross-entropy uses all real positions, including the final position; callers
    may mask repeated context positions when extending to multiple windows.
    Only bounded logit slices move between model devices.
    """
    import torch
    import torch.nn.functional as F
    if observer_logits.ndim != 3 or observer_logits.shape != performer_logits.shape or tuple(input_ids.shape) != tuple(observer_logits.shape[:2]) or (tuple(attention_mask.shape) != tuple(input_ids.shape)):
        raise ValueError('Inconsistent model logits, input IDs or masks')
    if positions_per_batch < 1:
        raise ValueError('positions_per_batch must be positive')
    mask = attention_mask.bool()
    causal_mask = mask[:, :-1] & mask[:, 1:]
    cross_mask = mask if cross_entropy_mask is None else mask & cross_entropy_mask.bool()
    if cross_mask.shape != mask.shape:
        raise ValueError('Cross-entropy mask shape differs from attention mask')
    nll_sum = cross_sum = 0.0
    nll_tokens = int(causal_mask.sum().item())
    cross_tokens = int(cross_mask.sum().item())
    if nll_tokens == 0 or cross_tokens == 0:
        raise ValueError('Binoculars has no valid token positions to score')
    with torch.inference_mode():
        for start in range(0, input_ids.shape[1], positions_per_batch):
            end = min(start + positions_per_batch, input_ids.shape[1])
            p = observer_logits[:, start:end, :].float()
            q = performer_logits[:, start:end, :].to(device=p.device, dtype=torch.float32)
            cross = -(p.softmax(-1) * q.log_softmax(-1)).sum(-1)
            cross_sum += float((cross * cross_mask[:, start:end].to(p.device)).sum().item())
            target_end = min(end, input_ids.shape[1] - 1)
            if start < target_end:
                q_nll = performer_logits[:, start:target_end, :].float()
                labels = input_ids[:, start + 1:target_end + 1].to(q_nll.device)
                loss = F.cross_entropy(q_nll.transpose(1, 2), labels, reduction='none')
                nll_sum += float((loss * causal_mask[:, start:target_end].to(q_nll.device)).sum().item())
    if not all((math.isfinite(x) for x in (nll_sum, cross_sum))) or cross_sum <= 0:
        raise ValueError('Binoculars produced non-finite or non-positive statistics')
    return {'nll_sum': nll_sum, 'nll_tokens': nll_tokens, 'cross_entropy_sum': cross_sum, 'cross_entropy_tokens': cross_tokens}

def ratio_from_components(components):
    nll_tokens = sum((p['nll_tokens'] for p in components))
    cross_tokens = sum((p['cross_entropy_tokens'] for p in components))
    if min(nll_tokens, cross_tokens) <= 0:
        raise ValueError('Cannot aggregate an empty Binoculars document')
    nll = sum((p['nll_sum'] for p in components)) / nll_tokens
    cross = sum((p['cross_entropy_sum'] for p in components)) / cross_tokens
    score = nll / cross
    if not all((math.isfinite(x) and x > 0 for x in (nll, cross, score))):
        raise ValueError('Invalid Binoculars ratio')
    return {'score': score, 'performer_nll': nll, 'observer_performer_cross_entropy': cross, 'nll_tokens': nll_tokens, 'cross_entropy_tokens': cross_tokens}

def _settings(detector):
    settings = dict(DEFAULT_MODELS, **detector)
    if settings.get('kind', 'binoculars') != 'binoculars':
        raise ValueError('This runner implements Binoculars only')
    if settings.get('load_in_4bit') or settings.get('load_in_8bit'):
        raise ValueError('Quantized Binoculars is a different unvalidated condition')
    for field in ['observer_revision', 'performer_revision']:
        if not re.fullmatch('[0-9a-f]{40}', str(settings[field])):
            raise ValueError('Pin each Binoculars model to an immutable commit')
    if settings.get('precision', 'float16') not in {'float16', 'bfloat16'}:
        raise ValueError('Use float16 or bfloat16 model weights')
    if int(settings.get('max_tokens', 512)) < 2:
        raise ValueError('max_tokens must be >= 2')
    return settings

def cached_weight_snapshot(repo_id, revision, roots):
    """Read a pinned snapshot only when config and every weight shard exist."""
    for root in roots:
        snapshot = Path(root) / ('models--' + repo_id.replace('/', '--')) / 'snapshots' / revision
        if not (snapshot / 'config.json').is_file():
            continue
        index = snapshot / 'model.safetensors.index.json'
        if index.is_file():
            try:
                names = set(json.loads(index.read_text())['weight_map'].values())
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if names and all((Path(n).name == n and (snapshot / n).is_file() for n in names)):
                return snapshot
        elif (snapshot / 'model.safetensors').is_file():
            return snapshot
    return None

def select_writable_cache(candidates, *, missing_models, min_free_bytes=None):
    """Probe another filesystem rather than changing permissions on a mount."""
    needed = missing_models * 15 * 2 ** 30 if min_free_bytes is None else min_free_bytes
    failures = []
    for candidate in candidates:
        candidate = Path(candidate)
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / '.humanization_write_probe'
            probe.write_text('ok')
            probe.unlink()
            free = shutil.disk_usage(candidate).free
            if free < needed:
                failures.append(f'{candidate}: insufficient free space')
                continue
            return candidate
        except OSError as exc:
            failures.append(f'{candidate}: {exc.strerror} (errno {exc.errno})')
    raise RuntimeError('No writable model cache with sufficient space. ' + '; '.join(failures))

def _prepare_model_storage(settings):
    initial = Path(os.environ.get('HUMANIZATION_HF_READ_CACHE') or os.environ.get('HF_HUB_CACHE') or str(Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))) / 'hub'))
    override = os.environ.get('HUMANIZATION_HF_WRITE_CACHE')
    fallbacks = [Path.home() / '.cache/huggingface/hub', project_root() / '.cache/huggingface/hub', Path(tempfile.gettempdir()) / 'humanization/huggingface/hub']
    roots = list(dict.fromkeys([initial, *([Path(override)] if override else []), *fallbacks]))
    snapshots = {p: cached_weight_snapshot(settings[p + '_model_id'], settings[p + '_revision'], roots) for p in ('observer', 'performer')}
    candidates = [Path(override)] if override else [initial, *fallbacks]
    writable = select_writable_cache(candidates, missing_models=sum((p is None for p in snapshots.values())))
    os.environ['HF_HUB_CACHE'] = str(writable)
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    scratch = writable / '_tmp'
    scratch.mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(scratch)
    for name in ('TRANSFORMERS_CACHE', 'HUGGINGFACE_HUB_CACHE'):
        if name in os.environ:
            os.environ[name] = str(writable)
    print(f'Model cache: {writable}; complete cached checkpoints: {sum((p is not None for p in snapshots.values()))}/2', flush=True)
    return (writable, snapshots)

def _load_pair(settings, observer_device, performer_device):
    import torch
    devices = [torch.device(observer_device), torch.device(performer_device)]
    if any((d.type != 'cuda' or d.index is None for d in devices)) or devices[0] == devices[1]:
        raise ValueError('Binoculars Falcon 7B requires two distinct indexed CUDA devices, e.g. cuda:0 and cuda:1')
    if not torch.cuda.is_available() or max((d.index for d in devices)) >= torch.cuda.device_count():
        raise RuntimeError('Select two CUDA-capable GPUs; both Falcon 7B models must fit simultaneously')
    dtype = getattr(torch, settings.get('precision', 'float16'))
    for device in devices:
        with torch.cuda.device(device):
            free, _ = torch.cuda.mem_get_info()
            if free < 14 * 2 ** 30:
                raise RuntimeError(f'{device}: free at least 14 GiB before loading an unquantized Falcon 7B')
            if dtype == torch.bfloat16 and (not torch.cuda.is_bf16_supported()):
                raise RuntimeError('This GPU does not support the requested bfloat16 condition; use explicit float16 on T4')
    writable, snapshots = _prepare_model_storage(settings)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def source(prefix):
        if snapshots[prefix] is not None:
            return (str(snapshots[prefix]), {'local_files_only': True})
        return (settings[prefix + '_model_id'], {'revision': settings[prefix + '_revision']})
    tokenizers = [AutoTokenizer.from_pretrained(source(prefix)[0], cache_dir=str(writable), trust_remote_code=False, **source(prefix)[1]) for prefix in ['observer', 'performer']]
    if tokenizers[0].get_vocab() != tokenizers[1].get_vocab():
        raise ValueError('Observer and performer vocabularies/token indices differ')
    models = []
    try:
        for prefix, device in zip(['observer', 'performer'], devices):
            location, kwargs = source(prefix)
            print(f'{prefix}: ' + ('reuse complete pinned snapshot' if snapshots[prefix] else 'download missing checkpoint'), flush=True)
            model = AutoModelForCausalLM.from_pretrained(location, **kwargs, cache_dir=str(writable), torch_dtype=dtype, device_map={'': device.index}, low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=False, attn_implementation='sdpa').eval()
            models.append(model)
        return (tokenizers, models, devices)
    except Exception:
        models.clear()
        gc.collect()
        torch.cuda.empty_cache()
        raise

def score_binoculars(records, detector, output, *, observer_device='cuda:0', performer_device='cuda:1'):
    import torch
    settings = _settings(detector)
    scoring = {k: settings[k] for k in DEFAULT_MODELS}
    scoring.update(protocol=PROTOCOL, precision=settings.get('precision', 'float16'), max_tokens=int(settings.get('max_tokens', 512)), attention='sdpa', code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    fingerprint = digest(scoring)
    cache_dir = Path(output) / 'zero_shot_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'{fingerprint}.jsonl'
    cached = {}
    if cache_path.exists():
        entries = read_jsonl(cache_path)
        if not entries or entries[0] != dict(type='binoculars_cache_v1', **scoring):
            raise ValueError('Binoculars cache header differs from scoring configuration')
        cached = {r['text_sha256']: r for r in entries[1:]}
    else:
        cache_path.write_text(json.dumps(dict(type='binoculars_cache_v1', **scoring)) + '\n')
    hashes = [hashlib.sha256(r['text'].encode()).hexdigest() for r in records]
    tokenizers = models = devices = None
    started = time.monotonic()
    try:
        if any((h not in cached for h in hashes)):
            tokenizers, models, devices = _load_pair(settings, observer_device, performer_device)
        results = []
        for position, (record, text_hash) in enumerate(zip(records, hashes), 1):
            if text_hash not in cached:
                ids = tokenizers[0].encode(record['text'], add_special_tokens=True)
                if ids != tokenizers[1].encode(record['text'], add_special_tokens=True):
                    raise ValueError('Observer and performer tokenization differs for the same plaintext')
                context_limits = [int(getattr(m.config, 'max_position_embeddings', scoring['max_tokens'])) for m in models]
                budget = min(scoring['max_tokens'], *context_limits)
                components = []
                for chunk, skip_cross in token_windows(ids, budget):
                    input_ids = torch.tensor([chunk], dtype=torch.long)
                    mask = torch.ones_like(input_ids)
                    with torch.inference_mode():
                        observer = models[0](input_ids=input_ids.to(devices[0]), attention_mask=mask.to(devices[0]), use_cache=False).logits
                        performer = models[1](input_ids=input_ids.to(devices[1]), attention_mask=mask.to(devices[1]), use_cache=False).logits
                        cross_mask = mask.clone()
                        cross_mask[:, :skip_cross] = 0
                        components.append(binoculars_components(observer, performer, input_ids, mask, cross_entropy_mask=cross_mask, positions_per_batch=int(settings.get('positions_per_batch', 32))))
                        del observer, performer
                item = {'text_sha256': text_hash, **ratio_from_components(components), 'n_chunks': len(components), 'tokens': len(ids)}
                if item['nll_tokens'] != len(ids) - 1 or item['cross_entropy_tokens'] != len(ids):
                    raise RuntimeError('Binoculars windowing failed complete token coverage')
                cached[text_hash] = item
                with cache_path.open('a') as handle:
                    handle.write(json.dumps(item) + '\n')
            results.append({k: v for k, v in record.items() if k != 'text'} | {'detector': detector['name'], **cached[text_hash]})
            if position % 25 == 0 or position == len(records):
                print(f"{detector['name']}: {position}/{len(records)} texts", flush=True)
    finally:
        del models, tokenizers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return (pd.DataFrame(results), dict(scoring, name=detector['name'], cache_fingerprint=fingerprint, observer_device=observer_device, performer_device=performer_device, elapsed_seconds=time.monotonic() - started, full_text_coverage=True, score_orientation='lower_is_more_ai_like', score_is_probability=False, upstream_source_commit=UPSTREAM_COMMIT, aggregation='ratio of global token means; one context token overlaps; denominator boundary counted once'))

def run(cohort_path, rewrites_path, config_path, output, *, observer_device='cuda:0', performer_device='cuda:1', analysis_only=False, only=None, evaluate=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cohort, rewrites = (read_jsonl(cohort_path), read_jsonl(rewrites_path))
    config = json.loads(Path(config_path).read_text())
    configured = config.get('zero_shot_detectors', [])
    if only:
        unknown = set(only) - {d['name'] for d in configured}
        if unknown:
            raise ValueError(f'Unknown zero-shot detectors: {sorted(unknown)}')
        configured = [dict(d, enabled=True) for d in configured if d['name'] in only]
    else:
        configured = [d for d in configured if d.get('enabled', True)]
    if not configured:
        raise ValueError('No zero-shot detectors selected')
    config = dict(config, zero_shot_detectors=configured)
    records = text_records(cohort, rewrites)
    input_hashes = {key: hashlib.sha256(Path(path).read_bytes()).hexdigest() for key, path in [('cohort_sha256', cohort_path), ('rewrites_sha256', rewrites_path), ('config_sha256', config_path)]}
    input_hashes['selected_detectors'] = [d['name'] for d in configured]
    manifest_path = output / 'zero_shot_run.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and any((previous.get(k) != v for k, v in input_hashes.items())):
        raise ValueError('Existing zero-shot inputs/config differ; use a new output directory')
    details = []
    if analysis_only:
        if not previous or previous.get('status') != 'completed':
            raise ValueError('Analysis-only requires a completed zero-shot inference manifest')
        scores = pd.read_csv(output / 'zero_shot_scores.csv', dtype={'source_id': str})
        details = previous['detectors']
    else:
        manifest_path.write_text(json.dumps(dict(input_hashes, status='running', n_text_records_per_detector=len(records)), indent=2) + '\n')
        frames = []
        try:
            for detector in configured:
                frame, metadata = score_binoculars(records, detector, output, observer_device=observer_device, performer_device=performer_device)
                frames.append(frame)
                details.append(metadata)
            scores = pd.concat(frames, ignore_index=True)
            scores.to_csv(output / 'zero_shot_scores.csv', index=False, lineterminator='\n')
        except Exception as exc:
            manifest_path.write_text(json.dumps(dict(input_hashes, status='failed', error_type=type(exc).__name__, error='Zero-shot scoring failed; inspect the notebook traceback. No credential-bearing exception is exported.', n_text_records_per_detector=len(records)), indent=2) + '\n')
            raise
    if not evaluate:
        return scores
    thresholds, summary = summarize_zero_shot(cohort, rewrites, scores, config)
    (output / 'zero_shot_thresholds.json').write_text(json.dumps(thresholds, indent=2) + '\n')
    summary.to_csv(output / 'zero_shot_summary.csv', index=False, lineterminator='\n')
    versions = {}
    for package in ['torch', 'transformers', 'accelerate']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = dict(input_hashes, status='completed', analysis_only=analysis_only, detectors=details, n_text_records_per_detector=len(records), versions=versions, asr_denominator='originals detected AI before rewriting; failed generations remain failures', calibration='human-only, source-group-disjoint; raw ratio < threshold, ties human', majority_vote_created=False, published_threshold_used=False, score_is_probability=False, naturalness_measured=False, intervals='source-group bootstrap with fixed checkpoints and calibration thresholds')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(summary[summary.domain == 'overall'].to_string(index=False), flush=True)
    return summary

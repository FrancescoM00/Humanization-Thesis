"""Populate the unchanged NLI probability cache with validated float32 ORT.

Sort globally by token length to avoid padding work. This changes scheduling,
not evidence retrieval, labels, model weights, or aggregation.
"""
import argparse
import hashlib
import json
from pathlib import Path
from humanization import resources as resource
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

def _argument_error(message):
    raise ValueError(message)

def run(args):
    start = time.monotonic()
    jobs = [json.loads(x) for x in (args.data_dir / 'nli_jobs.jsonl').read_text().splitlines()]
    cache_path = args.data_dir / 'nli_probability_cache.jsonl'
    lines = cache_path.read_text().splitlines()
    header = json.loads(lines[0])
    if header['model_path'] != str(args.model.resolve()):
        raise ValueError('Probability cache checkpoint does not match requested model')
    if header['config_sha256'] != hashlib.sha256((args.model / 'config.json').read_bytes()).hexdigest():
        raise ValueError('Probability cache checkpoint config changed')
    cached = {row['key']: row['probabilities'] for row in map(json.loads, lines[1:])}
    initial_cached = sum((job['key'] in cached for job in jobs))
    reference_path = args.data_dir / 'nli_pytorch_reference.jsonl'
    if not reference_path.is_file():
        raise ValueError('Initialize an independent native PyTorch reference with src/humanization/semantics/workflows/initialize_nli_reference_cache.py')
    reference_lines = reference_path.read_text().splitlines()
    if json.loads(reference_lines[0]) != header:
        raise ValueError('Native reference cache belongs to a different checkpoint')
    native_reference = {row['key']: row['probabilities'] for row in map(json.loads, reference_lines[1:])}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=['CPUExecutionProvider'])
    input_names = {x.name for x in session.get_inputs()}

    def predict(batch):
        encoded = tokenizer([x['premise'] for x in batch], [x['hypothesis'] for x in batch], padding=True, truncation=False, return_tensors='np')
        if encoded['input_ids'].shape[1] > 384:
            raise ValueError('Planned evidence exceeds the NLI token budget; refusing truncation')
        logits = session.run(None, {k: v for k, v in encoded.items() if k in input_names})[0]
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)
    references = [x for x in jobs if x['key'] in native_reference]
    if not references:
        raise ValueError('Native reference does not cover any of the planned evidence pairs')
    rng = np.random.default_rng(42)
    margin = lambda x: sorted(native_reference[x['key']])[-1] - sorted(native_reference[x['key']])[-2]
    selected = sorted(references, key=margin)[:32]
    selected += [references[i] for i in rng.choice(len(references), min(96, len(references)), replace=False)]
    selected = list({x['key']: x for x in selected}.values())
    differences, classes, threshold = ([], [], [])
    config = json.loads((args.model / 'config.json').read_text())
    entailment_index = next((int(i) for i, n in config['id2label'].items() if n.lower() == 'entailment'))
    for i in range(0, len(selected), args.batch_size):
        batch = selected[i:i + args.batch_size]
        actual = predict(batch)
        expected = np.array([native_reference[x['key']] for x in batch])
        differences.extend(np.abs(actual - expected).max(axis=1).tolist())
        classes.extend((actual.argmax(-1) == expected.argmax(-1)).tolist())
        threshold.extend(((actual[:, entailment_index] >= 0.5) == (expected[:, entailment_index] >= 0.5)).tolist())
    validation = {'n_real_cached_pairs': len(selected), 'max_probability_abs_difference': max(differences), 'mean_probability_max_abs_difference': float(np.mean(differences)), 'argmax_agreement': float(np.mean(classes)), 'entailment_threshold_0_5_agreement': float(np.mean(threshold)), 'tolerance': 0.0001, 'precision': 'float32', 'quantized': False, 'native_reference_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest()}
    (args.output / 'nli_onnx_real_pair_parity.json').write_text(json.dumps(validation, indent=2) + '\n')
    if max(differences) > 0.0001 or not all(classes) or (not all(threshold)):
        raise ValueError('Float32 ONNX backend did not pass prespecified parity checks')
    pending = [x for x in jobs if x['key'] not in cached]
    for job in pending:
        job['tokens'] = len(tokenizer.encode(job['premise'], add_special_tokens=False)) + len(tokenizer.encode(job['hypothesis'], add_special_tokens=False))
    pending.sort(key=lambda x: x['tokens'])
    processed = 0
    for i in range(0, len(pending), args.batch_size):
        batch = pending[i:i + args.batch_size]
        probabilities = predict(batch)
        with cache_path.open('a') as handle:
            for job, prob in zip(batch, probabilities):
                handle.write(json.dumps({'key': job['key'], 'probabilities': [float(x) for x in prob]}, separators=(',', ':')) + '\n')
        processed += len(batch)
        if processed % 200 < args.batch_size or processed == len(pending):
            progress = {'phase': 'unique_evidence_pairs', 'completed': initial_cached + processed, 'total': len(jobs), 'elapsed_seconds': time.monotonic() - start, 'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'max_pair_tokens_this_batch': batch[-1]['tokens'] + tokenizer.num_special_tokens_to_add(pair=True)}
            (args.output / 'nli_precompute_progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    result = {'status': 'executed', 'backend': 'onnxruntime_cpu_float32', 'unique_model_pairs': len(jobs), 'initial_cached_pairs': initial_cached, 'new_onnx_pairs': processed, 'elapsed_seconds': time.monotonic() - start, 'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'threads': args.threads, 'batch_size': args.batch_size, 'batching': 'global_token_length_sorted', 'onnx_sha256': hashlib.sha256(args.onnx.read_bytes()).hexdigest(), 'pairs_sha256': hashlib.sha256((args.data_dir / 'pairs.jsonl').read_bytes()).hexdigest(), 'jobs_sha256': hashlib.sha256((args.data_dir / 'nli_jobs.jsonl').read_bytes()).hexdigest(), 'parity': validation}
    (args.output / 'nli_precompute_run.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'model': Path('data/models/nli'), 'onnx': Path('data/models/nli_onnx/model.onnx'), 'data_dir': Path('data/processed/semantic_baseline'), 'output': Path('results/new_runs/03_semantics/baseline'), 'threads': 4, 'batch_size': 8}
    values.update(overrides)
    return SimpleNamespace(**values)

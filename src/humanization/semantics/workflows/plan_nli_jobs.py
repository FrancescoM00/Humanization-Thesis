"""Enumerate exact NLI evidence jobs without running the neural model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from transformers import AutoTokenizer
from humanization.semantics.metrics import _cached_chunks, _local, _nli_jobs

def _argument_error(message):
    raise ValueError(message)

def run(args):
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    pairs_path = args.data_dir / 'pairs.jsonl'
    records = [json.loads(x) for x in pairs_path.read_text().splitlines()]
    alignment = json.loads((args.data_dir / 'alignment.json').read_text())
    jobs = {}
    for row in records:
        item = _cached_chunks(row, alignment)
        if item is None:
            raise ValueError('Run embedding stage for all prepared pairs before planning NLI')
        groups = [_nli_jobs(item['original_chunks'], item['modified_chunks'], item['forward_candidates'], tokenizer, 384), _nli_jobs(item['modified_chunks'], item['original_chunks'], item['reverse_candidates'], tokenizer, 384)]
        local = _local(row)
        if local:
            groups += [_nli_jobs([local[0]], [local[1]], [[0]], tokenizer, 384), _nli_jobs([local[1]], [local[0]], [[0]], tokenizer, 384)]
        for group in groups:
            for alternatives in group:
                for pair in alternatives:
                    key = hashlib.sha256(json.dumps(pair, ensure_ascii=False).encode()).hexdigest()
                    jobs.setdefault(key, {'key': key, 'premise': pair[0], 'hypothesis': pair[1]})
    destination = args.data_dir / 'nli_jobs.jsonl'
    destination.write_text(''.join((json.dumps(j, ensure_ascii=False) + '\n' for j in jobs.values())))
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {'document_pairs': len(records), 'unique_model_pairs': len(jobs), 'pairs_sha256': hashlib.sha256(pairs_path.read_bytes()).hexdigest(), 'jobs_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'elapsed_seconds': time.monotonic() - start}
    (args.output / 'nli_plan.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'data_dir': ROOT / 'data/semantic_full', 'output': ROOT / 'results/new_runs/03_semantics/baseline', 'model': ROOT / 'data/models/nli'}
    values.update(overrides)
    return SimpleNamespace(**values)

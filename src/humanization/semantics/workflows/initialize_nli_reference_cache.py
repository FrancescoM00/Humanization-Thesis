"""Score a seeded native-PyTorch calibration subset before float32 ORT use."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def _argument_error(message):
    raise ValueError(message)

def run(args):
    reference_path = args.data_dir / 'nli_pytorch_reference.jsonl'
    if reference_path.exists():
        raise FileExistsError('Native reference cache already exists; preserve it for resume/parity checks')
    model_path = str(args.model.resolve())
    weights = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(args.model.glob('*')) if p.suffix in {'.safetensors', '.bin'}]
    header = {'type': 'nli_probability_cache_v1', 'model_path': model_path, 'config_sha256': hashlib.sha256((args.model / 'config.json').read_bytes()).hexdigest(), 'weight_files': [list(x) for x in weights]}
    jobs = [json.loads(x) for x in (args.data_dir / 'nli_jobs.jsonl').read_text().splitlines()]
    rng = np.random.default_rng(42)
    selected = [jobs[i] for i in rng.choice(len(jobs), min(args.n, len(jobs)), replace=False)]
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True).eval()
    records = []
    for i in range(0, len(selected), 8):
        batch = selected[i:i + 8]
        encoded = tokenizer([r['premise'] for r in batch], [r['hypothesis'] for r in batch], padding=True, truncation=False, return_tensors='pt')
        if encoded['input_ids'].shape[1] > 384:
            raise ValueError('Planned calibration input exceeds NLI budget')
        with torch.inference_mode():
            values = model(**encoded).logits.softmax(-1).numpy()
        records.extend(({'key': r['key'], 'probabilities': [float(v) for v in p]} for r, p in zip(batch, values)))
    contents = json.dumps(header, sort_keys=True) + '\n' + ''.join((json.dumps(r, separators=(',', ':')) + '\n' for r in records))
    reference_path.write_text(contents)
    full_cache = args.data_dir / 'nli_probability_cache.jsonl'
    if not full_cache.exists():
        full_cache.write_text(contents)
    print(f'Saved {len(records)} native reference predictions; existing full cache preserved.')

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'data_dir': Path('data/processed/semantic_baseline'), 'model': Path('data/models/nli'), 'n': 128}
    values.update(overrides)
    return SimpleNamespace(**values)

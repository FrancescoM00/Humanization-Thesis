"""Esperimento semantico con fasi isolate; le fasi neurali non eseguite non vengono mai imputate."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from humanization import resources as resource
import subprocess
import sys
import time
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def prepare(args):
    from humanization.semantics.perturbations import select_applicable_records, generate_perturbation_pairs, sanity_fixtures
    records = read_jsonl(args.input)
    selected, eligibility = select_applicable_records(records, n=args.n, seed=args.seed)
    write_json(args.output / 'eligibility_manifest.json', eligibility)
    if not eligibility['target_count_met']:
        raise RuntimeError(f'Only {len(selected)} documents support all controlled changes; requested {args.n}. Inspect eligibility_manifest.json.')
    generated = generate_perturbation_pairs(selected, seed=args.seed)
    if args.entity_policy == 'clean':
        clean = generate_perturbation_pairs(selected, seed=args.seed, conditions=('entity_substitution_clean',))
        for pair in clean:
            pair['perturbation'] = 'entity_substitution'
        clean_by_source = {pair['source_id']: pair for pair in clean}
        generated = [clean_by_source[pair['source_id']] if pair['perturbation'] == 'entity_substitution' else pair for pair in generated]
    skipped = [p for p in generated if p['status'] != 'applied']
    pairs = [dict(p, dataset='controlled', id=p['pair_id']) for p in generated if p['status'] == 'applied']
    controls = []
    for pair in pairs:
        if pair.get('target_original'):
            text = pair['target_original']
            controls.append({**pair, 'id': pair['id'] + ':local_identity', 'pair_id': pair['id'] + ':local_identity', 'original': text, 'modified': text, 'target_original': text, 'target_modified': text, 'dataset': 'local_control', 'perturbation': 'identity_for_' + pair['perturbation'], 'metadata': {'control_for': pair['id'], 'semantic_interpretation': 'exact_identity'}})
    pairs.extend(controls)
    pairs.extend((dict(p, id=p['pair_id'], dataset='manual_sanity') for p in sanity_fixtures()))
    opai_status = {'status': 'unrun', 'reason': 'Official OpAI-Bench files absent; Hugging Face access is required. No substitute data used.'}
    if args.opai_file:
        from humanization.datasets.opai import read_opai_file, align_opai_records
        provenance_path = args.opai_file.with_suffix('.provenance.json')
        source_metadata = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {'method': 'local_official_export'}
        actual_sha = hashlib.sha256(args.opai_file.read_bytes()).hexdigest()
        if source_metadata.get('sha256', actual_sha) != actual_sha:
            raise ValueError('OpAI source does not match its provenance checksum')
        source_metadata['sha256'] = actual_sha
        opai, manifest = align_opai_records(read_opai_file(args.opai_file), n_per_operation=args.n, seed=args.seed, source_metadata=source_metadata)
        pairs.extend((dict(p, dataset='opai', status='applied') for p in opai))
        write_json(args.output / 'opai_manifest.json', manifest)
        opai_status = {'status': 'prepared', 'n_pairs': len(opai)}
    args.data_dir.mkdir(parents=True, exist_ok=True)
    pair_path = args.data_dir / 'pairs.jsonl'
    pair_path.write_text(''.join((json.dumps(p, ensure_ascii=False) + '\n' for p in pairs)), encoding='utf-8')
    (args.data_dir / 'skipped.jsonl').write_text(''.join((json.dumps(p, ensure_ascii=False) + '\n' for p in skipped)), encoding='utf-8')
    counts = Counter((p['perturbation'] for p in pairs if p['dataset'] == 'controlled'))
    manifest = {'source_file': str(args.input), 'source_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(), 'pairs_sha256': hashlib.sha256(pair_path.read_bytes()).hexdigest(), 'seed': args.seed, 'n_source_machine_documents': len(selected), 'domain_counts': dict(Counter((r['domain'] for r in selected))), 'generator_counts': dict(Counter((r['generator'] for r in selected))), 'applied_counts': dict(counts), 'skipped_counts': dict(Counter((p['perturbation'] for p in skipped))), 'skip_reasons': dict(Counter((p['metadata']['reason'] for p in skipped))), 'n_local_controls': len(controls), 'n_manual_sanity_fixtures': len(sanity_fixtures()), 'eligibility_selected_cohort': True, 'eligible_source_count': eligibility['eligible_count'], 'opai': opai_status, 'semantic_labels_are_automatic_hypotheses': True, 'entity_policy': args.entity_policy, 'source_cohort_policy': 'fixed_v1_eligibility', 'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / 'src/humanization/semantics').glob('*.py'))}}
    write_json(args.output / 'preparation_manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)

def stage(args):
    from humanization.semantics.metrics import lexical_scores, embedding_scores, bert_scores, nli_scores
    pairs = read_jsonl(args.data_dir / 'pairs.jsonl')
    start = time.monotonic()

    def progress(completed, total):
        write_json(args.output / (args.stage + '_progress.json'), {'completed_pairs': completed, 'total_pairs': total, 'elapsed_seconds': round(time.monotonic() - start, 2), 'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
    if args.stage == 'lexical':
        frame = lexical_scores(pairs)
    elif args.stage == 'embedding':
        frame, alignment = embedding_scores(pairs, args.models / 'embedding', n_threads=args.threads, progress_callback=progress, device=args.device)
        write_json(args.data_dir / 'alignment.json', alignment)
    elif args.stage == 'bertscore':
        frame = bert_scores(pairs, args.models / 'bertscore', batch_size=4, n_threads=args.threads, progress_callback=progress, device=args.device)
    else:
        cache = args.data_dir / 'alignment.json'
        if not cache.is_file():
            raise RuntimeError('Run embedding stage first; NLI requires its bounded top-k sentence retrieval cache')
        frame = nli_scores(pairs, args.models / 'nli', alignment_cache=json.loads(cache.read_text()), batch_size=args.nli_batch_size, n_threads=args.threads, probability_cache_path=args.data_dir / 'nli_probability_cache.jsonl', progress_callback=progress, cache_only=args.require_nli_cache, device=args.device)
    frame.to_csv(args.output / (args.stage + '.csv'), index=False)
    versions = {}
    for name in ['torch', 'transformers', 'sentence-transformers', 'bert-score', 'rouge-score', 'sacrebleu', 'onnx', 'onnxruntime', 'optimum-onnx']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {'stage': args.stage, 'status': 'executed', 'n_pairs': len(frame), 'elapsed_seconds': time.monotonic() - start, 'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'pairs_sha256': hashlib.sha256((args.data_dir / 'pairs.jsonl').read_bytes()).hexdigest(), 'versions': versions, 'device': 'cpu' if args.stage == 'lexical' else args.device, 'threads': args.threads, 'batch_size': args.nli_batch_size if args.stage == 'nli' else 4 if args.stage == 'bertscore' else None, 'model_directory': None if args.stage == 'lexical' else str(args.models / args.stage), 'metrics_code_sha256': hashlib.sha256((ROOT / 'src/humanization/semantics/metrics.py').read_bytes()).hexdigest()}
    if args.stage == 'nli' and (args.output / 'nli_precompute_run.json').is_file():
        precompute = json.loads((args.output / 'nli_precompute_run.json').read_text())
        metadata['precomputation'] = precompute
        metadata['aggregation_elapsed_seconds'] = metadata['elapsed_seconds']
        history_path = args.output / 'nli_resumption_history.json'
        if history_path.is_file():
            history = json.loads(history_path.read_text())
            metadata['resumption_history'] = history
            metadata['elapsed_seconds'] += history.get('serial_onnx_completed', {}).get('elapsed_seconds', 0)
            metadata['elapsed_seconds'] += (history.get('initial_progress') or {}).get('elapsed_seconds', 0)
        metadata['elapsed_seconds'] += precompute['elapsed_seconds']
        metadata['peak_rss_mib'] = max(metadata['peak_rss_mib'], precompute['peak_rss_mib'])
        metadata['backend'] = 'validated_float32_onnx_cache_with_native_pytorch_reference'
        metadata['cache_only'] = args.require_nli_cache
        metadata['new_native_predictions_during_aggregation'] = int(frame.nli_new_pairs_scored.sum())
    write_json(args.output / (args.stage + '_run.json'), metadata)
    pending_path = args.output / 'pending_stages.json'
    if pending_path.is_file():
        pending = json.loads(pending_path.read_text())
        write_json(pending_path, [item for item in pending if item['stage'] != args.stage])

def _argument_error(message):
    raise ValueError(message)

def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'all':
        pending = []
        stages = ['prepare', 'lexical', 'embedding', 'bertscore', 'nli', 'report']
        common = ['--input', str(args.input), '--data-dir', str(args.data_dir), '--output', str(args.output), '--models', str(args.models), '--n', str(args.n), '--seed', str(args.seed), '--threads', str(args.threads), '--nli-batch-size', str(args.nli_batch_size), '--device', args.device]
        common += ['--entity-policy', args.entity_policy]
        if args.opai_file:
            common += ['--opai-file', str(args.opai_file)]
        if args.require_nli_cache:
            common += ['--require-nli-cache']
        for name in stages:
            if name in {'embedding', 'bertscore', 'nli'}:
                directory = args.models / name
                if not (directory / 'config.json').is_file() or not any(((directory / p).is_file() for p in ['model.safetensors', 'pytorch_model.bin'])):
                    pending.append({'stage': name, 'reason': 'Pretrained model files unavailable; metrics NOT computed'})
                    continue
                if name == 'nli' and any((p['stage'] == 'embedding' for p in pending)):
                    pending.append({'stage': name, 'reason': 'Sentence evidence retrieval stage unavailable'})
                    continue
            if name == 'report':
                write_json(args.output / 'pending_stages.json', pending)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--stage', name, *common], check=True)
        return
    if args.stage == 'prepare':
        prepare(args)
    elif args.stage == 'report':
        from humanization.evaluation.semantic_reporting import write_semantic_report
        write_semantic_report(read_jsonl(args.data_dir / 'pairs.jsonl'), args.output, args.seed)
    else:
        stage(args)

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'stage': 'all', 'input': ROOT / 'data/processed/generated/baseline/samples.jsonl', 'opai_file': None, 'data_dir': ROOT / 'data/processed/semantic_initial', 'output': ROOT / 'results/new_runs/03_semantics/baseline', 'models': ROOT / 'data/models', 'n': 200, 'seed': 42, 'entity_policy': 'v1', 'threads': 2, 'device': 'cpu', 'nli_batch_size': 8, 'require_nli_cache': False}
    values.update(overrides)
    return SimpleNamespace(**values)

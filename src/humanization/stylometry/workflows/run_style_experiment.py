"""Prepara dati appaiati, estrae feature fisse, valuta e salva i risultati."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
import pandas as pd
from humanization.evaluation.stylometric import effect_tables, evaluate, shortlist_features
from humanization.stylometry.features import FEATURE_GROUPS, FEATURE_SCHEMA, LENGTH_SENSITIVE_FEATURES, WIKIPEDIA_FEATURES, extract_records
from humanization.evaluation.reporting import write_report

def _argument_error(message):
    raise ValueError(message)

def run(args):
    from humanization.datasets.mgtbench import prepare_dataset
    start = time.monotonic()
    utc_started = datetime.now(timezone.utc).isoformat()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print('Preparing a balanced, source-paired sample...', flush=True)
    samples, manifest = prepare_dataset(args.raw_dir.resolve(), ROOT / ('data/processed/generated/pilot' if args.keep_incomplete else 'data/processed/generated/baseline'), n_pairs=args.pairs, seed=args.seed, clean_sentence_ends=not args.keep_incomplete)
    (out / 'data_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    quality = samples[['domain', 'generator', 'label']].copy()
    quality['whitespace_words'] = samples.text.str.split().str.len()
    quality['nonterminal_ending'] = ~samples.text.str.contains('[.!?…][\\s"\'’”\\)\\]]*$', regex=True)
    quality.groupby(['domain', 'label', 'generator']).agg(n=('label', 'size'), median_whitespace_words=('whitespace_words', 'median'), mean_whitespace_words=('whitespace_words', 'mean'), non_sentence_endings=('nonterminal_ending', 'sum')).to_csv(out / 'data_quality.csv')
    sample_hash = hashlib.sha256(samples.to_json(orient='records').encode()).hexdigest()
    versions = {name: importlib.metadata.version(name) for name in ['spacy', 'en-core-web-sm', 'textstat', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'nltk']}
    schema_hash = hashlib.sha256(json.dumps(FEATURE_SCHEMA, sort_keys=True).encode()).hexdigest()
    extractor_hash = hashlib.sha256((ROOT / 'src/humanization/stylometry/features.py').read_bytes()).hexdigest()
    cache_info = {'sample_sha256': sample_hash, 'schema_sha256': schema_hash, 'extractor_sha256': extractor_hash, 'model': args.model, 'versions': versions}
    cache_file, cache_meta = (out / 'features.csv', out / 'features_cache.json')
    if args.reuse_features and cache_file.exists() and cache_meta.exists():
        if json.loads(cache_meta.read_text()) != cache_info:
            raise ValueError('Feature cache differs from sample/extractor/model/versions; rerun without --reuse-features')
        df = pd.read_csv(cache_file)
        print(f'Reusing verified cache for {len(df)} documents.', flush=True)
    else:
        print(f'Extracting spaCy/textstat/MATTR features for {len(samples)} documents...', flush=True)
        df = pd.DataFrame(extract_records(samples.to_dict(orient='records'), model=args.model))
        df = df.drop(columns=['text'])
        df.to_csv(cache_file, index=False)
        cache_meta.write_text(json.dumps(cache_info, indent=2), encoding='utf-8')
    feature_names = list(FEATURE_SCHEMA)
    (out / 'feature_schema.json').write_text(json.dumps({name: {'group': FEATURE_GROUPS[name], 'definition': FEATURE_SCHEMA[name]} for name in feature_names}, indent=2), encoding='utf-8')
    df[feature_names].isna().sum().rename('n_missing').to_csv(out / 'missing_features.csv', index_label='feature')
    diagnostic = set(LENGTH_SENSITIVE_FEATURES)
    wiki = list(WIKIPEDIA_FEATURES)
    style = [name for name in feature_names if name not in diagnostic and name not in wiki]
    feature_sets = {'style': style, 'wikipedia': wiki, 'combined': style + wiki, 'length_only': ['n_words', 'n_characters', 'n_sentences']}
    print('Computing paired effects and cluster bootstrap intervals...', flush=True)
    effects = effect_tables(df, feature_names, seed=args.seed, n_bootstrap=args.bootstrap)
    effects.to_csv(out / 'feature_effects.csv', index=False)
    shortlist, robustness = shortlist_features(df, effects, style + wiki)
    robustness.to_csv(out / 'feature_robustness.csv', index=False)
    (out / 'shortlist.json').write_text(json.dumps({'status': 'exploratory_full_sample_selection_requires_independent_validation', 'features': shortlist, 'rule': {'min_abs_g_each_domain': 0.3, 'min_generator_sign_fraction': 5 / 7, 'max_spearman': 0.85, 'maximum_size': 12}}, indent=2), encoding='utf-8')
    print('Evaluating grouped logistic regression: within-domain, held-domain, held-generator...', flush=True)
    metrics, predictions = evaluate(df, feature_sets, seed=args.seed, n_splits=args.folds, n_bootstrap=args.bootstrap)
    metrics.to_csv(out / 'classifier_metrics.csv', index=False)
    predictions.to_csv(out / 'predictions.csv', index=False)
    metadata = {'utc_started': utc_started, 'seed': args.seed, 'clean_sentence_ends': not args.keep_incomplete, 'pairs_per_domain': args.pairs, 'bootstrap_replicates': args.bootstrap, 'folds': args.folds, 'feature_sets': feature_sets, 'versions': versions, 'sample_sha256': sample_hash, 'extractor_sha256': extractor_hash, 'code_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((ROOT / 'src/humanization/stylometry').glob('*.py'))}, 'elapsed_seconds': round(time.monotonic() - start, 2), 'shortlist_is_exploratory': True}
    (out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    write_report(df, effects, metrics, shortlist, robustness, manifest, out)
    print(metrics[['protocol', 'domain', 'generator', 'feature_set', 'roc_auc', 'macro_f1']].to_string(index=False), flush=True)
    print('Exploratory shortlist:', ', '.join(shortlist) or '(none)', flush=True)
    print('Results:', out, flush=True)

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'raw_dir': ROOT / 'data/raw/mgtbench', 'output': ROOT / 'results/new_runs/02_stylometry/baseline', 'pairs': 300, 'seed': 42, 'bootstrap': 500, 'folds': 5, 'model': 'en_core_web_sm', 'reuse_features': False, 'keep_incomplete': False}
    values.update(overrides)
    return SimpleNamespace(**values)

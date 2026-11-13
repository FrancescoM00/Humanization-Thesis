"""Ripete l'esperimento stilistico completo dopo aver pareggiato esattamente la lunghezza nelle coppie di origine."""
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
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
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
from humanization.datasets.length_matching import length_balance, prepare_length_matched_dataset
from humanization.stylometry.plots import save_style_outputs

def _argument_error(message):
    raise ValueError(message)

def run(args):
    started, utc = (time.monotonic(), datetime.now(timezone.utc).isoformat())
    out = args.output.resolve()
    if out == (ROOT / 'results/published/02_stylometry/baseline').resolve() or out == (ROOT / 'results_pilot').resolve():
        raise ValueError('Use a separate output directory; previous experiments must be preserved.')
    out.mkdir(parents=True, exist_ok=True)
    print('Preparing source-aligned, exactly word-count-matched sentence prefixes...', flush=True)
    samples, manifest = prepare_length_matched_dataset(args.raw_dir, ROOT / 'data/processed/generated/length_matched', n_pairs=args.pairs, seed=args.seed, minimum=args.minimum, maximum=args.maximum, target=args.target, model=args.model)
    (out / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    versions = {name: importlib.metadata.version(name) for name in ['spacy', 'en-core-web-sm', 'textstat', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'nltk']}
    cache_info = {'sample_sha256': manifest['samples_sha256'], 'versions': versions, 'model': args.model, 'extractor_sha256': hashlib.sha256((ROOT / 'src/humanization/stylometry/features.py').read_bytes()).hexdigest(), 'schema_sha256': hashlib.sha256(json.dumps(FEATURE_SCHEMA, sort_keys=True).encode()).hexdigest()}
    cache, metadata = (out / 'features.csv', out / 'features_cache.json')
    if args.reuse_features and cache.exists() and metadata.exists():
        if json.loads(metadata.read_text()) != cache_info:
            raise ValueError('Matched feature cache differs from sample/extractor/model; rerun without --reuse-features.')
        df = pd.read_csv(cache)
        print(f'Reusing verified matched cache for {len(df)} texts.', flush=True)
    else:
        print(f'Extracting all fixed spaCy/textstat/MATTR features for {len(samples)} matched texts...', flush=True)
        df = pd.DataFrame(extract_records(samples.to_dict('records'), model=args.model)).drop(columns='text')
        df.to_csv(cache, index=False)
        metadata.write_text(json.dumps(cache_info, indent=2))
    if not df.n_words.eq(df.matched_n_words).all():
        raise RuntimeError('Preparation/extraction lexical word counts disagree; stop rather than claim length balance.')
    by_pair = df.pivot(index='pair_id', columns='label', values='n_words')
    if not by_pair[0].eq(by_pair[1]).all():
        raise RuntimeError('Exact within-source length matching failed.')
    balance = length_balance(df)
    balance.to_csv(out / 'length_balance.csv', index=False)
    word_balance = balance[balance.measure == 'n_words']
    if not word_balance.standardized_mean_difference.eq(0).all() or not word_balance.ks_statistic.eq(0).all():
        raise RuntimeError('Word-length distributions are not exactly equal in a reported scope.')
    names = list(FEATURE_SCHEMA)
    (out / 'feature_schema.json').write_text(json.dumps({n: {'group': FEATURE_GROUPS[n], 'definition': FEATURE_SCHEMA[n]} for n in names}, indent=2))
    df[names].isna().sum().rename('n_missing').to_csv(out / 'missing_features.csv', index_label='feature')
    wiki = list(WIKIPEDIA_FEATURES)
    style = [n for n in names if n not in LENGTH_SENSITIVE_FEATURES and n not in wiki]
    feature_sets = {'style': style, 'wikipedia': wiki, 'combined': style + wiki, 'word_count_only': ['n_words'], 'length_only': ['n_words', 'n_characters', 'n_sentences']}
    print('Computing paired effects and robustness after length matching...', flush=True)
    effects = effect_tables(df, names, seed=args.seed, n_bootstrap=args.bootstrap)
    effects.to_csv(out / 'feature_effects.csv', index=False)
    shortlist, robustness = shortlist_features(df, effects, style + wiki)
    robustness.to_csv(out / 'feature_robustness.csv', index=False)
    (out / 'shortlist.json').write_text(json.dumps({'status': 'exploratory_full_matched_sample_selection_requires_independent_validation', 'features': shortlist, 'rule': {'min_abs_g_each_domain': 0.3, 'min_generator_sign_fraction': 5 / 7, 'max_spearman': 0.85, 'maximum_size': 12}}, indent=2))
    print('Evaluating grouped logistic models, including pure word-count control...', flush=True)
    metrics, predictions = evaluate(df, feature_sets, seed=args.seed, n_splits=args.folds, n_bootstrap=args.bootstrap)
    metrics.to_csv(out / 'classifier_metrics.csv', index=False)
    predictions.to_csv(out / 'predictions.csv', index=False)
    controls = metrics[metrics.feature_set == 'word_count_only']
    if not controls.roc_auc.sub(0.5).abs().lt(1e-12).all():
        raise RuntimeError('Pure word-count control is not exactly at chance; inspect paired folds and OOF predictions.')
    save_style_outputs(df, effects, metrics, shortlist, out)
    old_metrics_path = ROOT / 'results/published/02_stylometry/baseline/classifier_metrics.csv'
    if old_metrics_path.exists():
        old = pd.read_csv(old_metrics_path)
        keys = ['protocol', 'domain', 'generator', 'feature_set']
        comparison = old[keys + ['roc_auc', 'macro_f1']].merge(metrics[keys + ['roc_auc', 'macro_f1']], on=keys, suffixes=('_original', '_length_matched'))
        comparison['auc_change'] = comparison.roc_auc_length_matched - comparison.roc_auc_original
        comparison.to_csv(out / 'comparison_with_original.csv', index=False)
    old_shortlist = ROOT / 'results/published/02_stylometry/baseline/shortlist.json'
    previous = json.loads(old_shortlist.read_text()).get('features', []) if old_shortlist.exists() else []
    pd.DataFrame([{'feature': f, 'original_selected': f in previous, 'matched_selected': f in shortlist} for f in sorted(set(previous).union(shortlist))]).to_csv(out / 'shortlist_before_after.csv', index=False)
    (out / 'run_metadata.json').write_text(json.dumps({'utc_started': utc, 'seed': args.seed, 'pairs_per_domain': args.pairs, 'prefix_range': [args.minimum, args.maximum], 'prefix_target': args.target, 'exact_word_match': True, 'bootstrap_replicates': args.bootstrap, 'folds': args.folds, 'feature_sets': feature_sets, 'versions': versions, 'sample_sha256': manifest['samples_sha256'], 'elapsed_seconds': round(time.monotonic() - started, 2), 'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT / 'src/humanization/datasets/length_matching.py', ROOT / 'src/humanization/stylometry/features.py', ROOT / 'src/humanization/evaluation/stylometric.py', Path(__file__)]}, 'shortlist_is_exploratory': True}, indent=2))
    print(metrics[['protocol', 'domain', 'generator', 'feature_set', 'roc_auc', 'macro_f1']].to_string(index=False), flush=True)
    print('Matched exploratory shortlist:', shortlist, flush=True)
    print('Results:', out, flush=True)

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'raw_dir': ROOT / 'data/raw/mgtbench', 'output': ROOT / 'results/new_runs/02_stylometry/length_matched', 'pairs': 300, 'seed': 42, 'minimum': 90, 'maximum': 180, 'target': 150, 'bootstrap': 500, 'folds': 5, 'model': 'en_core_web_sm', 'reuse_features': False}
    values.update(overrides)
    return SimpleNamespace(**values)

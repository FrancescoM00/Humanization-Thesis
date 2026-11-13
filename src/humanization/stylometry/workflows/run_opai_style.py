"""Stilometria appaiata fra cinque fasi ufficiali adiacenti di editing OpAI."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from humanization.datasets.opai import _balanced_selection
from humanization.evaluation.stylometric import _bh
from humanization.stylometry.features import FEATURE_GROUPS, FEATURE_SCHEMA, WIKIPEDIA_FEATURES, extract_features, load_nlp
OPERATIONS = [(0, 1, 'polish'), (1, 2, 'paraphrase'), (2, 3, 'style'), (3, 4, 'compress'), (4, 5, 'expand')]
SHORTLIST = ['pos_num', 'pos_cconj', 'mattr_50', 'pos_propn']
LABELS = {'polish': 'Polish', 'paraphrase': 'Paraphrase', 'style': 'Style Rewrite', 'compress': 'Compress', 'expand': 'Expand'}

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def prefixes(doc, minimum=90, maximum=180):
    """Prefissi iniziali invariati, conclusi a un confine di punteggiatura analizzato."""
    found, n_words = ({}, 0)
    for sent in doc.sents:
        n_words += sum((token.is_alpha for token in sent))
        if n_words > maximum:
            break
        if n_words < minimum:
            continue
        prefix = doc.text[:sent.end_char].rstrip()
        if re.search('[.!?…][\\s"\'’”\\)\\]]*$', prefix):
            found.setdefault(n_words, prefix)
    return found

def summarize(frame, feature_names, rng, n_bootstrap, analysis):
    rows = []
    for _, _, operation in OPERATIONS:
        part = frame[frame.operation == operation]
        if not len(part):
            continue
        before = part[[f'before__{f}' for f in feature_names]].copy()
        after = part[[f'after__{f}' for f in feature_names]].copy()
        before.columns = after.columns = feature_names
        complete = np.isfinite(before.to_numpy()) & np.isfinite(after.to_numpy())
        before = before.where(complete).groupby(part.source_id).mean()
        after = after.where(complete).groupby(part.source_id).mean()
        delta = after - before
        groups = len(delta)
        weights = rng.multinomial(groups, np.full(groups, 1 / groups), size=n_bootstrap)
        values = delta.to_numpy()
        valid = np.isfinite(values)
        finite_values = np.where(valid, values, 0)
        draws_n = weights @ valid.astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            draws_mean = weights @ finite_values / draws_n
            draws_var = np.maximum((weights @ finite_values ** 2 - draws_n * draws_mean ** 2) / (draws_n - 1), 0)
            draws_g = (1 - 3 / (4 * draws_n - 5)) * draws_mean / np.sqrt(draws_var)
        for j, feature in enumerate(feature_names):
            d = delta[feature].dropna()
            n = len(d)
            sd = d.std(ddof=1)
            if n < 2:
                g, pvalue = (np.nan, np.nan)
            elif sd == 0:
                g = 0.0 if d.mean() == 0 else np.nan
                pvalue = 1.0 if d.mean() == 0 else 0.0
            else:
                g = (1 - 3 / (4 * n - 5)) * d.mean() / sd
                pvalue = stats.ttest_1samp(d, 0).pvalue
            good_mean = draws_mean[:, j][np.isfinite(draws_mean[:, j])]
            low, high = np.quantile(good_mean, [0.025, 0.975]) if len(good_mean) else (np.nan, np.nan)
            good_g = draws_g[:, j][np.isfinite(draws_g[:, j])]
            if len(d) and d.eq(0).all():
                glow = ghigh = 0.0
            elif len(good_g) >= n_bootstrap / 2:
                glow, ghigh = np.quantile(good_g, [0.025, 0.975])
            else:
                glow = ghigh = np.nan
            before_valid, after_valid = (before.loc[d.index, feature], after.loc[d.index, feature])
            rows.append({'analysis': analysis, 'operation': operation, 'feature': feature, 'family': FEATURE_GROUPS[feature], 'n_pairs': int(part[[f'before__{feature}', f'after__{feature}']].notna().all(axis=1).sum()), 'n_sources': n, 'before_mean': before_valid.mean(), 'after_mean': after_valid.mean(), 'mean_delta': d.mean(), 'median_source_delta': d.median(), 'delta_sd_between_sources': sd, 'delta_ci_low': low, 'delta_ci_high': high, 'paired_hedges_gz': g, 'paired_gz_ci_low': glow, 'paired_gz_ci_high': ghigh, 'paired_t_pvalue': pvalue, 'fraction_source_delta_positive': (d > 0).mean() if n else np.nan, 'fraction_source_delta_zero': (d == 0).mean() if n else np.nan})
    result = pd.DataFrame(rows)
    if len(result):
        result['q_bh_all_features_operations'] = _bh(result.paired_t_pvalue.to_numpy())
    return result

def paired_frame(pairs, features, analysis):
    rows = []
    for pair in pairs:
        a, b = (features[pair['before_hash']], features[pair['after_hash']])
        item = {k: v for k, v in pair.items() if k not in {'before_text', 'after_text'}}
        item['analysis'] = analysis
        for feature in FEATURE_SCHEMA:
            item[f'before__{feature}'] = a[feature]
            item[f'after__{feature}'] = b[feature]
            item[f'delta__{feature}'] = b[feature] - a[feature]
        rows.append(item)
    return pd.DataFrame(rows)

def heatmap(table, names, title, output):
    pivot = table.pivot(index='feature', columns='operation', values='paired_hedges_gz')
    pivot = pivot.reindex(index=names, columns=[x[2] for x in OPERATIONS])
    fig, ax = plt.subplots(figsize=(7.5, max(3.3, len(names) * 0.37)))
    im = ax.imshow(pivot, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    ax.set_xticks(range(5), [LABELS[x] for x in pivot.columns], rotation=15, ha='right')
    ax.set_yticks(range(len(names)), names, fontsize=9)
    for i in range(len(names)):
        for j in range(5):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f'{val:+.2f}', ha='center', va='center', fontsize=9, color='white' if abs(val) > 1.1 else 'black')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Hedges g_z appaiato, dopo − prima\n(colore limitato a ±2)')
    fig.tight_layout()
    fig.savefig(str(output) + '.png', dpi=180)
    plt.close(fig)

def _argument_error(message):
    raise ValueError(message)

def run(args):
    start = time.monotonic()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    from humanization.semantics.positive_controls import load_local_opai_trajectories
    trajectories, provenance = load_local_opai_trajectories(args.opai_dir, max_version=5)
    candidates = [{'key': key, 'domain': key[1], 'generator': key[2]} for key, versions in sorted(trajectories.items()) if set(range(6)).issubset(versions)]
    selected = _balanced_selection(candidates, args.n, random.Random(args.seed))
    if len(selected) != args.n:
        raise ValueError(f'Only {len(selected)} complete trajectories; requested {args.n}')
    texts, pairs = ({}, [])
    for candidate in selected:
        key = candidate['key']
        versions = trajectories[key]
        for before, after, operation in OPERATIONS:
            if versions[after]['operation'] not in {operation, 'style rewrite' if operation == 'style' else operation}:
                raise ValueError(f"Unexpected version operation: {versions[after]['operation']}")
            a, b = [versions[v]['text'] for v in [before, after]]
            ha, hb = (digest(a), digest(b))
            texts[ha], texts[hb] = (a, b)
            pairs.append({'pair_id': digest('|'.join([*key, str(before), str(after)]))[:24], 'source_id': key[0], 'domain': key[1], 'generator': key[2], 'operation': operation, 'from_version': before, 'to_version': after, 'before_hash': ha, 'after_hash': hb, 'text_changed': a != b})
    selected_records = [{'source_id': x['key'][0], 'domain': x['domain'], 'generator': x['generator']} for x in selected]
    manifest = {'official_source': provenance, 'seed': args.seed, 'n_selected_trajectories': len(selected), 'n_unique_sources': len({x['source_id'] for x in selected_records}), 'candidate_complete_trajectories': len(candidates), 'n_pairs': len(pairs), 'n_unique_full_texts': len(texts), 'selected_trajectories': selected_records, 'stratum_counts': dict(Counter((x['domain'] + '|' + x['generator'] for x in selected))), 'operations': OPERATIONS, 'selection': 'seeded round-robin domain × editor, same trajectories in all stages', 'source_unit': 'document_hash_id; editors averaged before inference', 'full_document_primary': True, 'prefix_control': 'same source and stage, exact lexical counts 90–180, closest to 150, parsed sentence ends', 'versions': {k: importlib.metadata.version(k) for k in ['spacy', 'en-core-web-sm', 'textstat', 'numpy', 'pandas', 'scipy']}, 'extractor_sha256': hashlib.sha256((ROOT / 'src/humanization/stylometry/features.py').read_bytes()).hexdigest(), 'feature_schema': FEATURE_SCHEMA}
    (out / 'data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.data_dir / 'pairs.jsonl').write_text(''.join((json.dumps({**x, 'original': texts[x['before_hash']], 'modified': texts[x['after_hash']]}, ensure_ascii=False) + '\n' for x in pairs)))
    cache_path = args.data_dir / 'feature_cache.json'
    cache = json.loads(cache_path.read_text()) if args.reuse_features and cache_path.is_file() else {}
    nlp = load_nlp()
    prefix_maps = {}
    missing = [(h, t) for h, t in texts.items() if h not in cache]
    print(f'Full documents: {len(texts)} unique; extracting {len(missing)}', flush=True)
    for i, ((h, text), doc) in enumerate(zip(missing, nlp.pipe((t for _, t in missing), batch_size=32))):
        cache[h] = extract_features(doc, text)
        prefix_maps[h] = prefixes(doc)
        if (i + 1) % 100 == 0:
            print(f'Extracted {i + 1}/{len(missing)}', flush=True)
            cache_path.write_text(json.dumps(cache, allow_nan=True))
    prefix_missing = [(h, t) for h, t in texts.items() if h not in prefix_maps]
    for (h, _), doc in zip(prefix_missing, nlp.pipe((t for _, t in prefix_missing), batch_size=32)):
        prefix_maps[h] = prefixes(doc)
    prefix_pairs, prefix_texts = ([], {})
    for pair in pairs:
        a, b = [prefix_maps[pair[k]] for k in ['before_hash', 'after_hash']]
        shared = set(a).intersection(b)
        if not shared:
            continue
        n = min(shared, key=lambda x: (abs(x - 150), x))
        ha, hb = (digest(a[n]), digest(b[n]))
        prefix_texts[ha], prefix_texts[hb] = (a[n], b[n])
        prefix_pairs.append({**pair, 'before_hash': ha, 'after_hash': hb, 'matched_n_words': n, 'text_changed': a[n] != b[n]})
    missing = [(h, t) for h, t in prefix_texts.items() if h not in cache]
    print(f'Prefix control: {len(prefix_pairs)}/{len(pairs)} pairs; {len(missing)} new unique passages', flush=True)
    for i, ((h, text), doc) in enumerate(zip(missing, nlp.pipe((t for _, t in missing), batch_size=32))):
        cache[h] = extract_features(doc, text)
        if (i + 1) % 100 == 0:
            print(f'Extracted prefixes {i + 1}/{len(missing)}', flush=True)
            cache_path.write_text(json.dumps(cache, allow_nan=True))
    cache_path.write_text(json.dumps(cache, allow_nan=True))
    features_frame = pd.DataFrame([{'text_sha256': h, **cache[h]} for h in texts])
    features_frame.to_csv(out / 'document_features.csv', index=False)
    full = paired_frame(pairs, cache, 'full_document')
    control = paired_frame(prefix_pairs, cache, 'exact_prefix_length_matched')
    if len(control):
        assert (control['before__n_words'] == control['after__n_words']).all()
        assert (control['before__n_words'] == control.matched_n_words).all()
    full.to_csv(out / 'paired_feature_values.csv', index=False)
    control.to_csv(out / 'prefix_matched_feature_values.csv', index=False)
    all_tables = []
    rng = np.random.default_rng(args.seed)
    for name, frame in [('full_document', full), ('exact_prefix_length_matched', control)]:
        if len(frame):
            table = summarize(frame, list(FEATURE_SCHEMA), rng, args.bootstrap, name)
            all_tables.append(table)
    summary = pd.concat(all_tables, ignore_index=True)
    summary.to_csv(out / 'summary.csv', index=False)
    strata = []
    for scope, column in [('domain', 'domain'), ('editor', 'generator')]:
        for (operation, name), part in full.groupby(['operation', column]):
            values = part.groupby('source_id')[[f'delta__{f}' for f in FEATURE_SCHEMA]].mean().mean()
            for feature, value in zip(FEATURE_SCHEMA, values):
                strata.append({'scope': scope, 'stratum': name, 'operation': operation, 'feature': feature, 'n_pairs': len(part), 'n_sources': part.source_id.nunique(), 'mean_delta': value})
    pd.DataFrame(strata).to_csv(out / 'stratified_mean_deltas.csv', index=False)
    full_table = summary[summary.analysis == 'full_document']
    prefix_table = summary[summary.analysis == 'exact_prefix_length_matched']
    chosen = SHORTLIST + ['sentence_length_mean', 'sentence_length_cv', 'word_length_mean'] + list(WIKIPEDIA_FEATURES)
    heatmap(full_table, chosen, 'OpAI: variazione stilometrica nelle cinque operazioni', out / 'style_movement')
    if len(prefix_table):
        heatmap(prefix_table, chosen, 'Controllo: prefissi appaiati con uguale numero di parole', out / 'style_movement_length_matched')
    length_rows = []
    for _, _, operation in OPERATIONS:
        part = full[full.operation == operation]
        pref = control[control.operation == operation]
        length_rows.append({'operation': LABELS[operation], 'pairs': len(part), 'unique_sources': part.source_id.nunique(), 'changed': int(part.text_changed.sum()), 'before_mean_words': part.groupby('source_id')['before__n_words'].mean().mean(), 'after_mean_words': part.groupby('source_id')['after__n_words'].mean().mean(), 'median_word_ratio': (part['after__n_words'] / part['before__n_words']).median(), 'prefix_matched_pairs': len(pref), 'prefix_sources': pref.source_id.nunique()})
    length = pd.DataFrame(length_rows)
    length.to_csv(out / 'length_changes.csv', index=False)
    manifest['elapsed_seconds'] = time.monotonic() - start
    manifest['prefix_matched_pairs'] = len(control)
    manifest['n_summary_rows'] = len(summary)
    (out / 'data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'elapsed_seconds': manifest['elapsed_seconds'], 'pairs': len(full), 'prefix_pairs': len(control), 'unique_full_texts': len(texts), 'summary_rows': len(summary)}, indent=2), flush=True)

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'opai_dir': ROOT / 'data/raw/opai', 'output': ROOT / 'results/new_runs/04_opai_style/document', 'data_dir': ROOT / 'data/processed/generated/opai_editing', 'n': 200, 'seed': 42, 'bootstrap': 1000, 'reuse_features': False}
    values.update(overrides)
    return SimpleNamespace(**values)

"""Distanze e corrispondenze originali delle frasi per la stilometria locale OpAI.

La versione Parquet pubblica omette gli sentence_ids espliciti ma conserva
l'array ordinato delle frasi del costruttore. Si usa tale corrispondenza solo
dopo aver verificato cardinalità costante, ricostruzione esatta del testo e
tutti i riferimenti v0 invariati.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
from scipy.spatial.distance import jensenshannon

from humanization.stylometry.features import FEATURE_GROUPS, FEATURE_SCHEMA, POS_TAGS

SHORTLIST = ['pos_num', 'pos_cconj', 'mattr_50', 'pos_propn']
EXCLUDED_DISTANCE = {
    'n_words', 'n_characters', 'n_sentences', 'ttr', 'hapax_ratio',
    'mattr_50', 'sentence_length_std', 'sentence_length_cv',
}
DISTANCE_FEATURE_CANDIDATES = [name for name in FEATURE_SCHEMA
                               if name not in EXCLUDED_DISTANCE
                               and FEATURE_GROUPS[name] != 'wikipedia']


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalize_whitespace(text):
    return ' '.join(text.split())


def validate_native_lineage(versions):
    """Verifica la corrispondenza posizionale pubblicata senza abbinamenti approssimativi."""
    if set(range(6)) - set(versions):
        raise ValueError('All adjacent versions v0..v5 are required')
    counts = {len(row['sentences']) for row in versions.values()}
    if len(counts) != 1:
        raise ValueError('Sentence count changes: positional lineage not certified')
    baseline = versions[0]['sentences']
    for version, row in versions.items():
        sentences, labels = row['sentences'], row['sentence_labels']
        if len(sentences) != len(labels) or not set(labels).issubset({0, 1}):
            raise ValueError('Sentence labels/array schema mismatch')
        if normalize_whitespace(' '.join(sentences)) != normalize_whitespace(row['text']):
            raise ValueError('Released sentences do not reconstruct the document')
        if any(label == 0 and text != baseline[i]
               for i, (text, label) in enumerate(zip(sentences, labels))):
            raise ValueError('Unedited position differs from its v0 anchor')
        if version:
            old = versions[version-1]
            if any(a != b and label == 0 for a, b, label in
                   zip(old['sentences'], sentences, labels)):
                raise ValueError('Text changes outside released targeted positions')
            if any(a > b for a, b in zip(old['sentence_labels'], labels)):
                raise ValueError('Cumulative sentence labels are not monotone')
    return [f's{i+1:04d}' for i in range(len(baseline))]


def distance_phi(before, after, names, scale):
    delta = np.asarray([after[name] - before[name] for name in names], dtype=float)
    if not np.isfinite(delta).all():
        return np.nan
    return float(np.sqrt(np.mean((delta / scale)**2)))


def pos_js(before, after):
    a = np.asarray([before['pos_'+tag.lower()] for tag in POS_TAGS], dtype=float)
    b = np.asarray([after['pos_'+tag.lower()] for tag in POS_TAGS], dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or a.sum() <= 0 or b.sum() <= 0:
        return np.nan
    return float(jensenshannon(a/a.sum(), b/b.sum(), base=2))


def source_bootstrap(values, rng, n_bootstrap=1000):
    """Campione scalare già mediato all'interno di ogni documento di origine."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    draws = rng.choice(values, size=(n_bootstrap,len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws,[.025,.975]))

"""Sentence/chunk similarity with the same worst-case aggregation as NLI.

The alignment units are the existing MiniLM-bounded sentence chunks used by
the NLI stage. For each hypothesis unit, retain its maximum similarity over
*all* premise units. Both directions are evaluated; mean/q10/min are unweighted
over hypothesis units, matching NLI's aggregation convention.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from humanization.semantics.metrics import (_cached_chunks, _cpu_threads, _hash, _identity, _model_path,
                      _records, _release_device_cache, _resolve_device)


BERT_F1_BASELINE = .814463  # Official English RoBERTa-base, layer 10.
AGGREGATORS = ('mean', 'q10', 'min')


def summarize_support(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite support for every hypothesis unit')
    return {'mean': float(values.mean()), 'q10': float(np.quantile(values, .1)),
            'min': float(values.min())}


def directional_row(pair, forward, reverse, prefix):
    out = _identity(pair)
    summaries = {direction: summarize_support(values)
                 for direction, values in [('forward', forward), ('reverse', reverse)]}
    for direction, values in summaries.items():
        out.update({f'{prefix}_{direction}_{key}': value for key, value in values.items()})
    out.update({f'{prefix}_bidirectional_{key}': min(summaries['forward'][key], summaries['reverse'][key])
                for key in AGGREGATORS})
    return out


def matrix_support(matrix):
    """Rows are original/premise units; columns are modified/hypothesis units."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError('Expected nonempty finite original-by-modified matrix')
    return values.max(axis=0), values.max(axis=1)


def candidate_support(matrix, forward_candidates, reverse_candidates):
    """Restrict maxima to exactly the same retrieved evidence as NLI."""
    values = np.asarray(matrix, dtype=float)
    if len(forward_candidates) != values.shape[1] or len(reverse_candidates) != values.shape[0]:
        raise ValueError('Candidate lists must cover every hypothesis unit')
    forward = np.array([values[candidates, j].max() for j, candidates in enumerate(forward_candidates)])
    reverse = np.array([values[i, candidates].max() for i, candidates in enumerate(reverse_candidates)])
    return forward, reverse


def unit_record(pair, item, forward, reverse, metric):
    return {'id': pair['id'], 'metric': metric,
            'original_sha256': _hash(pair['original']), 'modified_sha256': _hash(pair['modified']),
            'forward_support': [float(x) for x in forward],
            'reverse_support': [float(x) for x in reverse],
            'original_units': len(item['original_chunks']), 'modified_units': len(item['modified_chunks']),
            'unit_protocol': 'shared_NLI_alignment_sentence_chunks_MiniLM_budget_160',
            'alignment': 'maximum_similarity_over_all_premise_units',
            'aggregation': 'unweighted_hypothesis_units'}


def aligned_cosine_scores(pairs, alignment, progress_callback=None):
    """Reuse top-k cosine cache: it contains the exact global maximum (k>=1)."""
    rows, units = [], []
    records = list(_records(pairs))
    for index, pair in enumerate(records, 1):
        item = _cached_chunks(pair, alignment)
        if item is None:
            raise ValueError(f'Missing NLI alignment for {pair["id"]}')
        forward = [max(values) for values in item['forward_similarities']]
        reverse = [max(values) for values in item['reverse_similarities']]
        if len(forward) != len(item['modified_chunks']) or len(reverse) != len(item['original_chunks']):
            raise ValueError('Cosine alignment does not cover every hypothesis unit')
        row = directional_row(pair, forward, reverse, 'aligned_cosine')
        row.update(aligned_original_units=len(reverse), aligned_modified_units=len(forward))
        rows.append(row)
        units.append(unit_record(pair, item, forward, reverse, 'cosine'))
        if progress_callback and (index % 25 == 0 or index == len(records)):
            progress_callback(index, len(records))
    return pd.DataFrame(rows), units


def aligned_bert_scores(pairs, alignment, modelpath, *, num_layers=10, batch_size=4,
                        n_threads=4, token_cache_limit_mib=512, progress_callback=None, device='cpu'):
    """Max BERTScore F1 per shared sentence/chunk, with no document averaging.

    Unlike the old BERTScore document approximation, F1 is computed for every
    unit pair and maxima are aggregated directly. The shared units are not
    resegmented using another tokenizer: any unit exceeding RoBERTa's 512-token
    limit raises an error rather than silently truncating or changing units.
    """
    from bert_score import BERTScorer
    from bert_score.utils import get_bert_embedding
    import torch

    _cpu_threads(n_threads)
    device = _resolve_device(device)
    scorer = BERTScorer(model_type=_model_path(modelpath), num_layers=num_layers,
                        batch_size=batch_size, nthreads=min(n_threads, 4), idf=False,
                        device=device, lang='en', rescale_with_baseline=False,
                        use_fast_tokenizer=True)
    if device.startswith('cuda'):
        scorer._model.float()
    tokenizer = scorer._tokenizer
    token_limit = min(int(tokenizer.model_max_length), 512)
    idf = defaultdict(lambda: 1.0)
    idf[tokenizer.sep_token_id] = idf[tokenizer.cls_token_id] = 0.0
    vectors = OrderedDict()
    vector_bytes = 0
    vector_limit = int(token_cache_limit_mib * 1024 * 1024)
    scores = {}
    hashes = {}
    max_input_tokens = 0

    def key_for(text):
        if text not in hashes:
            hashes[text] = _hash(text)
        return hashes[text]

    def pair_key(a, b):
        # Raw F1 is symmetric at fixed model/IDF. Reuse transposed pairs.
        return tuple(sorted((key_for(a), key_for(b))))

    def ensure_embeddings(texts):
        nonlocal vector_bytes, max_input_tokens
        for text in texts:
            if text in vectors:
                vectors.move_to_end(text)
        missing = sorted(set(t for t in texts if t not in vectors), key=len, reverse=True)
        for text in missing:
            length = len(tokenizer.encode(text, add_special_tokens=True))
            max_input_tokens = max(max_input_tokens, length)
            if length > token_limit:
                raise ValueError('Shared alignment unit exceeds RoBERTa limit; refusing to truncate/resegment')
        for offset in range(0, len(missing), batch_size):
            batch = missing[offset:offset+batch_size]
            embeddings, masks, weights = get_bert_embedding(batch, scorer._model, tokenizer, idf,
                                                            device=device, all_layers=False)
            for i, text in enumerate(batch):
                length = int(masks[i].sum())
                vec = embeddings[i, :length].clone()
                vec = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                # BERTScore returns IDF weights on CPU even for CUDA embeddings.
                weight = weights[i, :length].to(device=vec.device).clone()
                weight = weight / weight.sum().clamp_min(1e-12)
                vectors[text] = (vec, weight)
                vector_bytes += vec.numel()*vec.element_size() + weight.numel()*weight.element_size()

    def score_matrix(a, b):
        wanted = {}
        for original in a:
            for modified in b:
                key = pair_key(original, modified)
                if key not in scores:
                    wanted.setdefault(key, (original, modified))
        ensure_embeddings([text for pair in wanted.values() for text in pair])
        with torch.inference_mode():
            for key, (original, modified) in wanted.items():
                ref, rw = vectors[original]
                hyp, hw = vectors[modified]
                similarity = hyp @ ref.T
                precision = float((similarity.max(dim=1).values * hw).sum())
                recall = float((similarity.max(dim=0).values * rw).sum())
                scores[key] = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        return np.array([[scores[pair_key(aa, bb)] for bb in b] for aa in a])

    rows, units = [], []
    records = list(_records(pairs))
    try:
        for index, pair in enumerate(records, 1):
            item = _cached_chunks(pair, alignment)
            if item is None:
                raise ValueError(f'Missing NLI alignment for {pair["id"]}')
            a, b = item['original_chunks'], item['modified_chunks']
            matrix = score_matrix(a, b)
            forward, reverse = matrix_support(matrix)
            restricted_forward, restricted_reverse = candidate_support(
                matrix, item['forward_candidates'], item['reverse_candidates'])
            row = directional_row(pair, forward, reverse, 'aligned_bert_f1')
            scaled_forward = (forward-BERT_F1_BASELINE)/(1-BERT_F1_BASELINE)
            scaled_reverse = (reverse-BERT_F1_BASELINE)/(1-BERT_F1_BASELINE)
            scaled = directional_row(pair, scaled_forward, scaled_reverse, 'aligned_bert_f1_rescaled')
            row.update({k: v for k, v in scaled.items() if k not in {'id', 'pair_id'}})
            restricted = directional_row(pair, restricted_forward, restricted_reverse, 'aligned_bert_top3_f1')
            row.update({k: v for k, v in restricted.items() if k not in {'id', 'pair_id'}})
            restricted_scaled = directional_row(
                pair, (restricted_forward-BERT_F1_BASELINE)/(1-BERT_F1_BASELINE),
                (restricted_reverse-BERT_F1_BASELINE)/(1-BERT_F1_BASELINE), 'aligned_bert_top3_f1_rescaled')
            row.update({k: v for k, v in restricted_scaled.items() if k not in {'id', 'pair_id'}})
            row.update(aligned_original_units=len(a), aligned_modified_units=len(b),
                       aligned_bert_num_layers=num_layers,
                       aligned_bert_baseline_f1=BERT_F1_BASELINE,
                       aligned_bert_max_input_tokens_cumulative=max_input_tokens)
            rows.append(row)
            record = unit_record(pair, item, forward, reverse, 'bertscore_f1_raw')
            record.update(top3_forward_support=[float(x) for x in restricted_forward],
                          top3_reverse_support=[float(x) for x in restricted_reverse])
            units.append(record)
            while vector_bytes > vector_limit and vectors:
                _, (vec, weight) = vectors.popitem(last=False)
                vector_bytes -= vec.numel()*vec.element_size() + weight.numel()*weight.element_size()
            if progress_callback and (index % 25 == 0 or index == len(records)):
                progress_callback(index, len(records))
    finally:
        del scorer
        gc.collect()
        _release_device_cache(device)
    return pd.DataFrame(rows), units

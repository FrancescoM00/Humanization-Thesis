"""Metriche lessicali, embedding ed entailment bidirezionale senza troncamento.

Le coppie contengono ``id``, ``original``, ``modified`` e, nelle prove locali,
``target_original``/``target_modified``. 

"""
from __future__ import annotations
import gc
import hashlib
import re
import json
from collections import defaultdict, OrderedDict
from collections.abc import Mapping
from pathlib import Path
import numpy as np
import pandas as pd
_SENTENCIZER = None

def _records(pairs):
    records = pairs.to_dict('records') if isinstance(pairs, pd.DataFrame) else list(pairs)
    ids = set()
    for row in records:
        if not isinstance(row.get('original'), str) or not row['original'].strip():
            raise ValueError('Each pair needs nonempty original text')
        if not isinstance(row.get('modified'), str) or not row['modified'].strip():
            raise ValueError('Each pair needs nonempty modified text')
        row_id = str(row.get('id', row.get('pair_id', '')))
        if not row_id or row_id in ids:
            raise ValueError('Pairs require unique, nonempty id values')
        ids.add(row_id)
        row = dict(row, id=row_id)
        yield row

def _local(row):
    a, b = (row.get('target_original'), row.get('target_modified'))
    return (a, b) if isinstance(a, str) and a.strip() and isinstance(b, str) and b.strip() else None

def _identity(row):
    out = {'id': row['id']}
    if 'pair_id' in row:
        out['pair_id'] = row['pair_id']
    return out

def _model_path(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f'Model must already exist in a local directory: {path}')
    return str(path.resolve())

def _cpu_threads(n_threads):
    import torch
    torch.set_num_threads(max(1, min(int(n_threads), 4)))

def _resolve_device(device):
    """Richiede un dispositivo CPU/CUDA esplicito; non ripiega mai silenziosamente su un altro."""
    import torch
    resolved = torch.device(device)
    if resolved.type not in {'cpu', 'cuda'}:
        raise ValueError('Semantic metrics support cpu or cuda devices')
    if resolved.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable. Enable a GPU accelerator or select --device cpu.')
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise RuntimeError(f'Requested CUDA device {resolved.index} is unavailable')
    return str(resolved)

def _release_device_cache(device):
    if device.startswith('cuda'):
        import torch
        torch.cuda.empty_cache()

def _hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def _sentences(text):
    global _SENTENCIZER
    if _SENTENCIZER is None:
        import spacy
        _SENTENCIZER = spacy.blank('en')
        _SENTENCIZER.add_pipe('sentencizer')
    return [sent.text.strip() for sent in _SENTENCIZER(text).sents if sent.text.strip()]

def token_chunks(text, tokenizer, max_tokens=160):
    """Suddivide il testo originale senza decodificare sottoparole o byte incompleti.

    Conserva tutti i caratteri non blank, anche quando occorre dividere
    parole molto lunghe.
    """
    if max_tokens < 1:
        raise ValueError('max_tokens must be positive')
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [text] if text.strip() else []
    chunks, start = ([], 0)
    while start < len(text):
        remainder = text[start:]
        if not remainder.strip():
            break
        remaining_ids = tokenizer.encode(remainder, add_special_tokens=False)
        if len(remaining_ids) <= max_tokens:
            chunks.append(remainder.strip())
            break
        end = 0
        if getattr(tokenizer, 'is_fast', False):
            encoded = tokenizer(remainder, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
            offsets = encoded['offset_mapping']
            hint = int(offsets[min(max_tokens, len(offsets)) - 1][1])
            whitespace = [m.end() for m in re.finditer('\\s+', remainder[:hint])]
            token_ends = [int(offset[1]) for offset in offsets[:max_tokens] if int(offset[1]) > 0]
            for candidate in dict.fromkeys(whitespace[::-1] + token_ends[::-1]):
                piece = remainder[:candidate].strip()
                if piece and len(tokenizer.encode(piece, add_special_tokens=False)) <= max_tokens:
                    end = candidate
                    break
        else:
            for match in re.finditer('\\S+\\s*', remainder):
                candidate = match.end()
                if len(tokenizer.encode(remainder[:candidate].strip(), add_special_tokens=False)) > max_tokens:
                    break
                end = candidate
        if end == 0:
            word = re.search('\\S+', remainder)
            if word is None:
                break
            for candidate in range(word.start() + 1, word.end() + 1):
                piece = remainder[:candidate].strip()
                if piece and len(tokenizer.encode(piece, add_special_tokens=False)) <= max_tokens:
                    end = candidate
                elif end:
                    break
        if end == 0:
            raise ValueError('Cannot preserve a nonempty text chunk within the requested token budget')
        chunks.append(remainder[:end].strip())
        start += end
    return chunks

def sentence_chunks(text, tokenizer, max_tokens=160):
    return [part for sentence in _sentences(text) for part in token_chunks(sentence, tokenizer, max_tokens)]

def _weights(chunks, tokenizer):
    return np.array([max(1, len(tokenizer.encode(t, add_special_tokens=False))) for t in chunks], dtype=float)

def lexical_scores(pairs):
    """ROUGE-L F1 sull'intero documento  e BLEU di frase diviso per 100.

    BLEU usa il tokenizer 13a di SacreBLEU, smoothing esponenziale, ordine
    effettivo e il testo originale come unico riferimento. 
    """
    from rouge_score.rouge_scorer import RougeScorer
    from sacrebleu import sentence_bleu
    scorer = RougeScorer(['rougeL'], use_stemmer=True)

    def score(a, b):
        return {'rouge_l_f1': float(scorer.score(a, b)['rougeL'].fmeasure), 'bleu': float(sentence_bleu(b, [a], tokenize='13a', smooth_method='exp', use_effective_order=True).score / 100)}
    rows = []
    for pair in _records(pairs):
        row = _identity(pair)
        row.update({'lexical_' + k: v for k, v in score(pair['original'], pair['modified']).items()})
        local = _local(pair)
        row.update({'local_lexical_' + k: v for k, v in score(*local).items()} if local else {'local_lexical_rouge_l_f1': np.nan, 'local_lexical_bleu': np.nan})
        rows.append(row)
    return pd.DataFrame(rows)

def _unit_mean(embeddings, weights):
    vector = np.average(embeddings, axis=0, weights=weights)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector

def _top_candidates(similarities, top_k):
    forward = [np.argsort(-similarities[:, i], kind='stable')[:top_k].tolist() for i in range(similarities.shape[1])]
    reverse = [np.argsort(-similarities[i], kind='stable')[:top_k].tolist() for i in range(similarities.shape[0])]
    return (forward, reverse)

def embedding_scores(pairs, modelpath, *, chunk_tokens=160, top_k=3, batch_size=16, n_threads=4, progress_callback=None, device='cpu'):
    """
    recupera i top-k segmenti originali per ogni segmento
    modificato e viceversa. 
    """
    from sentence_transformers import SentenceTransformer
    _cpu_threads(n_threads)
    device = _resolve_device(device)
    model = SentenceTransformer(_model_path(modelpath), device=device, local_files_only=True)
    if device.startswith('cuda'):
        model.float()
    tokenizer = model.tokenizer
    special = tokenizer.num_special_tokens_to_add(pair=False)
    budget = min(chunk_tokens, int(model.max_seq_length) - special)
    if budget < 1 or top_k < 1:
        raise ValueError('Invalid chunk budget or top_k')
    rows, cache = ([], {})
    embedding_cache = {}

    def encode_cached(texts):
        missing = list(dict.fromkeys((text for text in texts if text not in embedding_cache)))
        if missing:
            values = model.encode(missing, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            embedding_cache.update(zip(missing, values))
        return np.array([embedding_cache[text] for text in texts])
    records = list(_records(pairs))
    try:
        for index, pair in enumerate(records, 1):
            original = sentence_chunks(pair['original'], tokenizer, budget)
            modified = sentence_chunks(pair['modified'], tokenizer, budget)
            texts = original + modified
            values = encode_cached(texts)
            a, b = (values[:len(original)], values[len(original):])
            similarities = a @ b.T
            forward, reverse = _top_candidates(similarities, top_k)
            wa, wb = (_weights(original, tokenizer), _weights(modified, tokenizer))
            row = _identity(pair)
            row.update(embedding_cosine=float(_unit_mean(a, wa) @ _unit_mean(b, wb)), embedding_original_chunks=len(original), embedding_modified_chunks=len(modified), embedding_chunk_token_budget=budget, local_embedding_cosine=np.nan)
            local = _local(pair)
            if local:
                ca, cb = [sentence_chunks(t, tokenizer, budget) for t in local]
                vals = encode_cached(ca + cb)
                row['local_embedding_cosine'] = float(_unit_mean(vals[:len(ca)], _weights(ca, tokenizer)) @ _unit_mean(vals[len(ca):], _weights(cb, tokenizer)))
            cache[pair['id']] = {'original_sha256': _hash(pair['original']), 'modified_sha256': _hash(pair['modified']), 'original_chunks': original, 'modified_chunks': modified, 'forward_candidates': forward, 'reverse_candidates': reverse, 'forward_similarities': [[float(similarities[j, i]) for j in cand] for i, cand in enumerate(forward)], 'reverse_similarities': [[float(similarities[i, j]) for j in cand] for i, cand in enumerate(reverse)], 'chunk_token_budget': budget, 'retrieval_top_k': top_k}
            row['embedding_unique_chunks_encoded_cumulative'] = len(embedding_cache)
            rows.append(row)
            if progress_callback and (index % 25 == 0 or index == len(records)):
                progress_callback(index, len(records))
    finally:
        del model
        gc.collect()
        _release_device_cache(device)
    return (pd.DataFrame(rows), cache)

def _cached_chunks(pair, cache):
    if cache is None or pair['id'] not in cache:
        return None
    item = cache[pair['id']]
    if item.get('original_sha256') != _hash(pair['original']) or item.get('modified_sha256') != _hash(pair['modified']):
        raise ValueError(f"Stale alignment cache for {pair['id']}")
    return item

def bert_scores(pairs, modelpath, *, num_layers=10, chunk_tokens=160, batch_size=8, n_threads=4, progress_callback=None, device='cpu'):
    """Approssimazione BERTScore con abbinamento dei segmenti.
    """
    from bert_score import BERTScorer
    from bert_score.utils import get_bert_embedding
    import torch
    _cpu_threads(n_threads)
    device = _resolve_device(device)
    scorer = BERTScorer(model_type=_model_path(modelpath), num_layers=num_layers, batch_size=batch_size, nthreads=min(n_threads, 4), idf=False, device=device, lang='en', rescale_with_baseline=False, use_fast_tokenizer=True)
    if device.startswith('cuda'):
        scorer._model.float()
    tokenizer = scorer._tokenizer
    limit = getattr(tokenizer, 'model_max_length', 512)
    limit = min(int(limit), 512)
    budget = min(chunk_tokens, limit - tokenizer.num_special_tokens_to_add(pair=False))
    result_cache = {}
    token_embedding_cache = OrderedDict()
    token_embedding_bytes = 0
    seen_encoded_chunks = set()
    token_cache_limit_bytes = 512 * 1024 * 1024
    idf = defaultdict(lambda: 1.0)
    idf[tokenizer.sep_token_id] = 0.0
    idf[tokenizer.cls_token_id] = 0.0

    def ensure_token_embeddings(texts):
        nonlocal token_embedding_bytes
        for text in texts:
            if text in token_embedding_cache:
                token_embedding_cache.move_to_end(text)
        missing = sorted(set((text for text in texts if text not in token_embedding_cache)), key=len, reverse=True)
        if any((len(tokenizer.encode(text, add_special_tokens=True)) > limit for text in missing)):
            raise ValueError('BERTScore chunk exceeds tokenizer limit; refusing to truncate')
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            vectors, masks, weights = get_bert_embedding(batch, scorer._model, tokenizer, idf, device=device, all_layers=False)
            for i, text in enumerate(batch):
                length = int(masks[i].sum())
                vector = vectors[i, :length].clone()
                vector = vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                weight = weights[i, :length].to(device=vector.device).clone()
                weight = weight / weight.sum().clamp_min(1e-12)
                token_embedding_cache[text] = (vector, weight)
                token_embedding_bytes += vector.numel() * vector.element_size() + weight.numel() * weight.element_size()
                seen_encoded_chunks.add(text)

    def score_chunks(a, b):
        ensure_token_embeddings(a + b)
        keys = [(aa, bb) for aa in a for bb in b]
        missing = list(dict.fromkeys((key for key in keys if key not in result_cache)))
        with torch.inference_mode():
            for key in missing:
                ref, ref_weight = token_embedding_cache[key[0]]
                hyp, hyp_weight = token_embedding_cache[key[1]]
                similarity = hyp @ ref.T
                pp = float((similarity.max(dim=1).values * hyp_weight).sum())
                rr = float((similarity.max(dim=0).values * ref_weight).sum())
                ff = 2 * pp * rr / (pp + rr) if pp + rr else 0.0
                result_cache[key] = (pp, rr, ff)
        matrices = np.array([result_cache[key] for key in keys]).reshape(len(a), len(b), 3)
        best_a = matrices[:, :, 2].argmax(axis=0)
        best_b = matrices[:, :, 2].argmax(axis=1)
        precision = float(np.average(matrices[best_a, np.arange(len(b)), 0], weights=_weights(b, tokenizer)))
        recall = float(np.average(matrices[np.arange(len(a)), best_b, 1], weights=_weights(a, tokenizer)))
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return (precision, recall, f1)
    rows = []
    records = list(_records(pairs))
    try:
        for index, pair in enumerate(records, 1):
            a, b = [sentence_chunks(pair[key], tokenizer, budget) for key in ('original', 'modified')]
            row = _identity(pair)
            row.update(dict(zip(('bert_chunk_precision', 'bert_chunk_recall', 'bert_chunk_f1'), score_chunks(a, b))))
            row.update(bert_original_chunks=len(a), bert_modified_chunks=len(b), bert_num_layers=num_layers, bert_idf=False, bert_rescale_with_baseline=False)
            local = _local(pair)
            if local:
                ca, cb = [sentence_chunks(text, tokenizer, budget) for text in local]
                row.update(dict(zip(('local_bert_precision', 'local_bert_recall', 'local_bert_f1'), score_chunks(ca, cb))))
                row['local_bert_protocol'] = 'standard_sentence' if len(ca) == len(cb) == 1 else 'chunk_matched_approximation'
            else:
                row.update(local_bert_precision=np.nan, local_bert_recall=np.nan, local_bert_f1=np.nan, local_bert_protocol='unavailable')
            row['bert_unique_chunks_encoded_cumulative'] = len(seen_encoded_chunks)
            row['bert_unique_chunk_pairs_scored_cumulative'] = len(result_cache)
            rows.append(row)
            while token_embedding_bytes > token_cache_limit_bytes and token_embedding_cache:
                _, (vector, weight) = token_embedding_cache.popitem(last=False)
                token_embedding_bytes -= vector.numel() * vector.element_size() + weight.numel() * weight.element_size()
            if progress_callback and (index % 25 == 0 or index == len(records)):
                progress_callback(index, len(records))
    finally:
        del scorer
        gc.collect()
        _release_device_cache(device)
    return pd.DataFrame(rows)

def resolve_nli_labels(config, label_map=None):
   
    if label_map is not None:
        labels = {str(k).lower(): int(v) for k, v in label_map.items()}
    else:
        aliases = {'entailment': 'entailment', 'entails': 'entailment', 'contradiction': 'contradiction', 'contradictory': 'contradiction', 'neutral': 'neutral'}
        labels = {}
        for index, name in config.id2label.items():
            norm = str(name).strip().lower()
            if norm in aliases:
                labels[aliases[norm]] = int(index)
    if not {'entailment', 'contradiction'}.issubset(labels):
        raise ValueError('Checkpoint must name entailment/contradiction labels, or supply an explicit label_map')
    if labels['entailment'] == labels['contradiction']:
        raise ValueError('Entailment and contradiction cannot share an output index')
    size = getattr(config, 'num_labels', max((int(k) for k in config.id2label)) + 1)
    if any((index < 0 or index >= size for index in labels.values())):
        raise ValueError('NLI label indices are outside the checkpoint output range')
    return labels

def _summarize(support, contradiction, retrieved_contradiction, threshold, neutral=None):
    support, contradiction = (np.array(support), np.array(contradiction))
    if not len(support):
        raise ValueError('No hypothesis chunks were evaluated')
    return {'mean': float(support.mean()), 'min': float(support.min()), 'q10': float(np.quantile(support, 0.1)), 'low_support_fraction': float((support < threshold).mean()), 'contradiction_mean': float(contradiction.mean()), 'contradiction_max': float(contradiction.max()), 'retrieved_contradiction_max': float(max(retrieved_contradiction)), 'neutral_mean': float(np.mean(neutral)) if neutral else np.nan, 'n_hypothesis_chunks': len(support)}

def _nli_jobs(premises, hypotheses, candidates, tokenizer, max_pair_tokens=384):
    special = tokenizer.num_special_tokens_to_add(pair=True)
    budget = (max_pair_tokens - special) // 2
    if budget < 1:
        raise ValueError('NLI pair budget is too small')
    subpremises = [token_chunks(text, tokenizer, budget) for text in premises]
    jobs = []
    for hyp_idx, text in enumerate(hypotheses):
        for hypothesis in token_chunks(text, tokenizer, budget):
            evidence = [part for index in candidates[hyp_idx] for part in subpremises[index]]
            if not evidence:
                raise ValueError('Every hypothesis must have at least one evidence candidate')
            jobs.append([(premise, hypothesis) for premise in evidence])
    return jobs

def _nli_direction(premises, hypotheses, candidates, tokenizer, predict, labels, *, max_pair_tokens=384, low_support_threshold=0.5):
    jobs = _nli_jobs(premises, hypotheses, candidates, tokenizer, max_pair_tokens)
    unique_pairs = list(dict.fromkeys((pair for job in jobs for pair in job)))
    probabilities = predict(unique_pairs)
    lookup = dict(zip(unique_pairs, probabilities))
    support, contradiction, retrieved_contradiction, neutral = ([], [], [], [])
    for job in jobs:
        prob = np.array([lookup[pair] for pair in job])
        best = int(prob[:, labels['entailment']].argmax())
        support.append(prob[best, labels['entailment']])
        contradiction.append(prob[best, labels['contradiction']])
        retrieved_contradiction.append(float(prob[:, labels['contradiction']].max()))
        if 'neutral' in labels:
            neutral.append(float(prob[best, labels['neutral']]))
    return _summarize(support, contradiction, retrieved_contradiction, low_support_threshold, neutral)
from .nli import nli_scores

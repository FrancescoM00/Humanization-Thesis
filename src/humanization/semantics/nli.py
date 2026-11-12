"""Punteggi neurali di entailment su unità di frase allineate, senza classifica finale dei sistemi."""
from pathlib import Path
from collections import defaultdict,OrderedDict
import gc,hashlib,json,re
import numpy as np
import pandas as pd

def nli_scores(pairs, modelpath, alignment_cache=None, *, max_pair_tokens=384, low_support_threshold=0.5, batch_size=8, n_threads=4, label_map=None, probability_cache_path=None, progress_callback=None, cache_only=False, device='cpu'):
    """Supporto direzionale di frasi o segmenti tramite entailment massimo sulle evidenze.

    Forward: l'originale implica il modificato; reverse: il modificato implica
    l'originale. Si usano le evidenze top-k se presenti in cache, altrimenti
    si confrontano tutte le evidenze. Il mancato supporto nel recupero non
    prova una contraddizione o un errore fattuale. I riepiloghi di
    contraddizione usano l'evidenza con supporto migliore;
    retrieved_contradiction_max conserva anche il massimo su *tutti* i
    candidati. I bersagli locali modificati evitano recupero e diluizione.
    Una cache opzionale JSONL delle probabilità salva ogni batch di inferenza.
    L'intestazione la lega al percorso, alla configurazione e ai metadati dei
    pesi del checkpoint locale: un checkpoint diverso viene rifiutato,
    evitando di riusare score non più validi.
    """
    from .metrics import sentence_chunks, _cached_chunks, _cpu_threads, _identity, _local, _model_path, _nli_direction, _records, _release_device_cache, _resolve_device, resolve_nli_labels
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = _model_path(modelpath)
    _cpu_threads(n_threads)
    device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model_options = {'local_files_only': True}
    if device.startswith('cuda'):
        model_options['torch_dtype'] = torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(path, **model_options).to(device).eval()
    labels = resolve_nli_labels(model.config, label_map)
    configured_limit = min(int(getattr(tokenizer, 'model_max_length', 512)), int(getattr(model.config, 'max_position_embeddings', 512)))
    limit = min(max_pair_tokens, configured_limit)
    probability_cache = {}
    n_model_scored = 0
    cache_file = Path(probability_cache_path) if probability_cache_path else None
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        weights_metadata = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(Path(path).glob('*')) if p.suffix in {'.safetensors', '.bin'}]
        header = {'type': 'nli_probability_cache_v1', 'model_path': path, 'config_sha256': hashlib.sha256((Path(path) / 'config.json').read_bytes()).hexdigest(), 'weight_files': [list(item) for item in weights_metadata]}
        if device.startswith('cuda'):
            header.update(inference_device=device, inference_precision='float32')
        if cache_file.exists():
            with cache_file.open('rb') as handle:
                first = handle.readline()
                if json.loads(first) != header:
                    raise ValueError('NLI probability cache belongs to a different model checkpoint')
                valid_bytes = len(first)
                all_lines = handle.readlines()
            for index, line in enumerate(all_lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if index != len(all_lines) - 1 or line.endswith(b'\n'):
                        raise ValueError('Corrupt NLI probability cache') from None
                    with cache_file.open('r+b') as handle:
                        handle.truncate(valid_bytes)
                    break
                probability_cache[record['key']] = np.asarray(record['probabilities'], dtype=float)
                valid_bytes += len(line)
                if index == len(all_lines) - 1 and (not line.endswith(b'\n')):
                    with cache_file.open('ab') as handle:
                        handle.write(b'\n')
        else:
            with cache_file.open('x', encoding='utf-8') as handle:
                handle.write(json.dumps(header, sort_keys=True) + '\n')

    def probability_key(pair):
        return hashlib.sha256(json.dumps(pair, ensure_ascii=False).encode('utf-8')).hexdigest()

    def predict(text_pairs):
        nonlocal n_model_scored
        missing = list(dict.fromkeys((pair for pair in text_pairs if probability_key(pair) not in probability_cache)))
        if cache_only and missing:
            raise ValueError('NLI cache-only scoring found missing evidence probabilities; complete precomputation first')
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            encoded = tokenizer([p[0] for p in batch], [p[1] for p in batch], padding=True, truncation=False, return_tensors='pt')
            if encoded['input_ids'].shape[1] > limit:
                raise ValueError('NLI tokenizer exceeded the explicit safe pair budget; no text was truncated')
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                probability = model(**encoded).logits.softmax(-1).cpu().numpy()
            n_model_scored += len(batch)
            appended = []
            for pair, prob in zip(batch, probability):
                key = probability_key(pair)
                probability_cache[key] = prob
                appended.append({'key': key, 'probabilities': [float(v) for v in prob]})
            if cache_file is not None:
                with cache_file.open('a', encoding='utf-8') as handle:
                    for record in appended:
                        handle.write(json.dumps(record, separators=(',', ':')) + '\n')
                    handle.flush()
        return np.array([probability_cache[probability_key(pair)] for pair in text_pairs])
    rows = []
    records = list(_records(pairs))
    try:
        for index, pair in enumerate(records, 1):
            n_before = n_model_scored
            item = _cached_chunks(pair, alignment_cache)
            if item is None:
                budget = (limit - tokenizer.num_special_tokens_to_add(pair=True)) // 2
                a, b = [sentence_chunks(pair[key], tokenizer, budget) for key in ('original', 'modified')]
                forward = [list(range(len(a))) for _ in b]
                reverse = [list(range(len(b))) for _ in a]
            else:
                a, b = (item['original_chunks'], item['modified_chunks'])
                forward, reverse = (item['forward_candidates'], item['reverse_candidates'])
            row = _identity(pair)
            for prefix, premise, hypothesis, candidates in (('nli_forward_', a, b, forward), ('nli_reverse_', b, a, reverse)):
                values = _nli_direction(premise, hypothesis, candidates, tokenizer, predict, labels, max_pair_tokens=limit, low_support_threshold=low_support_threshold)
                row.update({prefix + k: v for k, v in values.items()})
            row.update(nli_retrieval_mode='top_k_cosine' if item else 'exhaustive', nli_max_pair_tokens=limit, nli_retrieval_top_k=item.get('retrieval_top_k', 0) if item else 0, nli_low_support_threshold=low_support_threshold)
            local = _local(pair)
            if local:
                for prefix, premise, hypothesis in (('local_nli_forward_', local[0], local[1]), ('local_nli_reverse_', local[1], local[0])):
                    values = _nli_direction([premise], [hypothesis], [[0]], tokenizer, predict, labels, max_pair_tokens=limit, low_support_threshold=low_support_threshold)
                    row.update({prefix + k: v for k, v in values.items()})
            row['nli_unique_pairs_scored_cumulative'] = len(probability_cache)
            row['nli_new_pairs_scored'] = n_model_scored - n_before
            rows.append(row)
            if progress_callback and (index % 25 == 0 or index == len(records)):
                progress_callback(index, len(records))
    finally:
        del model
        gc.collect()
        _release_device_cache(device)
    return pd.DataFrame(rows)

from .contextual_nli import LocalNLIPredictor,build_plan,score_plan

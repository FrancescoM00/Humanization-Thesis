"""Confronta i contesti delle evidenze NLI senza cambiare il precedente calcolo per frase.

Le unità dell'ipotesi restano fisse. Le premesse sono unità singole, unioni
contigue di fino a due o tre unità, oppure l'intero documento esatto quando
la coppia rientra nei limiti del tokenizer. Nessuna evidenza viene troncata.
Aumentare i candidati alza per costruzione il supporto massimo: un punteggio
più alto non dimostra una maggiore accuratezza.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PROTOCOL = 'contextual_nli_v1'
MODES = ('single_all', 'window2', 'window3', 'document')


def digest_pair(pair):
    return hashlib.sha256(json.dumps(list(pair), ensure_ascii=False).encode()).hexdigest()


def contiguous_contexts(units, max_units):
    """Tutte le finestre contigue non vuote di ampiezza 1..max_units, nell'ordine originale."""
    if max_units < 1 or not units or any(not text.strip() for text in units):
        raise ValueError('Require nonempty units and a positive window width')
    return list(dict.fromkeys(' '.join(units[start:start+width])
                             for width in range(1, min(max_units, len(units))+1)
                             for start in range(len(units)-width+1)))


def pair_length(tokenizer, premise, hypothesis):
    return len(tokenizer.encode(premise, hypothesis, add_special_tokens=True,
                                truncation=False))


def build_plan(pair, alignment, tokenizer, *, max_pair_tokens=384, modes=MODES,
               length_cache=None):
    """Pianifica confronti esatti verificati dal tokenizer e registra la copertura mancante.

    Se fornite, le unità di allineamento esistenti vengono riusate senza modifiche,
    verificandone gli hash originali. Per gli allineamenti mancanti si usa la stessa
    funzione di segmentazione del calcolo esistente, con 160 token per unità
    dell'ipotesi. Una cache condivisa opzionale conserva le lunghezze esatte delle
    coppie per questo tokenizer, mai approssimazioni.
    """
    from humanization.semantics.metrics import sentence_chunks

    if max_pair_tokens < 4 or not modes or any(mode not in MODES for mode in modes):
        raise ValueError('Invalid pair budget or context modes')
    item = alignment.get(pair['id']) if alignment else None
    if item:
        for field in ('original', 'modified'):
            sha = hashlib.sha256(pair[field].encode()).hexdigest()
            if item.get(field+'_sha256') != sha:
                raise ValueError('Alignment text hash differs from the selected pair')
        original, modified = item['original_chunks'], item['modified_chunks']
    else:
        budget = min(160, (max_pair_tokens-tokenizer.num_special_tokens_to_add(pair=True))//2)
        original, modified = [sentence_chunks(pair[k], tokenizer, budget) for k in ('original', 'modified')]
    plan = []
    def length(premise, hypothesis):
        if length_cache is None:
            return pair_length(tokenizer, premise, hypothesis)
        key = (id(tokenizer), premise, hypothesis)
        if key not in length_cache:
            length_cache[key] = pair_length(tokenizer, premise, hypothesis)
        return length_cache[key]
    for direction, premise_units, hypotheses, document in (
            ('forward', original, modified, pair['original']),
            ('reverse', modified, original, pair['modified'])):
        for mode in modes:
            contexts = [document] if mode == 'document' else contiguous_contexts(
                premise_units, {'single_all': 1, 'window2': 2, 'window3': 3}[mode])
            for index, hypothesis in enumerate(hypotheses):
                eligible = [p for p in contexts if length(p, hypothesis) <= max_pair_tokens]
                plan.append({'id': pair['id'], 'direction': direction, 'mode': mode,
                             'hypothesis_index': index, 'hypothesis': hypothesis,
                             'premises': eligible, 'n_candidates_total': len(contexts),
                             'n_candidates_eligible': len(eligible),
                             'n_candidates_over_budget': len(contexts)-len(eligible),
                             'max_pair_tokens': max_pair_tokens})
    return plan


def summarize_units(units, *, low_support_threshold=.5):
    """Una copertura incompleta del documento non produce mai una media dei soli casi disponibili."""
    groups = {}
    for row in units:
        groups.setdefault((row['id'], row['direction'], row['mode']), []).append(row)
    summaries = []
    for (row_id, direction, mode), part in groups.items():
        scores = [r['entailment'] for r in part if r['eligible']]
        complete = len(scores) == len(part)
        values = np.array(scores, dtype=float)
        summaries.append({'id': row_id, 'direction': direction, 'mode': mode,
            'n_hypotheses': len(part), 'n_eligible_hypotheses': len(scores),
            'coverage_fraction': len(scores)/len(part), 'complete': complete,
            'mean': float(values.mean()) if complete else np.nan,
            'q10': float(np.quantile(values, .1)) if complete else np.nan,
            'min': float(values.min()) if complete else np.nan,
            'low_support_fraction': float((values < low_support_threshold).mean()) if complete else np.nan,
            'low_support_threshold': low_support_threshold,
            'candidate_count_mean': float(np.mean([r['n_candidates_eligible'] for r in part])),
            'over_budget_candidates': sum(r['n_candidates_over_budget'] for r in part)})
    return summaries


def score_plan(plan, predict, labels, *, low_support_threshold=.5):
    unique = list(dict.fromkeys((p, row['hypothesis']) for row in plan for p in row['premises']))
    probability = predict(unique)
    if len(probability) != len(unique):
        raise ValueError('Prediction count does not match the explicit pair plan')
    lookup = dict(zip(unique, probability))
    units = []
    for row in plan:
        record = {k: v for k, v in row.items() if k != 'premises'}
        record['eligible'] = bool(row['premises'])
        if not row['premises']:
            record.update(entailment=None, contradiction=None, neutral=None, best_premise=None)
        else:
            probabilities = np.array([lookup[p, row['hypothesis']] for p in row['premises']])
            best = int(probabilities[:, labels['entailment']].argmax())
            probs = probabilities[best]
            record.update(entailment=float(probs[labels['entailment']]),
                          contradiction=float(probs[labels['contradiction']]),
                          neutral=float(probs[labels['neutral']]) if 'neutral' in labels else None,
                          best_premise=row['premises'][best])
        units.append(record)
    return summarize_units(units, low_support_threshold=low_support_threshold), units


class LocalNLIPredictor:
    """Checkpoint base locale, float32 esplicito e cache separata delle coppie riprendibile."""
    def __init__(self, modelpath, tokenizer, *, device='cpu', batch_size=2,
                 threads=2, max_pair_tokens=384, cache_path=None, progress=None):
        import torch
        from transformers import AutoModelForSequenceClassification
        from humanization.semantics.metrics import _resolve_device, resolve_nli_labels

        self.path = Path(modelpath).resolve()
        self.tokenizer = tokenizer
        self.device = _resolve_device(device)
        self.batch_size = max(1, batch_size)
        self.limit = max_pair_tokens
        self.progress = progress
        self.n_new = 0
        torch.set_num_threads(max(1, min(threads, 4)))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.path), local_files_only=True, torch_dtype=torch.float32).to(self.device).float().eval()
        self.labels = resolve_nli_labels(self.model.config)
        safe_limit = min(int(getattr(self.model.config, 'max_position_embeddings', 512)),
                         int(getattr(tokenizer, 'model_max_length', 512)))
        if self.limit > safe_limit:
            raise ValueError('Requested pair budget exceeds tokenizer/checkpoint context')
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        header = {'type': PROTOCOL+'_probabilities', 'model_path': str(self.path),
                  'config_sha256': hashlib.sha256((self.path/'config.json').read_bytes()).hexdigest(),
                  'weight_files': [[p.name, p.stat().st_size, p.stat().st_mtime_ns]
                                   for p in sorted(self.path.iterdir()) if p.suffix in {'.bin', '.safetensors'}],
                  'device': self.device, 'precision': 'float32', 'max_pair_tokens': self.limit}
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            if self.cache_path.exists():
                lines = self.cache_path.read_text().splitlines()
                if not lines or json.loads(lines[0]) != header:
                    raise ValueError('Contextual NLI cache belongs to another checkpoint/protocol/device')
                for line in lines[1:]:
                    r = json.loads(line)
                    self.cache[r['key']] = np.array(r['probabilities'], dtype=float)
            else:
                self.cache_path.write_text(json.dumps(header, sort_keys=True)+'\n')

    def __call__(self, pairs):
        import torch
        missing = list(dict.fromkeys(pair for pair in pairs if digest_pair(pair) not in self.cache))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start+self.batch_size]
            encoded = self.tokenizer([p for p, h in batch], [h for p, h in batch],
                                     padding=True, truncation=False, return_tensors='pt')
            if encoded['input_ids'].shape[1] > self.limit:
                raise ValueError('Tokenizer exceeded budget; no evidence has been truncated')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                probabilities = self.model(**encoded).logits.softmax(-1).cpu().numpy()
            appended = []
            for pair, prob in zip(batch, probabilities):
                key = digest_pair(pair)
                self.cache[key] = prob
                appended.append({'key': key, 'probabilities': [float(x) for x in prob]})
            if self.cache_path:
                with self.cache_path.open('a') as f:
                    for row in appended:
                        f.write(json.dumps(row, separators=(',', ':'))+'\n')
                    f.flush()
            self.n_new += len(batch)
            if self.progress:
                self.progress(min(start+self.batch_size, len(missing)), len(missing))
        return np.array([self.cache[digest_pair(pair)] for pair in pairs])

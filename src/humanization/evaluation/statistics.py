"""Bootstrap per gruppi di fonte e calcolo degli intervalli, senza inferenza dei modelli."""
from collections import defaultdict
import hashlib,math,json
import numpy as np
import pandas as pd

def digest_seed(*parts):
    value = json.dumps(list(parts), sort_keys=True, ensure_ascii=False)
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)

def historical_seed(parts, seed):
    """Conserva il seme precedente con valore 42, consentendo un nuovo seme esplicito."""
    return digest_seed(*parts) if seed == 42 else digest_seed(seed, *parts)

def bootstrap_mean(values, groups, seed, draws):
    if not len(values):
        return (math.nan, math.nan)
    data = pd.DataFrame({'group': groups, 'value': np.asarray(values, dtype=float)})
    grouped = data.groupby('group', sort=True).value.agg(['sum', 'count']).to_numpy()
    index = np.random.default_rng(seed).integers(0, len(grouped), size=(draws, len(grouped)))
    sample = grouped[index].sum(axis=1)
    return tuple(np.quantile(sample[:, 0] / sample[:, 1], [0.025, 0.975]))

def wilson(successes, total):
    if not total:
        return (math.nan, math.nan)
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return (0.0 if not successes else max(0.0, center - half), 1.0 if successes == total else min(1.0, center + half))

def _asr_interval(success, group_ids, seed, n_bootstrap):
    if not len(success) or n_bootstrap < 1:
        return (math.nan, math.nan)
    grouped = defaultdict(list)
    for value, group in zip(success, group_ids):
        grouped[group].append(float(value))
    aggregates = np.array([(sum(values), len(values)) for _, values in sorted(grouped.items())])
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(aggregates), size=(n_bootstrap, len(aggregates)))
    sampled = aggregates[index].sum(axis=1)
    return tuple(np.quantile(sampled[:, 0] / sampled[:, 1], [0.025, 0.975]))
asr_interval = _asr_interval

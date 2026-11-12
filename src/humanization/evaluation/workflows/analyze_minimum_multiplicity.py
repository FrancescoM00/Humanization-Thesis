"""Quantifica l'effetto delle molte occasioni sul minimo entro coppie corrette fisse.

Usa solo output salvati dei modelli. L'evidenza della premessa resta fissa;
campiona uniformemente m unità di ipotesi dai due vettori di supporto
direzionale della stessa coppia. Il minimo atteso è esatto, senza Monte Carlo
o ulteriore inferenza neurale.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from humanization.semantics.metrics import _nli_jobs
from humanization.evaluation.reporting import markdown_table

def expected_minimum(values, m):
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    denominator = math.comb(n, m)
    return sum((float(values[i]) * math.comb(n - i - 1, m - 1) / denominator for i in range(n - m + 1)))

def _argument_error(message):
    raise ValueError(message)

def run(args):
    data = ROOT / 'data/processed/semantic_extension'
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(x) for x in (data / 'positive_pairs.jsonl').read_text().splitlines()]
    alignment = json.loads((data / 'alignment.json').read_text())
    lines = (data / 'nli_probability_cache.jsonl').read_text().splitlines()
    probabilities = {x['key']: x['probabilities'] for x in map(json.loads, lines[1:])}
    config = json.loads((ROOT / 'data/models/nli/config.json').read_text())
    entailment = next((int(i) for i, label in config['id2label'].items() if label.lower() == 'entailment'))
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'data/models/nli', local_files_only=True)
    scores = pd.read_csv(ROOT / 'results/published/03_semantics/extension/nli.csv').set_index('id')
    arrays = {}
    for filename, name in [('aligned_cosine_units.jsonl', 'Coseno'), ('aligned_bertscore_units.jsonl', 'BERTScore riscalato')]:
        for row in map(json.loads, (ROOT / 'results/published/03_semantics/fairness/aligned_old' / filename).read_text().splitlines()):
            support = np.array(row['forward_support'] + row['reverse_support'])
            if name.startswith('BERT'):
                support = (support - 0.814463) / (1 - 0.814463)
            arrays[row['id'], name] = support
    rows = []
    for pair in pairs:
        item = alignment[pair['id']]
        support = []
        for original, modified, candidates in [(item['original_chunks'], item['modified_chunks'], item['forward_candidates']), (item['modified_chunks'], item['original_chunks'], item['reverse_candidates'])]:
            for alternatives in _nli_jobs(original, modified, candidates, tokenizer, 384):
                values = []
                for premise, hypothesis in alternatives:
                    key = hashlib.sha256(json.dumps((premise, hypothesis), ensure_ascii=False).encode()).hexdigest()
                    values.append(probabilities[key][entailment])
                support.append(max(values))
        old = scores.loc[pair['id']]
        assert np.isclose(min(support), min(old.nli_forward_min, old.nli_reverse_min), atol=1e-07, rtol=0)
        arrays[pair['id'], 'NLI'] = np.array(support)
        if len(support) < 40:
            continue
        for name in ['NLI', 'Coseno', 'BERTScore riscalato']:
            values = arrays[pair['id'], name]
            assert len(values) == len(support)
            for m in [5, 10, 20, 40]:
                rows.append({'id': pair['id'], 'source_group': pair['source_group_id'], 'metric': name, 'sampled_hypothesis_units': m, 'available_units': len(values), 'expected_minimum': expected_minimum(values, m), 'expected_unit_mean': float(values.mean())})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'pair_expectations.csv', index=False)
    rng = np.random.default_rng(42)
    summary = []
    for (metric, m), part in frame.groupby(['metric', 'sampled_hypothesis_units']):
        sources = part.groupby('source_group').expected_minimum.mean()
        boot = rng.choice(sources.to_numpy(), size=(1000, len(sources)), replace=True).mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        summary.append({'metric': metric, 'sampled_hypothesis_units': m, 'n_pairs': len(part), 'n_sources': len(sources), 'mean': sources.mean(), 'ci_low': low, 'ci_high': high})
    table = pd.DataFrame(summary)
    table.to_csv(output / 'summary.csv', index=False)
    contrasts = []
    for metric, part in frame.groupby('metric'):
        pivot = part.pivot(index=['id', 'source_group'], columns='sampled_hypothesis_units', values='expected_minimum')
        delta = (pivot[40] - pivot[5]).groupby(level='source_group').mean()
        boot = rng.choice(delta.to_numpy(), size=(1000, len(delta)), replace=True).mean(axis=1)
        contrasts.append({'metric': metric, 'n_sources': len(delta), 'mean_40_minus_5': delta.mean(), 'ci_low': np.quantile(boot, 0.025), 'ci_high': np.quantile(boot, 0.975)})
    contrast = pd.DataFrame(contrasts)
    contrast.to_csv(output / 'paired_contrasts.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, name in zip(axes, ['NLI', 'Coseno', 'BERTScore riscalato']):
        part = table[table.metric == name].sort_values('sampled_hypothesis_units')
        ax.plot(part.sampled_hypothesis_units, part['mean'], marker='o', color='#167c80')
        ax.fill_between(part.sampled_hypothesis_units, part.ci_low, part.ci_high, alpha=0.2, color='#167c80')
        ax.set_title(name)
        ax.set_xlabel('Unità ipotesi controllate')
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel('Minimo atteso, stesse coppie positive')
    fig.tight_layout()
    fig.savefig(output / 'multiplicity.png', dpi=180)
    fig.savefig(output / 'multiplicity.pdf')
    plt.close(fig)
    text = ['# Numero di opportunità: controllo entro la stessa coppia', '', 'Si usano soltanto coppie OpAI presunte conservative con almeno 40 unità ipotesi complessive nelle due direzioni. La coorte è identica per tutti i modelli e tutti i valori di m. Le evidenze e i punteggi delle unità rimangono fissi: si calcola esattamente il minimo atteso su un sottoinsieme uniforme di m unità, senza sostituirle e senza nuove inferenze. Le direzioni sono riunite: m conta ipotesi verificate, non frasi uniche del documento.', '', markdown_table(table), '', markdown_table(contrast), '', '![Effetto del numero di ipotesi](multiplicity.png)', '', 'La monotonia del minimo è una proprietà matematica; qui si quantifica la sua entità sui punteggi osservati, a contenuto e qualità della coppia fissi. La media delle unità ha invece aspettativa costante per costruzione. Questo non è un esperimento su documenti troncati né dimostra accuratezza semantica. Gli IC ricampionano le sorgenti, mantenendo insieme editor e operazioni.']
    (output / 'report.md').write_text('\n'.join(text) + '\n')
    print(table.to_string(index=False))
    print(contrast.to_string(index=False))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'output': ROOT / 'results/new_runs/03_semantics/fairness/minimum_multiplicity'}
    values.update(overrides)
    return SimpleNamespace(**values)

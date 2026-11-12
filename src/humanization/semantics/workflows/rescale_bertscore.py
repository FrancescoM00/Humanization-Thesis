#!/usr/bin/env python3
"""Applica agli score grezzi salvati l'esatta trasformazione affine della baseline della libreria.

Gli score documentali restano approssimazioni basate sui segmenti. Il
riscalamento è monotono e non cambia gli ordinamenti F1 o la ROC-AUC.
Non richiede inferenze aggiuntive.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pandas as pd

from humanization.paths import project_root

ROOT = project_root()


def rescale(input_path, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.find_spec('bert_score')
    baseline_path = Path(spec.origin).parent/'rescale_baseline/en/roberta-base.tsv'
    baseline = pd.read_csv(baseline_path).set_index('LAYER').loc[10]
    frame = pd.read_csv(input_path)
    for prefix in ['bert_chunk_','local_bert_']:
        for metric, name in [('precision','P'),('recall','R'),('f1','F')]:
            column = prefix+metric
            if column in frame:
                value = float(baseline[name])
                frame[column+'_baseline_distance'] = frame[column]-value
                frame[column+'_rescaled'] = (frame[column]-value)/(1-value)
    frame.to_csv(output_path,index=False)
    metadata = {'model':'FacebookAI/roberta-base','language':'en','num_layers':10,
                'idf':False,'baseline_values':baseline.to_dict(),
                'baseline_path':str(baseline_path.relative_to(ROOT) if baseline_path.is_relative_to(ROOT) else baseline_path),
                'baseline_sha256':hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                'transform':'(raw - baseline) / (1 - baseline), separately for P, R and F',
                'library_equivalence':'BERTScorer rescale_with_baseline=True uses this exact affine transform after raw inference.',
                'document_protocol':'Affine-rescaled chunk-matched approximation; not standard full-document BERTScore.',
                'rank_and_auc_unchanged':True,'n_pairs':len(frame),
                'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest()}
    output_path.with_suffix('.baseline.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/published/03_semantics/extension/bertscore.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'results/new_runs/03_semantics/extension/bertscore_rescaled.csv')
    args=parser.parse_args()
    rescale(args.input,args.output)

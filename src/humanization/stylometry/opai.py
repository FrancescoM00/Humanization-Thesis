"""Modifiche stilistiche OpAI: feature appaiate e distanze sulle frasi modificate."""
import json
import numpy as np
import pandas as pd
from humanization.configuration import path,output,record


def run(config):
    out=output(config['output']);mode=config.get('mode','reproduce');scope=config.get('scope','both')
    if mode=='reproduce':
        from .workflows.run_opai_style import summarize,SHORTLIST
        root=path(config['reference'])
        frame=pd.read_csv(root/'document/paired_feature_values.csv')
        result=summarize(frame,SHORTLIST,np.random.default_rng(config.get('seed',42)),config.get('bootstrap',1000),'full_document')
        result.to_csv(out/'document_feature_summary.csv',index=False)
        units=pd.read_csv(root/'touched/aligned_sentence_distances.csv')
        # Tutte le modifiche dello stesso originale restano nello stesso gruppo prima della media.
        units=units[units.eligible_for_distance.eq(True)]
        grouped=units.groupby(['source_id','operation'])[['unit_distance_rms','unit_pos_js']].mean().reset_index()
        rows=[];rng=np.random.default_rng(config.get('seed',42));n=config.get('bootstrap',1000)
        for operation,part in grouped.groupby('operation'):
            for metric in ['unit_distance_rms','unit_pos_js']:
                values=part[metric].dropna().to_numpy();draw=rng.choice(values,(n,len(values)),replace=True).mean(1)
                low,high=np.quantile(draw,[.025,.975]);rows.append(dict(operation=operation,metric=metric,n_sources=len(values),mean=float(values.mean()),ci95_low=float(low),ci95_high=float(high)))
        pd.DataFrame(rows).to_csv(out/'touched_distance_summary.csv',index=False)
    elif mode=='compute':
        from .workflows import run_opai_style as document,run_opai_touched_style as touched
        data=output(path('data/processed/generated/opai_editing'))
        if scope in {'document','both'}:
            document.run(document.default_arguments(opai_dir=path(config['raw_dir']),output=output(out/'document'),data_dir=data,n=config.get('n',200),seed=config.get('seed',42),bootstrap=config.get('bootstrap',1000)))
        if scope in {'touched','both'}:
            touched.run(touched.default_arguments(opai_dir=path(config['raw_dir']),pairs=data/'pairs.jsonl',data_dir=output(data/'touched'),output=output(out/'touched'),full_features=out/'document/paired_feature_values.csv',seed=config.get('seed',42),bootstrap=config.get('bootstrap',1000)))
    else:raise ValueError('OpAI editing supports reproduce or compute')
    return record(out,config,mode,scope=scope)

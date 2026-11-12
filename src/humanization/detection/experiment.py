"""Produce individual detector scores; ASR and intervals are evaluated later."""
import hashlib,json
import pandas as pd
from humanization.configuration import path,output,record,model_config
from humanization.datasets.records import read_jsonl,text_records,_score_key


def run(config):
    mode=config.get('mode','replay');out=output(config['output']);cohort=path(config['cohort']);rewrites=path(config['rewrites'])
    if mode in {'replay','reproduce'}:
        expected={_score_key(x):x for x in text_records(read_jsonl(cohort),read_jsonl(rewrites))}
        reference=path(config['reference'])
        for folder,name,destination in [('detection','detector_scores.csv',out),('zero_shot','zero_shot_scores.csv',output(path(config['run_dir'])/'zero_shot'))]:
            file=reference/folder/name
            if not file.exists() or (folder=='zero_shot' and not config.get('include_binoculars',True)):continue
            frame=pd.read_csv(file).fillna({'humanizer':''})
            frame=frame[[(_score_key(x) in expected) for x in frame.to_dict('records')]]
            for row in frame.to_dict('records'):
                if row.get('text_sha256')!=hashlib.sha256(expected[_score_key(row)]['text'].encode()).hexdigest():
                    raise ValueError('Cached detector scores do not match the actual rewritten text; use compute')
            frame.to_csv(destination/name,index=False)
    elif mode=='compute':
        from .pipeline import run as score
        configfile=model_config(config,out)
        score(cohort,rewrites,configfile,out,config.get('device','cpu'),evaluate=False)
        if config.get('include_binoculars',True):
            from .binoculars import run as binoculars
            binoculars(cohort,rewrites,configfile,output(path(config['run_dir'])/'zero_shot'),observer_device=config.get('observer_device','cuda:0'),performer_device=config.get('performer_device','cuda:1'),evaluate=False)
    else:raise ValueError('Detection supports replay or compute')
    return record(out,config,mode,models_executed=mode=='compute',final_metrics_computed=False)

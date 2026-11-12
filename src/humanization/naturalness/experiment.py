"""Offline judge replay, local AB/BA preparation and explicit API execution."""
import json
import pandas as pd
from humanization.configuration import path,output,record
from humanization.evaluation.naturalness import summarize_orders,evaluate_orders


def run(config,execute=False):
    mode=config.get('mode','reproduce');out=output(config['output'])
    if mode=='reproduce':
        summaries=[];diagnostics=[]
        for folder in sorted(path(config['reference']).iterdir()):
            file=folder/'judgments.jsonl'
            if not file.exists():continue
            rows=[json.loads(x) for x in file.read_text().splitlines() if x.strip()]
            diagnostics.append(dict(comparison=folder.name,**evaluate_orders(rows)))
            for row in summarize_orders(rows):summaries.append(dict(comparison=folder.name,**row))
        frame=pd.DataFrame(summaries);frame.to_csv(out/'order_consistency.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(out/'comparison_diagnostics.csv',index=False)
        results={'n_pairs':len(frame),'n_order_consistent':int(frame.order_consistent.sum()) if len(frame) else 0,'reference_is_human':False,'human_validation_available':False}
        (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        return record(out,config,mode,models_executed=False,api_executed=False,**results)
    if execute and not config.get('enabled',False):raise ValueError('Set enabled: true in naturalness.yaml before explicit --execute')
    from .pairwise import load_pairs,make_plan,read_jsonl
    from .pipeline import run_plan
    source_key=None
    if config.get('failed_generation_policy')=='report_and_exclude':
        data=read_jsonl(path(config['rewrites']));systems={config['humanizer_a'],config['humanizer_b']}
        grouped={}
        for row in data:
            if row['humanizer'] in systems:grouped.setdefault(row['source_id'],{})[row['humanizer']]=row
        eligible=[];excluded=[]
        for source,rows in sorted(grouped.items()):
            if set(rows)==systems and all(r.get('status')=='ok' for r in rows.values()):eligible.append({'source_id':source})
            else:excluded.append({'source_id':source,'reason':'at least one selected generation failed or is missing'})
        source_key=out/'eligible_sources.csv'
        pd.DataFrame(eligible,columns=['source_id']).to_csv(source_key,index=False)
        pd.DataFrame(excluded,columns=['source_id','reason']).to_csv(out/'excluded_sources.csv',index=False)
    pairs=load_pairs([path(config['rewrites'])],config['humanizer_a'],config['humanizer_b'],source_key=source_key,max_pairs=config.get('max_pairs',6),seed=config.get('seed',42))
    plan=make_plan(pairs,config['model'],reasoning_effort='none')
    manifest=run_plan(plan,out,execute=execute,max_requests=config.get('max_requests',16))
    return record(out,config,mode,models_executed=False,api_executed=bool(execute),judge_manifest=manifest)

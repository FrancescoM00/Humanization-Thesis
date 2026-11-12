"""Produce nuove riscritture oppure riutilizza esplicitamente quelle di riferimento congelate."""
import json,shutil
from humanization.configuration import path,output,record,model_config
from .pipeline import run_humanizers


def run(config):
    mode=config.get('mode','replay');out=output(config['output'])
    root=output(config['run_dir']);cohorts=output(root/'cohorts')
    shutil.copy2(path(config['cohort']),cohorts/'study.jsonl')
    pilot=path(config.get('pilot_cohort','data/processed/study_126/cohorts/pilot.jsonl'))
    if pilot.exists():shutil.copy2(pilot,cohorts/'pilot.jsonl')
    selected=config.get('selected')
    if mode in {'replay','reproduce'}:
        rows=[json.loads(x) for x in (path(config['reference'])/'rewrites.jsonl').read_text().splitlines()]
        if selected:rows=[r for r in rows if r['humanizer'] in selected]
        (out/'rewrites.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        (out/'pairs.jsonl').write_text((out/'rewrites.jsonl').read_text())
        return record(out,config,'replay',models_executed=False,n_rewrites=len(rows),selected=selected)
    if mode!='compute':raise ValueError('Rewriting supports replay or compute')
    result=run_humanizers(path(config['cohort']),model_config(config,out),out,device=config.get('device','cuda:0'),only=selected)
    return record(out,config,mode,manifest=result)

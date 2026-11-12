"""Compute per-pair semantic scores or reanalyse previously computed scores."""
import json
from humanization.configuration import path,output,record


def run(config):
    mode=config.get('mode','reproduce');out=output(config['output'])
    if mode=='reproduce':
        from humanization.evaluation.workflows import analyze_semantic_extension as workflow
        args=workflow.default_arguments(pairs=path(config['pairs']),scores=path(config['reference'])/'scores.csv',output=out,bootstrap=config.get('bootstrap',1000),seed=config.get('seed',42))
        workflow.run(args)
    elif mode=='compute':
        from .workflows import run_semantic_experiment as workflow
        rows=[json.loads(x) for x in path(config['rewrites']).read_text().splitlines()]
        pairs=[x for x in rows if x.get('status')=='ok']
        if not pairs:raise ValueError('No successful rewrites: run experiment04 compute first')
        run_root=output(config['run_dir']);data=output(run_root/'semantic');scores=output(data/'scores')
        (data/'pairs.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in pairs))
        models=path(config.get('models','data/models'))
        if config.get('download_models'):
            from .workflows import download_semantic_models as downloader
            revisions={'embedding':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41','bertscore':'e2da8e2f811d1448a5b465c236feacd80ffbac7b','nli':'6f5cf0a2b59cabb106aca4c287eed12e357e90eb'}
            for name,revision in revisions.items():downloader.run(downloader.default_arguments(model=name,revision=revision,output=models))
        for stage in config.get('stages',['lexical','embedding','bertscore','nli']):
            args=workflow.default_arguments(stage=stage,data_dir=data,output=scores,models=models,device=config.get('device','cpu'),threads=config.get('threads',4),nli_batch_size=config.get('batch_size',8))
            workflow.stage(args)
        from .workflows import score_contextual_nli as contextual
        contextual.run(contextual.default_arguments(pairs=data/'pairs.jsonl',alignment=data/'alignment.json',model=models/'nli',output=output(run_root/'semantic_context'),modes=config.get('nli_modes',['window2']),device=config.get('device','cpu'),batch_size=config.get('batch_size',8),threads=config.get('threads',4)))
        out=scores
    else:raise ValueError('Semantics supports reproduce or compute')
    return record(out,config,mode,note='Per-pair scores; final system comparisons are produced by experiment07')

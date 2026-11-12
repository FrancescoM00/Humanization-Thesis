"""Feature extraction and reproducible stylometric analyses."""
import shutil
from humanization.configuration import path,output,record


def run(config):
    config=dict(config);mode=config.get('mode','reproduce');analysis=config.get('analysis','shortlist')
    out=output(config['output'])
    if mode=='compute' and analysis=='rewrites':out=output(path(config['run_dir'])/'style')
    if mode=='reproduce':
        if analysis in {'shortlist','shortlist_100'}:
            if analysis=='shortlist':
                from humanization.evaluation.workflows import analyze_shortlist_validation as workflow
                source=path('results/published/02_stylometry/shortlist')
            else:
                from humanization.evaluation.workflows import analyze_shortlist_length_robustness as workflow
                source=path('results/published/02_stylometry/shortlist_length_robustness')
            profile='shortlist' if analysis=='shortlist' else 'shortlist_robustness'
            workflow.run(workflow.default_arguments(results=source,output=out,protocol=path('configs/protocols/'+profile+'.json')))
        else:
            import pandas as pd
            from humanization.evaluation.stylometric import effect_tables
            source=path('results/published/02_stylometry/'+analysis)
            frame=pd.read_csv(source/'features.csv')
            features=[x for x in frame.columns if x not in {'id','pair_id','group_id','domain','generator','label','source_row'} and pd.api.types.is_numeric_dtype(frame[x])]
            tables=effect_tables(frame,features,seed=config.get('seed',42),n_bootstrap=config.get('bootstrap',500))
            for name,table in tables.items():table.to_csv(out/(name+'.csv'),index=False)
    elif mode=='compute':
        if analysis in {'shortlist','shortlist_100'}:
            from humanization.paths import project_root
            if analysis=='shortlist':
                from humanization.evaluation.shortlist import run_validation
                run_validation(path('configs/protocols/shortlist.json'),out,project_root())
            else:
                from humanization.evaluation.shortlist_robustness import run_robustness
                run_robustness(path('configs/protocols/shortlist_robustness.json'),out,project_root())
            return record(out,config,mode,models_executed=False,classifiers_refitted=True,analysis=analysis)
        if analysis=='rewrites':
            from .workflows import score_humanizer_style as workflow
            args=workflow.default_arguments(cohort=path(config['cohort']),pairs=path(config['rewrites']),output=out)
        else:
            from .workflows import run_style_experiment,run_length_matched_style
            workflow=run_length_matched_style if analysis=='length_matched' else run_style_experiment
            args=workflow.default_arguments(raw_dir=path(config['raw_dir']),output=out,pairs=config.get('pairs',300),seed=config.get('seed',42),bootstrap=config.get('bootstrap',500))
        workflow.run(args)
    else:raise ValueError('Stylometry supports reproduce or compute')
    return record(out,config,mode,analysis=analysis)

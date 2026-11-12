"""L’interfaccia a riga di comando passa ai moduli le configurazioni lette dai file."""
import argparse,json
from .configuration import load,path,output,record


NAMES={'00':'datasets','01':'stylometry','02':'semantics','03':'opai','04':'humanizers','05':'detection','06':'naturalness','07':'evaluation'}


def main(number):
    parser=argparse.ArgumentParser(description='Humanization experiment '+number)
    parser.add_argument('--config',default='configs/'+NAMES[number]+'.yaml')
    parser.add_argument('--mode',choices=['reproduce','compute','replay','prepare'])
    parser.add_argument('--analysis')
    parser.add_argument('--scope',choices=['document','touched','both'])
    parser.add_argument('--device')
    parser.add_argument('--output')
    parser.add_argument('--bootstrap',type=int)
    parser.add_argument('--execute',action='store_true',help='Explicitly execute optional judge API calls')
    parser.add_argument('--download-models',action='store_true')
    args=parser.parse_args();config=load(args.config)
    for key in ['mode','analysis','scope','device','bootstrap','output']:
        if getattr(args,key) is not None:config[key]=getattr(args,key)
    if args.download_models:config['download_models']=True
    if args.execute and number!='06':parser.error('--execute applies only to the optional naturalness judge')
    if number=='00':
        from .datasets.workflows.import_mgtbench import inspect
        verified=inspect(path(config['raw_dir']))
        if config.get('mode')=='compute':
            from .datasets.mgtbench import prepare_dataset
            prepare_dataset(path(config['raw_dir']),output(config['processed_dir']),n_pairs=config.get('n_pairs',300),seed=config.get('seed',42))
        result={'verified_files':{n:{'rows':v['rows'],'sha256':v['sha256']} for n,v in verified.items()},'original_csvs_modified':False}
        directory=output('results/new_runs/data_verification');(directory/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    elif number=='01':
        from .stylometry.pipeline import run
        result=run(config)
    elif number=='02':
        from .semantics.pipeline import run
        result=run(config)
    elif number=='03':
        from .stylometry.opai import run
        result=run(config)
    elif number=='04':
        from .rewriting.experiment import run
        result=run(config)
    elif number=='05':
        from .detection.experiment import run
        result=run(config)
    elif number=='06':
        from .naturalness.experiment import run
        result=run(config,args.execute)
    else:
        from .evaluation.pipeline import run
        result=run(config)
    print(json.dumps(result,indent=2,ensure_ascii=False,default=str))

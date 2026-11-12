"""Valuta gli output salvati per testo, indipendentemente dall'esecuzione dei modelli."""
import json
from humanization.configuration import path,output,record
from .study import analyze_study,pilot_pair_comparison,compare_reference


def run(config):
    mode=config.get('mode','reproduce');out=output(config['output']);n=config.get('bootstrap',5000);seed=config.get('seed',42)
    if config.get('analysis')=='nli_calibration':
        from .workflows.calibrate_contextual_nli import analyze_scores
        analyze_scores(path('data/processed/nli_calibration'),path('results/published/08_nli_calibration/calibration/contextual_nli.csv'),out,bootstrap=n,seed=seed)
        return record(out,config,mode,models_executed=False,metrics_computed=True)
    if config.get('analysis')=='hedging':
        from .hedging import analyze,write_report
        source=path('results/published/hedging')
        import shutil
        for name in ['preparation_manifest.json','replacements.csv']:shutil.copy2(source/name,out/name)
        if (source/'discussione.md').is_file():shutil.copy2(source/'discussione.md',out/'discussione.md')
        analyze(source/'pairs.jsonl',source/'nli.csv',out,path('configs/protocols/hedging.json'),source/'assistant_review.csv');write_report(out)
        return record(out,config,mode,models_executed=False,metrics_computed=True)
    source=path(config['source']) if mode=='reproduce' else path(config['run_dir'])
    if mode!='reproduce':
        import pandas as pd
        from humanization.configuration import load
        from .detection import calibrate_threshold,calibrate_low_threshold
        settings=load('configs/detection.yaml')
        for folder,filename,threshold_file,calibrate in [('detection','detector_scores.csv','thresholds.json',calibrate_threshold),('zero_shot','zero_shot_scores.csv','zero_shot_thresholds.json',calibrate_low_threshold)]:
            scorefile=source/folder/filename
            if not scorefile.exists():
                if folder=='detection':raise FileNotFoundError('Run experiment05 for the current rewrites before evaluation')
                continue
            frame=pd.read_csv(scorefile)
            calibration=frame[frame.role.eq('calibration_human')&frame.split.eq('calibration')]
            thresholds={name:calibrate(part.score.to_numpy(),settings.get('calibration_fpr',.05)) for name,part in calibration.groupby('detector')}
            (source/folder/threshold_file).write_text(json.dumps(thresholds,indent=2)+'\n')
    reference=path(config['reference']) if mode=='reproduce' else None
    result=analyze_study(source,path(config['pilot_cohort']),output(out/'study_126'),n,seed,reference)
    if result.get('reference_mismatched_columns'):raise RuntimeError('Study numerical reproduction differs from the published reference')
    if mode=='reproduce':
        pilot=output(out/'pilot_42')
        pilot_pair_comparison(path(config['pilot_qwen']),path(config['pilot_dipper']),pilot,n,seed)
        checks=compare_reference(pilot,path(config['pilot_reference']),['paired_asr_comparison.csv'])
        if len(checks) and not checks.matches_at_1e_12.all():raise RuntimeError('Pilot numerical reproduction differs from reference')
        result['pilot_reference_checks']=len(checks)
    (out/'evaluation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    return record(out,config,mode,models_executed=False,metrics_computed=True,reference_numeric_checks=result.get('reference_numeric_checks',0))

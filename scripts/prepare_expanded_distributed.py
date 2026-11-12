#!/usr/bin/env python3
"""Build a larger source-disjoint OpAI distributed-corruption panel locally.

The original 392-pair panel is never modified. Selection is based only on
source identity, inherited/new deterministic split and sentence applicability,
not on any semantic score. One editor is fixed per source before screening.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys

for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(variable,'3')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from semantics.positive_controls import load_local_opai_trajectories
from semantics.opai import _balanced_selection, OFFICIAL_DATASET
from semantics.distributed_perturbations import FAMILIES, SENTENCE_COUNTS, sentence_candidates, _ordered_changes, _pair
from semantics.perturbations import load_perturbation_nlp


OPERATIONS = [('style',2,3),('paraphrase',1,2),('polish',0,1)]


def source_split(source, old, seed):
    group='opai:'+source
    if group in old:
        return old[group]
    value=int.from_bytes(hashlib.sha256(f'{seed}|expanded-source-split|{group}'.encode()).digest()[:8],'big')
    return 'calibration' if value%2==0 else 'test'


def positive_pair(candidate, versions, operation, before, after, old_splits, seed):
    key=candidate['key']
    a,b=versions[before],versions[after]
    expected={operation,'style rewrite' if operation=='style' else operation}
    if b['operation'] not in expected:
        raise ValueError('Official operation does not match its version')
    ident=f'{key[0]}|{key[1]}|{key[2]}|{before}|{after}'
    pair_id='opai_expanded:'+hashlib.sha256(ident.encode()).hexdigest()[:24]
    group='opai:'+key[0]
    split=source_split(key[0],old_splits,seed)
    return {'id':pair_id,'pair_id':pair_id,'source_id':key[0],'source_group_id':group,
            'domain':key[1],'generator':key[2],'dataset':'opai_distributed_positive',
            'perturbation':'opai_'+operation,'evaluation_split':split,'status':'applied',
            'original':a['text'],'modified':b['text'],'target_original':None,'target_modified':None,
            'metadata':{'dataset':OFFICIAL_DATASET,'comparison':'adjacent_operation',
                        'from_version':f'v{before}','to_version':f'v{after}',
                        'operation':operation,'source_record_id':a['record_id'],'target_record_id':b['record_id'],
                        'source_group_id':group,'evaluation_split':split,
                        'experiment_cohort':'expanded_source_panel',
                        'expected_semantic_class':'preserved','evaluation_label':0,
                        'semantic_label_verified':False,'positive_control_is_presumed_not_gold':True,
                        'text_changed':a['text']!=b['text'],
                        'selection_basis':'one editor fixed per new source, then rule applicability; no scores'}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opai-dir',type=Path,default=ROOT/'data/opai')
    p.add_argument('--old-positive-pairs',type=Path,default=ROOT/'data/semantic_extension/positive_pairs.jsonl')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/semantic_fairness')
    p.add_argument('--manifest',type=Path,default=ROOT/'runs/03_semantics/fairness/expanded_manifest.json')
    p.add_argument('--n-sources-per-family',type=int,default=50)
    p.add_argument('--minimum-screened',type=int,default=800)
    p.add_argument('--max-sources',type=int,default=1200)
    p.add_argument('--seed',type=int,default=20261106)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    paths=[args.data_dir/'expanded_positive_pairs.jsonl',args.data_dir/'expanded_distributed_pairs.jsonl']
    if any(path.exists() for path in paths) and not args.overwrite:
        raise FileExistsError('Expanded input already exists; preserve it or explicitly --overwrite')
    if args.n_sources_per_family%2:
        raise ValueError('Use an even source target for equal calibration/test quotas')
    old_pairs=[json.loads(x) for x in args.old_positive_pairs.read_text().splitlines()]
    old_splits={x['source_group_id']:x['evaluation_split'] for x in old_pairs}
    trajectories,provenance=load_local_opai_trajectories(args.opai_dir)
    profiles=[{'key':key,'domain':key[1],'generator':key[2]} for key,versions in sorted(trajectories.items())
              if {0,1,2,3}.issubset(versions) and 'opai:'+key[0] not in old_splits]
    ordered_profiles=_balanced_selection(profiles,len(profiles),random.Random(f'{args.seed}:editor-choice'))
    unique={}
    for candidate in ordered_profiles:
        unique.setdefault(candidate['key'][0],candidate)
    candidates=_balanced_selection(list(unique.values()),args.max_sources,
                                   random.Random(f'{args.seed}:screening-order'))
    quota=args.n_sources_per_family//2
    pools={family:{split:[] for split in ['calibration','test']} for family in FAMILIES}
    eligible_source={family:set() for family in FAMILIES}
    screening=[]
    nlp=load_perturbation_nlp()
    seen_by_operation={}
    def enough():
        return all(len(pools[f][split])>=quota for f in FAMILIES for split in pools[f])
    for operation,before,after in OPERATIONS:
        if operation!='style' and enough():
            break
        active=[c for c in candidates if operation=='style' or
                any(c['key'][0] not in eligible_source[family] and
                    len(pools[family][source_split(c['key'][0],old_splits,args.seed)])<quota
                    for family in FAMILIES)]
        positives=[positive_pair(c,trajectories[c['key']],operation,before,after,old_splits,args.seed) for c in active]
        seen=0
        for index,(candidate,positive,doc) in enumerate(zip(active,positives,nlp.pipe(
                (x['modified'] for x in positives),batch_size=16)),1):
            options,n_sentences,n_parser=sentence_candidates(doc,positive,args.seed)
            seen+=1
            for family in FAMILIES:
                changes=_ordered_changes(options[family],positive,family,args.seed)
                screening.append({'source_id':positive['source_id'],'source_group_id':positive['source_group_id'],
                                  'operation':operation,'domain':positive['domain'],'generator':positive['generator'],
                                  'evaluation_split':positive['evaluation_split'],'corruption_family':family,
                                  'n_applicable_sentences':len(changes),'n_metric_sentences':n_sentences,
                                  'n_parser_sentences':n_parser})
                if len(changes)>=5 and positive['source_id'] not in eligible_source[family]:
                    eligible_source[family].add(positive['source_id'])
                    pools[family][positive['evaluation_split']].append({
                        'positive':positive,'changes':changes,'n_sentences':n_sentences,
                        'n_parser_sentences':n_parser,'domain':positive['domain'],'generator':positive['generator']})
            if index%50==0 or index==len(active):
                print(json.dumps({'phase':'screening','operation':operation,'processed':index,'available':len(active),
                                  'qualified':{f:{s:len(rows) for s,rows in v.items()} for f,v in pools.items()}}),flush=True)
            if enough() and (operation!='style' or seen>=args.minimum_screened):
                print(json.dumps({'phase':'quotas_available','operation':operation,'processed':index,
                                  'qualified':{f:{s:len(rows) for s,rows in v.items()} for f,v in pools.items()}}),flush=True)
                break
        seen_by_operation[operation]=seen
    new_pairs=[]
    used_positive={}
    family_manifest={}
    for family in FAMILIES:
        chosen=[]
        for split,rows in pools[family].items():
            chosen.extend(_balanced_selection(rows,quota,random.Random(f'{args.seed}:{family}:{split}:selection')))
        for item in chosen:
            positive=item['positive']; used_positive[positive['id']]=positive
            for k in SENTENCE_COUNTS:
                pair=_pair(positive,item['changes'],family,k,item['n_sentences'],item['n_parser_sentences'],True,args.seed)
                pair['metadata'].update(experiment_cohort='expanded_source_panel',panel='expanded_common_5_primary')
                new_pairs.append(pair)
        family_manifest[family]={
            'eligible_sources_by_split':{s:len(rows) for s,rows in pools[family].items()},
            'selected_sources':len(chosen),
            'selected_sources_by_split':dict(Counter(x['positive']['evaluation_split'] for x in chosen)),
            'selected_sources_by_domain':dict(Counter(x['domain'] for x in chosen)),
            'selected_sources_by_generator':dict(Counter(x['generator'] for x in chosen)),
            'selected_operations':dict(Counter(x['positive']['metadata']['operation'] for x in chosen)),
            'selected_positive_ids':[x['positive']['id'] for x in chosen],
            'subtypes_by_k':{str(k):dict(Counter(subtype for pair in new_pairs if pair['corruption_family']==family
                                    and pair['n_corrupted_sentences']==k
                                    for subtype,count in pair['metadata']['corruption_subtypes'].items()
                                    for _ in range(count))) for k in SENTENCE_COUNTS}}
    new_positive=list(used_positive.values())
    assert len(new_positive)<=2*args.n_sources_per_family
    assert len(new_pairs)<=2*args.n_sources_per_family*len(SENTENCE_COUNTS)
    assert not ({x['source_group_id'] for x in new_positive}&set(old_splits))
    assert len({p['id'] for p in new_pairs+new_positive})==len(new_pairs+new_positive)
    args.data_dir.mkdir(parents=True,exist_ok=True); args.manifest.parent.mkdir(parents=True,exist_ok=True)
    for path,rows in zip(paths,[new_positive,new_pairs]):
        path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    manifest={'experiment_cohort':'expanded_source_panel','seed':args.seed,
              'candidate_unique_new_sources':len(candidates),'old_source_groups_excluded':len(old_splits),
              'screened_sources_by_operation':seen_by_operation,'requested_sources_per_family':args.n_sources_per_family,
              'n_positive_pairs':len(new_positive),'n_positive_source_groups':len({p['source_group_id'] for p in new_positive}),
              'n_distributed_negative_pairs':len(new_pairs),'families':family_manifest,
              'source_metadata':provenance,'screening':screening,
              'source_split_policy':'Exclude old99; hash(seed, original source) fixes split; select equal source quotas by split before model scoring.',
              'editor_policy':'one editor chosen per source from seeded domain-editor round-robin before applicability screening',
              'operation_priority':[x[0] for x in OPERATIONS],
              'labels_are_operation_and_rule_proxy_not_human_gold':True,
              'selection_independent_of_semantic_scores':True,
              'common_panel':True,'requested_sentence_counts':list(SENTENCE_COUNTS),
              'all_requested_source_quotas_met':all(len(v['selected_positive_ids'])==args.n_sources_per_family for v in family_manifest.values()),
              'files':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'original_panel_392_untouched':True,
              'composition_limit':'Subtle stress cases may predominantly be quantitative; not representative of real-humanizer errors.'}
    args.manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['screening','source_metadata','families']},indent=2),flush=True)
    print(json.dumps(family_manifest,indent=2),flush=True)


if __name__=='__main__':
    main()

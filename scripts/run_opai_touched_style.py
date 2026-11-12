#!/usr/bin/env python3
"""Localized OpAI features/distances from validated native sentence arrays."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'2')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.cache'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from semantics.touched_style import (DISTANCE_FEATURE_CANDIDATES, SHORTLIST, digest,
    distance_phi, pos_js, source_bootstrap, validate_native_lineage)
from stylometry.features import FEATURE_SCHEMA, extract_features, load_nlp
from stylometry.report import markdown_table
from scripts.run_opai_style import OPERATIONS, LABELS, summarize


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=True)+'\n')


def load_rows(pairs, opai_dir):
    keys={(p['source_id'],p['domain'],p['generator']) for p in pairs}
    rows={}
    schema_audit=[]
    for path in sorted((opai_dir/'official_parquet').glob('*.parquet')):
        parquet=pq.ParquetFile(path)
        schema_audit.append({'file':path.name,'columns':parquet.schema_arrow.names})
        cols=['document_hash_id','domain','generator','version_index','sentences','sentence_labels','text']
        for batch in parquet.iter_batches(columns=cols,batch_size=256):
            for row in batch.to_pylist():
                key=(row['document_hash_id'],row['domain'],row['generator'])
                version=row['version_index']
                if key not in keys or version>5:
                    continue
                for column in ['sentences','sentence_labels']:
                    row[column]=json.loads(row[column])
                versions=rows.setdefault(key,{})
                if version in versions and versions[version]!=row:
                    raise ValueError('Ambiguous duplicate official record')
                versions[version]=row
    if set(rows)!=keys:
        raise ValueError('A selected trajectory is absent from official Parquet')
    return rows,schema_audit


def compare_full_document_scope(frame, full_path, output, seed, n_bootstrap):
    """Compare full/touched feature deltas on exactly the same valid pairs."""
    full=pd.read_csv(full_path)
    selected_features=SHORTLIST+['word_length_mean']
    metadata=['pair_id','source_id','domain','generator','operation']
    columns=metadata+[prefix+feature for feature in selected_features for prefix in ['before__','after__','delta__']]
    rows=[]
    for scope,scope_frame in frame.groupby('scope'):
        joined=scope_frame.merge(full[columns],on=metadata,suffixes=('_target','_full'),validate='one_to_one')
        if len(joined)!=len(scope_frame):
            raise ValueError('Full-document features do not cover exactly the localized pairs')
        for split in ['all','test']:
            selected=joined if split=='all' else joined[joined.evaluation_split==split]
            for operation,part in selected.groupby('operation'):
                for feature in selected_features:
                    check=[prefix+feature+suffix for prefix in ['before__','after__'] for suffix in ['_target','_full']]
                    valid=part[np.isfinite(part[check].to_numpy()).all(axis=1)]
                    source=valid.groupby('source_id')[[f'delta__{feature}_target',f'delta__{feature}_full']].mean()
                    contrast=source[f'delta__{feature}_target']-source[f'delta__{feature}_full']
                    condition_rng=np.random.default_rng(int(digest(f'{seed}|scope-comparison|{split}|{operation}|{feature}')[:16],16))
                    low,high=source_bootstrap(contrast,condition_rng,n_bootstrap)
                    rows.append({'scope':scope,'split':split,'operation':operation,'feature':feature,
                        'n_pairs_common':len(valid),'n_sources_common':len(source),
                        'mean_delta_target':source[f'delta__{feature}_target'].mean(),
                        'mean_delta_full_same_pairs':source[f'delta__{feature}_full'].mean(),
                        'mean_paired_target_minus_full':contrast.mean(),
                        'contrast_ci_low':low,'contrast_ci_high':high})
    comparison=pd.DataFrame(rows)
    comparison.to_csv(output/'matched_scope_comparison.csv',index=False)
    return comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs',type=Path,default=ROOT/'data/opai_style/pairs.jsonl')
    parser.add_argument('--opai-dir',type=Path,default=ROOT/'data/opai')
    parser.add_argument('--split-pairs',type=Path,default=ROOT/'inputs/semantic_extension/positive_pairs.jsonl',
                        help='Frozen positive pairs supplying the historical calibration/test split')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/opai_touched_style')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/04_opai_style/touched')
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--reuse-features',action='store_true')
    args=parser.parse_args()
    start=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=True)
    args.data_dir.mkdir(parents=True,exist_ok=True)
    pairs=[json.loads(line) for line in args.pairs.read_text().splitlines()]
    rows,schema_audit=load_rows(pairs,args.opai_dir)
    split_path=args.split_pairs
    known={p['source_id']:p['evaluation_split'] for p in map(json.loads,split_path.read_text().splitlines())}
    splits={key[0]:known.get(key[0], 'calibration' if int(digest(f'{args.seed}:{key[0]}')[:8],16)%2==0 else 'test')
            for key in rows}
    valid_rows={}; rejected=[]
    for key,versions in rows.items():
        try:
            validate_native_lineage(versions)
            valid_rows[key]=versions
        except ValueError as error:
            rejected.append({'source_id':key[0],'domain':key[1],'generator':key[2],'reason':str(error)})
    texts={}; units=[]; passages=[]; reference=[]; alignments=[]
    def register(text):
        h=digest(text);texts[h]=text;return h
    for key,versions in valid_rows.items():
        source,domain,generator=key
        if splits[source]=='calibration':
            for sentence in versions[0]['sentences']:
                reference.append({'source_id':source,'hash':register(sentence)})
        for before,after,operation in OPERATIONS:
            left,right=versions[before],versions[after]
            pair_id=digest('|'.join([*key,str(before),str(after)]))[:24]
            base={'pair_id':pair_id,'source_id':source,'domain':domain,'generator':generator,
                  'operation':operation,'evaluation_split':splits[source],
                  'n_document_sentences':len(left['sentences'])}
            targeted=[i for i,label in enumerate(right['sentence_labels']) if label]
            changed=[i for i in targeted if left['sentences'][i]!=right['sentences'][i]]
            for i in targeted:
                units.append({**base,'sentence_id':f's{i+1:04d}',
                              'before_hash':register(left['sentences'][i]),
                              'after_hash':register(right['sentences'][i]),
                              'actually_changed':i in changed})
            for scope,positions in [('native_targeted',targeted),('actually_changed',changed)]:
                a='\n\n'.join(left['sentences'][i] for i in positions)
                b='\n\n'.join(right['sentences'][i] for i in positions)
                passages.append({**base,'scope':scope,'n_touched_sentences':len(positions),
                                 'n_actually_changed':len(changed),'before_hash':register(a),'after_hash':register(b)})
            alignments.append({**base,'native_sentence_ids':[f's{i+1:04d}' for i in targeted],
                               'actually_changed_ids':[f's{i+1:04d}' for i in changed],
                               'targeted_before':[left['sentences'][i] for i in targeted],
                               'targeted_after':[right['sentences'][i] for i in targeted]})
    (args.data_dir/'native_aligned_passages.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in alignments))
    cache_path=args.data_dir/'feature_cache.json'
    cache=json.loads(cache_path.read_text()) if args.reuse_features and cache_path.exists() else {}
    old_cache_path=ROOT/'data/opai_style/feature_cache.json'
    if old_cache_path.exists():
        old_cache=json.loads(old_cache_path.read_text())
        for h in texts:
            if h in old_cache and h not in cache: cache[h]=old_cache[h]
    missing=[(h,text) for h,text in texts.items() if h not in cache]
    print(f'{len(valid_rows)} trajectories; {len(units)} targeted sentence pairs; {len(missing)} new unique texts',flush=True)
    nlp=load_nlp()
    for i,((h,text),doc) in enumerate(zip(missing,nlp.pipe((text for _,text in missing),batch_size=64)),1):
        cache[h]=extract_features(doc,text)
        if i%500==0:
            print(f'Extracted {i}/{len(missing)}',flush=True)
            cache_path.write_text(json.dumps(cache,allow_nan=True))
    cache_path.write_text(json.dumps(cache,allow_nan=True))
    # Calibration consists solely of v0 native sentences, source-weighted.
    ref=pd.DataFrame([{**r,**cache[r['hash']]} for r in reference]).drop_duplicates(['source_id','hash'])
    ref=ref[ref.n_words>=5]
    weights=1/ref.groupby('source_id').source_id.transform('size').to_numpy()
    values=ref[DISTANCE_FEATURE_CANDIDATES].to_numpy(dtype=float)
    if not np.isfinite(values).all(): raise ValueError('Nonfinite fixed feature in v0 reference')
    mean=np.average(values,axis=0,weights=weights)
    scale=np.sqrt(np.average((values-mean)**2,axis=0,weights=weights))
    keep=scale>1e-8
    names=np.array(DISTANCE_FEATURE_CANDIDATES)[keep].tolist(); scale=scale[keep]
    standardizer={'reference':'v0 native sentences, >=5 lexical words; calibration sources only; equal total weight per source',
                  'n_reference_sentences':len(ref),'n_reference_sources':ref.source_id.nunique(),
                  'features':names,'scale':dict(zip(names,scale.tolist())),
                  'dropped_constant_features':np.array(DISTANCE_FEATURE_CANDIDATES)[~keep].tolist(),
                  'excluded_by_design':['global counts','TTR/hapax','MATTR50','sentence SD/CV','Wikipedia phrase proxies'],
                  'primary':'RMS across fixed standardized feature deltas and targeted aligned sentence pairs',
                  'alternative':'sqrt Jensen-Shannon divergence of pooled touched UPOS distributions, log base 2'}
    standardizer['coverage_diagnostic']='sqrt(n_valid_touched_sentences / n_document_sentences) times distance_rms_sentences; same frozen feature scales; reports spatial coverage separately from intensity per touched sentence'
    write_json(args.output/'standardizer.json',standardizer)
    unit_rows=[]
    for unit in units:
        a,b=cache[unit['before_hash']],cache[unit['after_hash']]
        good=a['n_words']>=5 and b['n_words']>=5
        unit_rows.append({**unit,'before_words':a['n_words'],'after_words':b['n_words'],
                          'unit_distance_rms':distance_phi(a,b,names,scale) if good else np.nan,
                          'unit_pos_js':pos_js(a,b) if good else np.nan,
                          'eligible_for_distance':good})
    unit_frame=pd.DataFrame(unit_rows)
    unit_frame.to_csv(args.output/'aligned_sentence_distances.csv',index=False)
    passage_rows=[]
    for passage in passages:
        a,b=cache[passage['before_hash']],cache[passage['after_hash']]
        u=unit_frame[unit_frame.pair_id==passage['pair_id']]
        if passage['scope']=='actually_changed': u=u[u.actually_changed]
        distance=u.unit_distance_rms.dropna().to_numpy()
        zero=passage['n_touched_sentences']==0
        item={**passage,'n_distance_sentences':len(distance),
              'distance_rms_sentences':0.0 if zero else float(np.sqrt(np.mean(distance**2))) if len(distance) else np.nan,
              'distance_rms_pooled':0.0 if zero else distance_phi(a,b,names,scale),
              'distance_pos_js_pooled':0.0 if zero else pos_js(a,b)}
        item['distance_rms_coverage']=item['distance_rms_sentences']*np.sqrt(len(distance)/passage['n_document_sentences'])
        for name in FEATURE_SCHEMA:
            item['before__'+name]=a[name];item['after__'+name]=b[name];item['delta__'+name]=b[name]-a[name]
        passage_rows.append(item)
    frame=pd.DataFrame(passage_rows)
    frame.to_csv(args.output/'touched_feature_values.csv',index=False)
    matched_scope=compare_full_document_scope(frame,ROOT/'results/04_opai_style/document/paired_feature_values.csv',
        args.output,args.seed,args.bootstrap)
    rng=np.random.default_rng(args.seed)
    feature_tables=[]
    for scope,part in frame.groupby('scope'):
        # Empty unchanged scopes do not manufacture zero rates/features.
        feature_tables.append(summarize(part[part.n_touched_sentences>0],list(FEATURE_SCHEMA),rng,args.bootstrap,scope))
    features=pd.concat(feature_tables,ignore_index=True)
    features.to_csv(args.output/'feature_summary.csv',index=False)
    metrics=['distance_rms_sentences','distance_rms_pooled','distance_pos_js_pooled','distance_rms_coverage']
    summary=[]
    for scope,scope_part in frame.groupby('scope'):
        for split in ['all','calibration','test']:
            part=scope_part if split=='all' else scope_part[scope_part.evaluation_split==split]
            for operation,op in part.groupby('operation'):
                for metric in metrics:
                    values=op.groupby('source_id')[metric].mean().dropna()
                    condition_rng=np.random.default_rng(int(digest(f'{args.seed}|distance|{split}|{operation}|{metric}')[:16],16))
                    low,high=source_bootstrap(values,condition_rng,args.bootstrap)
                    summary.append({'scope':scope,'split':split,'operation':operation,'metric':metric,
                                    'n_pairs':op[metric].notna().sum(),'n_sources':len(values),
                                    'mean':values.mean(),'median_source':values.median(),'ci_low':low,'ci_high':high,
                                    'mean_touched_sentences':op.n_touched_sentences.mean(),
                                    'mean_before_words':op['before__n_words'].mean(),
                                    'mean_after_words':op['after__n_words'].mean()})
    summary=pd.DataFrame(summary);summary.to_csv(args.output/'distance_summary.csv',index=False)
    orders=[];triples=[]
    for scope,scope_part in frame.groupby('scope'):
        for metric in metrics:
            pivot=scope_part.pivot(index=['source_id','domain','generator','evaluation_split'],columns='operation',values=metric)
            tri=pivot[['polish','paraphrase','style']].dropna().reset_index()
            tri['strict_order']=(tri.polish<tri.paraphrase)&(tri.paraphrase<tri['style'])
            tri['nonstrict_order']=(tri.polish<=tri.paraphrase)&(tri.paraphrase<=tri['style'])
            tri['para_minus_polish']=tri.paraphrase-tri.polish
            tri['style_minus_para']=tri['style']-tri.paraphrase
            tri['scope']=scope;tri['metric']=metric;triples.append(tri)
            for split in ['all','calibration','test']:
                part=tri if split=='all' else tri[tri.evaluation_split==split]
                for column in ['strict_order','nonstrict_order','para_minus_polish','style_minus_para']:
                    source=part.groupby('source_id')[column].mean()
                    condition_rng=np.random.default_rng(int(digest(f'{args.seed}|order|{split}|{metric}|{column}')[:16],16))
                    low,high=source_bootstrap(source,condition_rng,args.bootstrap)
                    orders.append({'scope':scope,'metric':metric,'split':split,'quantity':column,
                                   'n_trajectories':len(part),'n_sources':len(source),'mean':source.mean(),
                                   'ci_low':low,'ci_high':high})
    order=pd.DataFrame(orders);order.to_csv(args.output/'ordinal_summary.csv',index=False)
    pd.concat(triples,ignore_index=True).to_csv(args.output/'trajectory_orders.csv',index=False)
    write_json(args.output/'manifest.json',{'n_selected_trajectories':len(rows),'n_certified_lineages':len(valid_rows),
        'n_sources':len({k[0] for k in valid_rows}),'n_pairs':len(pairs),'n_targeted_sentence_pairs':len(units),
        'n_primary_sentence_pairs_eligible':int(unit_frame.eligible_for_distance.sum()),
        'n_primary_sentence_pairs_excluded_lt5_words':int((~unit_frame.eligible_for_distance).sum()),
        'n_unique_feature_texts':len(texts),'rejected_trajectories':rejected,'schema_audit':schema_audit,
        'explicit_sentence_ids_in_release':False,
        'lineage':'Position in native ordered sentences array; IDs reconstructed s0001… under validated builder invariants, not fuzzy alignment',
        'checks':['all v0..v5 counts equal','text reconstructed by native arrays','label0 anchors equal v0','no changes outside cumulative native labels','monotone labels'],
        'targeted_definition':'native label_after=1; builder rewrites cumulative touched set at each stage, not only newly selected positions',
        'actually_changed_definition':'native targeted position with unequal before/after strings',
        'scalar_definitions':standardizer,'hypothesis':'Polish < Paraphrase < Style Rewrite; tested, never imposed',
        'source_splits':{split:sum(1 for s in splits if splits[s]==split) for split in ['calibration','test']},
        'known_semantic_source_splits_preserved':len(known),'elapsed_seconds':time.monotonic()-start,
        'code_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['scripts/run_opai_touched_style.py','src/semantics/touched_style.py']}})
    report(frame,features,summary,order,args.output,matched_scope)
    print(json.dumps({'status':'executed','pairs':len(pairs),'certified_trajectories':len(valid_rows),
                      'elapsed_seconds':time.monotonic()-start},indent=2),flush=True)


def report(frame,features,summary,order,output,matched_scope):
    selected=summary[(summary.scope=='native_targeted')&(summary.split=='test')]
    shortlist=features[(features.analysis=='native_targeted')&features.feature.isin(SHORTLIST)]
    ordinals=order[(order.scope=='native_targeted')&(order.split=='test')]
    count_frame=frame.assign(mattr_pair_defined=frame[['before__mattr_50','after__mattr_50']].notna().all(axis=1))
    counts=count_frame.groupby(['scope','operation']).agg(n_pairs=('pair_id','size'),
        mean_touched_sentences=('n_touched_sentences','mean'),mean_before_words=('before__n_words','mean'),
        mean_distance_sentences=('n_distance_sentences','mean'),n_mattr_pair_defined=('mattr_pair_defined','sum')).reset_index()
    primary=selected[selected.metric=='distance_rms_sentences'].set_index('operation')
    coverage=selected[selected.metric=='distance_rms_coverage'].set_index('operation')
    p,r,s=[primary.loc[name,'mean'] for name in ['polish','paraphrase','style']]
    cp,cr,cs=[coverage.loc[name,'mean'] for name in ['polish','paraphrase','style']]
    lines=['# Stile sulle sole frasi toccate di OpAI', '',
        f'Nel test la distanza RMS per frase è Polish {p:.3f}, Paraphrase {r:.3f}, Style {s:.3f}: '
        'Polish è più leggero, ma l’ordine completo richiesto non è confermato dalla metrica primaria. '
        f'Quando si include la copertura degli edit, le medie diventano {cp:.3f}, {cr:.3f}, {cs:.3f}. '
        'Questo distingue intensità locale e quantità del documento riscritta; non si scartano le metriche che non restituiscono l’ordine atteso.', '',
        'Sono usate le stesse 200 traiettorie dello studio documentale. Il rilascio Parquet non include la colonna `sentence_ids`, '
        'ma conserva `sentences` e `sentence_labels`. Il costruttore ufficiale mantiene un array ordinato per gli ID s0001… e impone una frase per ID. '
        'Su tutte le traiettorie incluse sono state verificate cardinalità costante v0..v5, ricostruzione del testo, '
        'ancore non modificate identiche al v0 e assenza di cambi fuori dalle posizioni marcate. Gli ID sono dunque **ricostruiti dalla lineage posizionale nativa verificata**, '
        'non presenti come campo nel Parquet né ottenuti con matching approssimato.', '',
        'Le etichette native sono cumulative: a ogni stadio il builder riscrive tutte le frasi già selezionate più le nuove. '
        'L’analisi primaria native_targeted include tali posizioni; actually_changed include solo quelle effettivamente diverse fra le versioni adiacenti. '
        'Questo distingue il budget di editing dal numero di edit riusciti.', '',markdown_table(counts),'',
        '## Distanze scalari predefinite', '',
        'La metrica primaria è la radice della media dei quadrati di Δφ/σ sulle singole frasi allineate e sulle dimensioni fisse, '
        'poi aggregata sulle sole frasi toccate. Non viene divisa per il numero di frasi dell’intero documento. '
        'Le σ sono apprese soltanto dalle frasi native v0 delle fonti di calibrazione, con uguale peso totale per fonte; '
        'sono esclusi conteggi globali, TTR/hapax, MATTR50, SD/CV di frase e proxy retorici Wikipedia. '
        'Le feature costanti nella calibrazione sono eliminate una sola volta per tutte le operazioni. '
        'Frasi con meno di cinque parole lessicali non entrano nella distanza primaria e vengono contate.', '',
        'Si riportano anche la norma sul passaggio concatenato e la distanza Jensen–Shannon delle distribuzioni POS del passaggio (base 2, intervallo 0–1). '
        'Tutte le 105 feature, inclusa la shortlist, sono comunque estratte sui passaggi toccati concatenati. '
        'MATTR resta NaN sotto 50 parole: non viene riempito con zero o cambiando finestra.', '',
        'La diagnostica distance_rms_coverage moltiplica la RMS delle frasi toccate per sqrt(n_frasi_toccate_valide/N_frasi_documento). '
        'Equivale a distribuire l’energia quadratica degli edit sull’intero documento, assegnando delta zero alle frasi intatte. '
        'Separa così la copertura spaziale dall’intensità per frase; non sostituisce la metrica primaria e non ritocca le scale. '
        'Il numero di frasi toccate cresce per costruzione tra le operazioni, quindi un ordine di questa diagnostica può riflettere anche il budget.', '',
        'Per l’energia pesata per copertura i due scope coincidono: i target invariati hanno delta zero. '
        'I bootstrap impiegano gli stessi ricampionamenti tra scope per ogni metrica/operazione, evitando differenze Monte Carlo artificiose quando gli score sono identici.', '',
        'L’ordine Polish < Paraphrase < Style Rewrite è un’ipotesi di coerenza, non una gold label e non viene imposto scegliendo pesi dopo i risultati. '
        'Le operazioni possono modificare dimensioni differenti; uno stile più formale non comporta necessariamente maggiore distanza POS.', '',
        '## Distanze nel test separato', '',markdown_table(selected[['operation','metric','n_pairs','n_sources','mean','ci_low','ci_high']]),'',
        'IC 95% bootstrap per documento sorgente, dopo la media degli editor. La scala è appresa sulle sole fonti di calibrazione; '
        'i risultati test non partecipano al suo apprendimento.', '',
        '## Ordine nelle stesse traiettorie', '',markdown_table(ordinals[['metric','quantity','n_trajectories','n_sources','mean','ci_low','ci_high']]),'',
        'La frazione strict_order conta Polish < Paraphrase < Style entro la stessa traiettoria; nonstrict_order ammette pareggi. '
        'I due delta misurano Paraphrase−Polish e Style−Paraphrase. Questi risultati distinguono ordine delle medie e ordine nei singoli esempi.', '',
        '## Shortlist sui passaggi toccati', '',markdown_table(shortlist[['operation','feature','n_pairs','n_sources','mean_delta','delta_ci_low','delta_ci_high','q_bh_all_features_operations']]),'',
        'Gli effetti sulle feature sono esplorativi e usano le fonti come unità; BH corregge le 105×5 ipotesi entro scope. '
        'Le parti toccate hanno lunghezze e contenuti diversi: questo disegno rimuove la diluizione nel documento intero, '
        'non trasforma la variazione in una prova di naturalezza.', '',
        '## Documento intero e passaggi sugli stessi esempi validi', '',
        'Il confronto seguente usa esattamente gli stessi pair_id ed editor per le due misure, con entrambe le versioni '
        'finite sia nel documento intero sia nel passaggio toccato. È essenziale per MATTR50: non si confronta il sottocampione '
        'di passaggi con almeno 50 parole con tutti i documenti. Si mediano prima gli editor della stessa fonte e si ricampiona '
        'il contrasto appaiato Δtarget−Δfull. Le differenze di scala e di passaggio osservato rimangono: un contrasto è descrittivo, '
        'non una stima causale della sola diluizione.', '',
        markdown_table(matched_scope[(matched_scope.scope=='native_targeted')&(matched_scope.split=='all')&
            matched_scope.feature.isin(SHORTLIST)][['operation','feature','n_pairs_common','n_sources_common',
            'mean_delta_target','mean_delta_full_same_pairs','mean_paired_target_minus_full','contrast_ci_low','contrast_ci_high']]), '',
        'File: `distance_summary.csv`, `ordinal_summary.csv`, `trajectory_orders.csv`, `touched_feature_values.csv`, '
        '`aligned_sentence_distances.csv`, `feature_summary.csv`, `matched_scope_comparison.csv`, `standardizer.json`, `manifest.json`. '
        'I testi restano in `data/opai_touched_style/`. Riproduzione: `.venv/bin/python scripts/run_opai_touched_style.py`; '
        'aggiungere `--reuse-features` per usare le feature già estratte.', '']
    case_path=output/'qualitative_cases.md'
    if case_path.exists():
        lines.extend(['## Lettura diagnostica di cinque inversioni', '',case_path.read_text(), ''])
    (output/'report.md').write_text('\n'.join(lines).rstrip()+'\n')
    for metric in ['distance_rms_sentences','distance_pos_js_pooled','distance_rms_coverage']:
        fig,ax=plt.subplots(figsize=(7,4))
        for scope,color,offset in [('native_targeted','#167c80',-.06),('actually_changed','#d78c3c',.06)]:
            part=summary[(summary.scope==scope)&(summary.split=='test')&(summary.metric==metric)].set_index('operation').reindex(['polish','paraphrase','style'])
            x=np.arange(3)+offset
            ax.errorbar(x,part['mean'],yerr=[np.maximum(part['mean']-part.ci_low,0),np.maximum(part.ci_high-part['mean'],0)],
                        fmt='o-',capsize=4,label=scope,color=color)
        ax.set_xticks(range(3),['Polish','Paraphrase','Style Rewrite'])
        ax.set_ylabel(metric)
        ax.set_title('Sole frasi toccate: test separato, IC 95% per fonte')
        ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(output/(metric+'.png'),dpi=170);fig.savefig(output/(metric+'.pdf'));plt.close(fig)


if __name__=='__main__':
    main()

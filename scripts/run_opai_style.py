#!/usr/bin/env python3
"""Paired stylometry across five adjacent, official OpAI editing stages."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from semantics.opai import _balanced_selection
from stylometry.analysis import _bh
from stylometry.features import FEATURE_GROUPS, FEATURE_SCHEMA, WIKIPEDIA_FEATURES, extract_features, load_nlp
from stylometry.report import markdown_table

OPERATIONS = [(0, 1, 'polish'), (1, 2, 'paraphrase'), (2, 3, 'style'),
              (3, 4, 'compress'), (4, 5, 'expand')]
SHORTLIST = ['pos_num', 'pos_cconj', 'mattr_50', 'pos_propn']
LABELS = {'polish': 'Polish', 'paraphrase': 'Paraphrase', 'style': 'Style Rewrite',
          'compress': 'Compress', 'expand': 'Expand'}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prefixes(doc, minimum=90, maximum=180):
    """Unchanged opening prefixes, ending at a parsed punctuation boundary."""
    found, n_words = {}, 0
    for sent in doc.sents:
        n_words += sum(token.is_alpha for token in sent)
        if n_words > maximum:
            break
        if n_words < minimum:
            continue
        prefix = doc.text[:sent.end_char].rstrip()
        if re.search(r'''[.!?…][\s"'’”\)\]]*$''', prefix):
            found.setdefault(n_words, prefix)
    return found


def summarize(frame, feature_names, rng, n_bootstrap, analysis):
    rows = []
    for _, _, operation in OPERATIONS:
        part = frame[frame.operation == operation]
        if not len(part):
            continue
        # Editors of the same human original are repeated measurements: average
        # their paired changes before inference so each source has equal weight.
        before = part[[f'before__{f}' for f in feature_names]].copy()
        after = part[[f'after__{f}' for f in feature_names]].copy()
        before.columns = after.columns = feature_names
        # Keep the same editors on both sides for every feature (e.g. MATTR
        # can be undefined in a short edited version).
        complete = np.isfinite(before.to_numpy()) & np.isfinite(after.to_numpy())
        before = before.where(complete).groupby(part.source_id).mean()
        after = after.where(complete).groupby(part.source_id).mean()
        delta = after - before
        groups = len(delta)
        weights = rng.multinomial(groups, np.full(groups, 1/groups), size=n_bootstrap)
        values = delta.to_numpy()
        valid = np.isfinite(values)
        finite_values = np.where(valid, values, 0)
        draws_n = weights @ valid.astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            draws_mean = (weights @ finite_values) / draws_n
            draws_var = np.maximum(((weights @ (finite_values**2)) - draws_n*draws_mean**2)/(draws_n-1), 0)
            draws_g = (1-3/(4*draws_n-5))*draws_mean/np.sqrt(draws_var)
        for j, feature in enumerate(feature_names):
            d = delta[feature].dropna()
            n = len(d)
            sd = d.std(ddof=1)
            if n < 2:
                g, pvalue = np.nan, np.nan
            elif sd == 0:
                g = 0.0 if d.mean() == 0 else np.nan
                pvalue = 1.0 if d.mean() == 0 else 0.0
            else:
                g = (1-3/(4*n-5))*d.mean()/sd
                pvalue = stats.ttest_1samp(d, 0).pvalue
            good_mean = draws_mean[:, j][np.isfinite(draws_mean[:, j])]
            low, high = np.quantile(good_mean, [.025,.975]) if len(good_mean) else (np.nan,np.nan)
            good_g = draws_g[:, j][np.isfinite(draws_g[:, j])]
            if len(d) and d.eq(0).all():
                glow = ghigh = 0.0
            elif len(good_g) >= n_bootstrap/2:
                glow, ghigh = np.quantile(good_g, [.025,.975])
            else:
                glow = ghigh = np.nan
            before_valid, after_valid = before.loc[d.index,feature], after.loc[d.index,feature]
            rows.append({'analysis':analysis,'operation':operation,'feature':feature,'family':FEATURE_GROUPS[feature],
                         'n_pairs':int(part[[f'before__{feature}',f'after__{feature}']].notna().all(axis=1).sum()),
                         'n_sources':n,'before_mean':before_valid.mean(),'after_mean':after_valid.mean(),
                         'mean_delta':d.mean(),'median_source_delta':d.median(),'delta_sd_between_sources':sd,
                         'delta_ci_low':low,'delta_ci_high':high,'paired_hedges_gz':g,
                         'paired_gz_ci_low':glow,'paired_gz_ci_high':ghigh,'paired_t_pvalue':pvalue,
                         'fraction_source_delta_positive':(d>0).mean() if n else np.nan,
                         'fraction_source_delta_zero':(d==0).mean() if n else np.nan})
    result = pd.DataFrame(rows)
    if len(result):
        # One predeclared family for all feature × operation tests in an analysis.
        result['q_bh_all_features_operations'] = _bh(result.paired_t_pvalue.to_numpy())
    return result


def paired_frame(pairs, features, analysis):
    rows=[]
    for pair in pairs:
        a,b=features[pair['before_hash']],features[pair['after_hash']]
        item={k:v for k,v in pair.items() if k not in {'before_text','after_text'}}
        item['analysis']=analysis
        for feature in FEATURE_SCHEMA:
            item[f'before__{feature}']=a[feature]
            item[f'after__{feature}']=b[feature]
            item[f'delta__{feature}']=b[feature]-a[feature]
        rows.append(item)
    return pd.DataFrame(rows)


def heatmap(table, names, title, output):
    pivot=table.pivot(index='feature',columns='operation',values='paired_hedges_gz')
    pivot=pivot.reindex(index=names,columns=[x[2] for x in OPERATIONS])
    fig,ax=plt.subplots(figsize=(7.5,max(3.3,len(names)*.37)))
    im=ax.imshow(pivot, cmap='RdBu_r',vmin=-2,vmax=2,aspect='auto')
    ax.set_xticks(range(5),[LABELS[x] for x in pivot.columns],rotation=15,ha='right')
    ax.set_yticks(range(len(names)),names,fontsize=9)
    for i in range(len(names)):
        for j in range(5):
            val=pivot.iloc[i,j]
            if np.isfinite(val):
                ax.text(j,i,f'{val:+.2f}',ha='center',va='center',fontsize=9,color='white' if abs(val)>1.1 else 'black')
    ax.set_title(title)
    fig.colorbar(im,ax=ax,label='Hedges g_z appaiato, dopo − prima\n(colore limitato a ±2)')
    fig.tight_layout()
    fig.savefig(str(output)+'.png',dpi=180)
    fig.savefig(str(output)+'.pdf')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opai-dir',type=Path,default=ROOT/'data/opai')
    p.add_argument('--output',type=Path,default=ROOT/'runs/04_opai_style/document')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/opai_style')
    p.add_argument('--n',type=int,default=200)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--reuse-features',action='store_true')
    args=p.parse_args()
    start=time.monotonic()
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    args.data_dir.mkdir(parents=True,exist_ok=True)
    from semantics.positive_controls import load_local_opai_trajectories
    trajectories,provenance=load_local_opai_trajectories(args.opai_dir,max_version=5)
    candidates=[{'key':key,'domain':key[1],'generator':key[2]} for key,versions in sorted(trajectories.items()) if set(range(6)).issubset(versions)]
    selected=_balanced_selection(candidates,args.n,random.Random(args.seed))
    if len(selected)!=args.n:
        raise ValueError(f'Only {len(selected)} complete trajectories; requested {args.n}')
    texts,pairs={},[]
    for candidate in selected:
        key=candidate['key']
        versions=trajectories[key]
        for before,after,operation in OPERATIONS:
            if versions[after]['operation'] not in {operation,'style rewrite' if operation=='style' else operation}:
                raise ValueError(f'Unexpected version operation: {versions[after]["operation"]}')
            a,b=[versions[v]['text'] for v in [before,after]]
            ha,hb=digest(a),digest(b)
            texts[ha],texts[hb]=a,b
            pairs.append({'pair_id':digest('|'.join([*key,str(before),str(after)]))[:24],
                          'source_id':key[0],'domain':key[1],'generator':key[2],
                          'operation':operation,'from_version':before,'to_version':after,
                          'before_hash':ha,'after_hash':hb,'text_changed':a!=b})
    selected_records=[{'source_id':x['key'][0],'domain':x['domain'],'generator':x['generator']} for x in selected]
    manifest={'official_source':provenance,'seed':args.seed,'n_selected_trajectories':len(selected),
              'n_unique_sources':len({x['source_id'] for x in selected_records}),
              'candidate_complete_trajectories':len(candidates),
              'n_pairs':len(pairs),'n_unique_full_texts':len(texts),
              'selected_trajectories':selected_records,
              'stratum_counts':dict(Counter(x['domain']+'|'+x['generator'] for x in selected)),
              'operations':OPERATIONS,'selection':'seeded round-robin domain × editor, same trajectories in all stages',
              'source_unit':'document_hash_id; editors averaged before inference',
              'full_document_primary':True,'prefix_control':'same source and stage, exact lexical counts 90–180, closest to 150, parsed sentence ends',
              'versions':{k:importlib.metadata.version(k) for k in ['spacy','en-core-web-sm','textstat','numpy','pandas','scipy']},
              'extractor_sha256':hashlib.sha256((ROOT/'src/stylometry/features.py').read_bytes()).hexdigest(),
              'feature_schema':FEATURE_SCHEMA}
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Raw benchmark passages remain solely in ignored data/.
    (args.data_dir/'pairs.jsonl').write_text(''.join(json.dumps({**x,'original':texts[x['before_hash']],'modified':texts[x['after_hash']]},ensure_ascii=False)+'\n' for x in pairs))
    cache_path=args.data_dir/'feature_cache.json'
    cache=json.loads(cache_path.read_text()) if args.reuse_features and cache_path.is_file() else {}
    nlp=load_nlp()
    prefix_maps={}
    missing=[(h,t) for h,t in texts.items() if h not in cache]
    print(f'Full documents: {len(texts)} unique; extracting {len(missing)}',flush=True)
    for i,((h,text),doc) in enumerate(zip(missing,nlp.pipe((t for _,t in missing),batch_size=32))):
        cache[h]=extract_features(doc,text)
        prefix_maps[h]=prefixes(doc)
        if (i+1)%100==0:
            print(f'Extracted {i+1}/{len(missing)}',flush=True)
            cache_path.write_text(json.dumps(cache,allow_nan=True))
    # Prefix boundaries must still be parsed if the full feature cache is reused.
    prefix_missing=[(h,t) for h,t in texts.items() if h not in prefix_maps]
    for (h,_),doc in zip(prefix_missing,nlp.pipe((t for _,t in prefix_missing),batch_size=32)):
        prefix_maps[h]=prefixes(doc)
    prefix_pairs,prefix_texts=[],{}
    for pair in pairs:
        a,b=[prefix_maps[pair[k]] for k in ['before_hash','after_hash']]
        shared=set(a).intersection(b)
        if not shared:
            continue
        n=min(shared,key=lambda x:(abs(x-150),x))
        ha,hb=digest(a[n]),digest(b[n])
        prefix_texts[ha],prefix_texts[hb]=a[n],b[n]
        prefix_pairs.append({**pair,'before_hash':ha,'after_hash':hb,'matched_n_words':n,'text_changed':a[n]!=b[n]})
    missing=[(h,t) for h,t in prefix_texts.items() if h not in cache]
    print(f'Prefix control: {len(prefix_pairs)}/{len(pairs)} pairs; {len(missing)} new unique passages',flush=True)
    for i,((h,text),doc) in enumerate(zip(missing,nlp.pipe((t for _,t in missing),batch_size=32))):
        cache[h]=extract_features(doc,text)
        if (i+1)%100==0:
            print(f'Extracted prefixes {i+1}/{len(missing)}',flush=True)
            cache_path.write_text(json.dumps(cache,allow_nan=True))
    cache_path.write_text(json.dumps(cache,allow_nan=True))
    features_frame=pd.DataFrame([{'text_sha256':h,**cache[h]} for h in texts])
    features_frame.to_csv(out/'document_features.csv',index=False)
    full=paired_frame(pairs,cache,'full_document')
    control=paired_frame(prefix_pairs,cache,'exact_prefix_length_matched')
    if len(control):
        assert (control['before__n_words']==control['after__n_words']).all()
        assert (control['before__n_words']==control.matched_n_words).all()
    full.to_csv(out/'paired_feature_values.csv',index=False)
    control.to_csv(out/'prefix_matched_feature_values.csv',index=False)
    all_tables=[]
    rng=np.random.default_rng(args.seed)
    for name,frame in [('full_document',full),('exact_prefix_length_matched',control)]:
        if len(frame):
            table=summarize(frame,list(FEATURE_SCHEMA),rng,args.bootstrap,name)
            all_tables.append(table)
    summary=pd.concat(all_tables,ignore_index=True)
    summary.to_csv(out/'summary.csv',index=False)
    # Domain/editor means remain descriptive; no extra uncorrected significance claims.
    strata=[]
    for scope,column in [('domain','domain'),('editor','generator')]:
        for (operation,name),part in full.groupby(['operation',column]):
            values=part.groupby('source_id')[[f'delta__{f}' for f in FEATURE_SCHEMA]].mean().mean()
            for feature,value in zip(FEATURE_SCHEMA,values):
                strata.append({'scope':scope,'stratum':name,'operation':operation,'feature':feature,'n_pairs':len(part),
                               'n_sources':part.source_id.nunique(),'mean_delta':value})
    pd.DataFrame(strata).to_csv(out/'stratified_mean_deltas.csv',index=False)
    full_table=summary[summary.analysis=='full_document']
    prefix_table=summary[summary.analysis=='exact_prefix_length_matched']
    chosen=SHORTLIST+['sentence_length_mean','sentence_length_cv','word_length_mean']+list(WIKIPEDIA_FEATURES)
    heatmap(full_table,chosen,'OpAI: variazione stilometrica nelle cinque operazioni',out/'style_movement')
    if len(prefix_table):
        heatmap(prefix_table,chosen,'Controllo: prefissi appaiati con uguale numero di parole',out/'style_movement_length_matched')
    length_rows=[]
    for _,_,operation in OPERATIONS:
        part=full[full.operation==operation]
        pref=control[control.operation==operation]
        length_rows.append({'operation':LABELS[operation],'pairs':len(part),'unique_sources':part.source_id.nunique(),
                            'changed':int(part.text_changed.sum()),
                            'before_mean_words':part.groupby('source_id')['before__n_words'].mean().mean(),
                            'after_mean_words':part.groupby('source_id')['after__n_words'].mean().mean(),
                            'median_word_ratio':(part['after__n_words']/part['before__n_words']).median(),
                            'prefix_matched_pairs':len(pref),'prefix_sources':pref.source_id.nunique()})
    length=pd.DataFrame(length_rows)
    length.to_csv(out/'length_changes.csv',index=False)
    compact=full_table[full_table.feature.isin(SHORTLIST)][['operation','feature','before_mean','after_mean','mean_delta','delta_ci_low','delta_ci_high','paired_hedges_gz','q_bh_all_features_operations']]
    top=full_table[~full_table.feature.isin(['n_words','n_characters','n_sentences','ttr','hapax_ratio'])].copy()
    top['abs_g']=top.paired_hedges_gz.abs()
    top=top[top.q_bh_all_features_operations<.05].sort_values('abs_g',ascending=False).groupby('operation',sort=False).head(5)
    top=top.sort_values(['operation','abs_g'],ascending=[True,False])
    highlights=[]
    for operation,feature in [('polish','word_length_mean'),('paraphrase','mattr_50'),
                              ('style','word_length_mean'),('compress','pos_propn'),
                              ('expand','sentence_length_mean')]:
        row=full_table[(full_table.operation==operation)&(full_table.feature==feature)].iloc[0]
        highlights.append(f'{LABELS[operation]}: {feature} Δ={row.mean_delta:+.4f} '
                          f'(IC95% [{row.delta_ci_low:+.4f}; {row.delta_ci_high:+.4f}]), '
                          f'g_z={row.paired_hedges_gz:+.2f}')
    prefix_sig=prefix_table[prefix_table.feature.isin(SHORTLIST)&(prefix_table.q_bh_all_features_operations<.05)]
    prefix_sig_names=', '.join(LABELS[r.operation]+'/'+r.feature for r in prefix_sig.itertuples()) or 'nessuna'
    lines=['# OpAI-Bench: quali feature cambiano fra versioni adiacenti','',
           f'{len(selected)} traiettorie comuni ai cinque stadi, {manifest["n_unique_sources"]} originali distinti, {len(pairs)} coppie e {len(texts)} testi unici. Campionamento bilanciato dominio × editor; seed {args.seed}.',
           '', 'Le operazioni muovono le feature in modi diversi. Alcuni esempi, senza convertirli in giudizi di naturalezza: '
           + '; '.join(highlights)+'.',
           '',markdown_table(length),'',
           'La differenza è sempre **dopo − prima**. Polish confronta v0 umano con v1 modificato dall’AI; gli altri confronti riguardano versioni già modificate. Le versioni sono cumulative: il confronto adiacente isola meglio lo stadio corrente, ma la sua risposta dipende anche dalla storia precedente. Una variazione delle feature non dimostra maggiore naturalezza né umanizzazione.',
           '', '## Shortlist della prima fase', '',markdown_table(compact),'',
           '![Variazione delle feature](style_movement.png)','',
           '## Cinque variazioni standardizzate maggiori per operazione', '',
           'Si ordinano per |g_z| le feature con q BH <0,05, escludendo soltanto i conteggi globali e TTR/hapax diagnostici. La tabella completa contiene tutte le 105 feature, anche nulle o non significative.',
           '',markdown_table(top[['operation','feature','mean_delta','delta_ci_low','delta_ci_high','paired_hedges_gz','q_bh_all_features_operations']]),'',
           '## Controllo della lunghezza', '',
           'Analisi primaria sui documenti interi. Il controllo usa soltanto prefissi iniziali con lo stesso esatto numero di parole lessicali spaCy (90–180, il valore comune più vicino a150), terminati al confine di una frase del parser e con punteggiatura finale. Non si appaiano fonti diverse e non si aggiungono parole. Caratteri e numero di frasi restano liberi.',
           '', '![Controllo con lunghezza abbinata](style_movement_length_matched.png)' if len(prefix_table) else 'Nessuna coppia con confini adatti.', '',
           f'Nel controllo sono disponibili {len(control)}/{len(full)} coppie. Fra le quattro feature della shortlist, le combinazioni con q BH <0,05 sono: {prefix_sig_names}. '
           'Una perdita di significatività rispetto al documento intero può dipendere dalla selezione, dalla diversa posizione osservata e dalla minore numerosità; non prova che il risultato iniziale fosse causato dalla sola lunghezza.', '',
           'Questo controllo cambia la popolazione: seleziona coppie con un confine di frase a conteggio identico e misura passaggi iniziali, non documenti completi. Non è una stima causale dell’effetto della sola lunghezza. Le frequenze normalizzate e MATTR50 riducono la dipendenza dai conteggi, ma possono comunque variare con contenuto, posizione e lunghezza. Vedere `prefix_matched_feature_values.csv` e le righe `exact_prefix_length_matched` di `summary.csv`.',
           '', '## Inferenza e file', '',
           f'Le feature sono le 105 definizioni fisse di `stylometry/features.py`: spaCy small, textstat, MATTR50 manuale. Prima si mediano le variazioni dei diversi editor dello stesso documento umano. Il documento sorgente è l’unità indipendente: media/mediana, test t appaiato su variazioni per fonte, IC95% bootstrap per fonte ({args.bootstrap} repliche). Hedges g_z = correzione J × media delle differenze / SD delle differenze fra fonti; non coincide con Hedges g pooled della prima fase. Gli IC di g_z ricampionano le stesse fonti. BH corregge insieme tutte le 105 ×5 ipotesi entro ciascuna delle due analisi.',
           '', 'Fonti con feature mancante vengono escluse soltanto per quella feature; `n_sources` e `n_pairs` lo rendono esplicito. Feature con delta sempre zero hanno g_z=0 e p=1; delta costante nonzero ha standardizzazione indefinita. I valori p non stimano la rilevanza pratica e gli IC non includono l’incertezza del parser.',
           '', '`paired_feature_values.csv`: valori prima/dopo/delta per coppia, senza testi. `document_features.csv`: cache derivata per testo unico. `summary.csv`: tutte le feature nei due protocolli. `stratified_mean_deltas.csv`: medie descrittive per dominio/editor. `data_manifest.json`: selezione e provenienza dei Parquet ufficiali. Tutti i testi restano localmente in `data/opai_style/`, ignorati da Git.',
           '', 'Riproduzione: `.venv/bin/python scripts/run_opai_style.py`; ripresa della cache locale: aggiungere `--reuse-features`.', '']
    (out/'report.md').write_text('\n'.join(lines))
    manifest['elapsed_seconds']=time.monotonic()-start
    manifest['prefix_matched_pairs']=len(control)
    manifest['n_summary_rows']=len(summary)
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'elapsed_seconds':manifest['elapsed_seconds'],'pairs':len(full),'prefix_pairs':len(control),
                      'unique_full_texts':len(texts),'summary_rows':len(summary)},indent=2),flush=True)


if __name__=='__main__':
    main()

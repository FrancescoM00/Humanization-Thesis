"""Native epistemic-caution challenges with score-independent source selection.

Rules provide intended experimental labels, not human gold labels. A triple
removes the target caution, strengthens commitment while retaining caution, and
paraphrases the original caution. No generative or NLI model is used here.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from humanization.paths import resolve_path

from humanization.datasets.mgtbench import DOMAINS, GENERATORS, canonical, process_spaces, trim_incomplete_tail
from humanization.semantics.perturbations import _edit, _replace_edits, _stable_hex, load_perturbation_nlp


CONDITIONS = ("hedging_complete", "hedging_partial", "hedging_preserved")
_NEGATION = {"not", "n't", "never", "neither", "nor", "no", "none", "without"}
_HEDGES = {"may", "might", "could", "can", "possibly", "probably", "perhaps", "maybe",
           "apparently", "potentially", "likely", "unlikely", "possible", "probable",
           "uncertain", "seem", "seems", "appear", "appears", "suggest", "suggests",
           "reportedly", "allegedly"}
_SAFE_MAY_VERBS = {"cause", "lead", "result", "reflect", "explain", "affect", "increase",
                   "decrease", "vary", "differ", "depend", "persist", "exist", "occur",
                   "arise", "represent", "contribute", "reduce", "improve", "indicate",
                   "require", "involve", "provide", "help", "pose", "prevent", "create"}
_HUMAN_SUBJECTS = {"person", "people", "participant", "student", "patient", "worker",
                    "employee", "doctor", "nurse", "man", "woman", "child", "character",
                    "citizen", "member", "reader", "user", "individual", "customer",
                    "guest", "attendee", "applicant", "candidate", "visitor", "prisoner"}
_FAMILY_PRIORITY = {"explicit_possible": 0, "explicit_probable": 1,
                    "epistemic_adverb_probability": 2, "epistemic_adverb_possibility": 3,
                    "initial_possibility_adverb": 4, "modal_perfect": 5,
                    "modal_be": 6, "modal_lexical": 7}
_PREFILTER = re.compile(r"\b(?:may|might)\s+\w+|\b(?:possibly|probably|perhaps|maybe)\b|\bit is (?:possible|probable|likely) that\b", re.I)
_SOURCE_KEY = re.compile(r"(?:mgtbench:)?(Essay|Reuters|WritingPrompts|WP):(\d+):")


def text_digest(text):
    return hashlib.sha256(canonical(text).encode()).hexdigest()


def _case_start(text, upper):
    if not text:
        return text
    return (text[0].upper() if upper else text[0].lower()) + text[1:]


def _lower_clause(text, token):
    return text if (token.pos_ == "PROPN" or token.text == "I"
                    or (len(token.text) > 1 and token.text.isupper())) else _case_start(text, False)


def _safe(sentence):
    words = [t for t in sentence if t.is_alpha]
    return (5 <= len(words) <= 70 and sentence.text.strip()[:1].isalpha()
            and sentence.text.rstrip().endswith(".")
            and len([line for line in sentence.text.splitlines() if line.strip()]) == 1
            and not any(t.text in {'"', '“', '”', "'", '‘', '’', '?', '!', ';', ':'}
                        or t.text.casefold() in _NEGATION | {"if", "unless", "whether", "suppose", "allowed", "permitted", "authorized", "entitled"}
                        or (t.dep_ == "conj" and t.pos_ in {"VERB", "AUX"})
                        for t in sentence))


def _subject(sentence):
    subjects = [c for c in sentence.root.children if c.dep_ in {"nsubj", "nsubjpass"}]
    if len(subjects) != 1:
        return None
    subject = subjects[0]
    if any(t.dep_ in {"relcl", "acl", "ccomp", "advcl"} for t in subject.subtree):
        return None
    if any(t.pos_ in {"VERB", "AUX"} and "Fin" in t.morph.get("VerbForm")
           for t in subject.subtree):
        return None  # Parser errors can put a reporting verb inside the subject.
    return subject


def _number(subject):
    word = subject.text.casefold()
    if (any(t.dep_ == "conj" for t in subject.children)
            and any(t.text.casefold() in {"or", "nor"} for t in subject.subtree)):
        return None  # Disjunctive/parenthetic coordination has uncertain agreement.
    if word == "i":
        return "first"
    if word in {"you", "we", "they", "these", "those"} or any(c.dep_ == "conj" for c in subject.children):
        return "plural"
    if word in {"he", "she", "it", "this", "that"}:
        return "singular"
    number = subject.morph.get("Number")
    # Plural-looking proper names such as United States have uncertain agreement.
    if subject.pos_ == "PROPN" and number == ["Plur"]:
        return None
    return "singular" if number == ["Sing"] else "plural" if number == ["Plur"] else None


def _finite(base, number):
    if number is None:
        return None
    lemma = base.lemma_.casefold()
    if lemma == "be":
        return {"first": "am", "plural": "are", "singular": "is"}[number]
    if number != "singular":
        return lemma
    if lemma == "have":
        return "has"
    if lemma == "do":
        return "does"
    if not re.fullmatch(r"[a-z]+", lemma):
        return None
    if re.search(r"[^aeiou]y$", lemma):
        return lemma[:-1] + "ies"
    return lemma + ("es" if re.search(r"(?:s|sh|ch|x|z|o)$", lemma) else "s")


def _retain_outer_space(original, body):
    leading = original[:len(original)-len(original.lstrip())]
    trailing = original[len(original.rstrip()):]
    return leading + body.strip() + trailing


def sentence_triplet(sentence: Any) -> dict | None:
    """Return one conservative native-caution triple, or reject the sentence.

    Complete/partial refer to the target proposition, not all claims in a text.
    Ambiguous permissions, ability, multiple hedges and polarity changes are
    excluded; the remaining semantic readings still require textual review.
    """
    if not _safe(sentence):
        return None
    raw = sentence.text
    stripped = raw.strip()
    tokens = [t for t in sentence if not t.is_space]
    # An explicit proposition-level wrapper supplies a clear possibility scope.
    match = re.match(r"It is (possible|probable|likely) that\s+(.+\.)$", stripped, re.I)
    if match:
        marker, clause = match.group(1).lower(), match.group(2)
        remaining = re.findall(r"\b\w+\b", clause.casefold())
        if any(w in _HEDGES | _NEGATION for w in remaining):
            return None
        first = next((t for t in tokens if t.idx >= sentence.start_char + raw.index(clause)), None)
        if first is None:
            return None
        complete = _case_start(clause, True)
        partial = re.sub(r"\b"+marker+r"\b", "likely" if marker == "possible" else "almost certain", stripped, count=1, flags=re.I)
        preserved = ("Possibly, " if marker == "possible" else "Probably, ") + _lower_clause(clause, first)
        family = "explicit_possible" if marker == "possible" else "explicit_probable"
    else:
        if any(t.dep_ in {'ccomp', 'relcl', 'advcl', 'acl'} for t in tokens):
            return None  # Do not put an independently asserted clause under a new wrapper.
        subject = _subject(sentence)
        if subject is None:
            return None
        modals = [t for t in tokens if t.tag_ == "MD" and t.text.casefold() in {"may", "might"}]
        if modals:
            if len(modals) != 1 or sum(t.tag_ == "MD" for t in tokens) != 1:
                return None
            modal = modals[0]
            if modal.head != sentence.root or modal.i+1 >= sentence.end:
                return None
            base = modal.nbor(1)
            if base.tag_ != "VB" or base.pos_ not in {"VERB", "AUX"}:
                return None
            if base.lemma_ == "be" and any(c.lemma_ == "able" for c in base.children):
                return None  # Keep ability constructions outside this targeted panel.
            if any(t.text.casefold() in _HEDGES and t != modal for t in tokens):
                return None
            subject_tokens = list(subject.subtree)
            start, end = min(t.idx for t in subject_tokens), max(t.idx+len(t.text) for t in subject_tokens)
            if raw[:start-sentence.start_char].strip() or sentence.doc.text[end:modal.idx].strip():
                return None
            # May+state/perfect is usually epistemic. Lexical may is restricted
            # to inanimate subjects and non-permission outcome predicates.
            human_subject = (subject.ent_type_ == "PERSON" or subject.lemma_.casefold() in _HUMAN_SUBJECTS
                             or subject.text.casefold() in {"i", "you", "he", "she", "we", "they"})
            epistemic_aux = (base.lemma_ == "be" or (base.lemma_ == "have" and base.dep_ == "aux"))
            if modal.text.casefold() == "may" and not epistemic_aux:
                if human_subject or base.lemma_ not in _SAFE_MAY_VERBS:
                    return None
            if modal.text.casefold() == "may" and human_subject and base.lemma_ == "be" and sentence.root != base:
                return None  # "Guests may be seated" may grant permission.
            if subject.text.casefold() in {"i", "you", "we"}:
                return None
            finite = _finite(base, _number(subject))
            if finite is None:
                return None
            a, b = modal.idx-sentence.start_char, base.idx+len(base.text)-sentence.start_char
            complete_raw = raw[:a] + finite + raw[b:]
            auxiliary = base.lemma_ == "be" or (base.lemma_ == "have" and base.dep_ == "aux")
            partial_words = finite + " probably" if auxiliary else "probably " + finite
            partial_raw = raw[:a] + partial_words + raw[b:]
            # The positive is a genuine rephrasing of the same possibility.
            complete = complete_raw.strip()
            partial = partial_raw.strip()
            first = tokens[0]
            preserved = "It is possible that " + _lower_clause(complete, first)
            marker = modal.text.casefold()
            family = "modal_be" if base.lemma_ == "be" else "modal_perfect" if base.lemma_ == "have" and base.dep_ == "aux" else "modal_lexical"
        else:
            adverbs = [t for t in tokens if t.text.casefold() in {"possibly", "perhaps", "maybe", "probably"}
                       and t.dep_ == "advmod" and t.head == sentence.root]
            if len(adverbs) != 1:
                return None
            adverb = adverbs[0]
            if any(t.text.casefold() in _HEDGES and t != adverb for t in tokens):
                return None
            if any(t.tag_ == "MD" and t.text.casefold() != "will" for t in tokens):
                return None
            marker = adverb.text.casefold()
            a = adverb.idx-sentence.start_char
            b = a+len(adverb.text_with_ws)
            initial = not raw[:a].strip()
            if initial:
                while b < len(raw) and (raw[b].isspace() or raw[b] == ','):
                    b += 1
            elif adverb.i+1 < sentence.end and (adverb.nbor(-1).is_punct or adverb.nbor(1).is_punct):
                return None
            subject_start = min(t.idx for t in subject.subtree)-sentence.start_char
            introductory = raw[b:subject_start] if initial else raw[:subject_start]
            if introductory.strip():
                return None  # Avoid moving reporting/introductory adjuncts inside the new wrapper.
            complete = (raw[:a]+raw[b:]).strip()
            if initial:
                complete = _case_start(complete, True)
            partial_marker = "almost certainly" if marker == "probably" else "probably"
            partial = (raw[:a]+_case_start(partial_marker, True) + raw[a+len(adverb.text):]).strip() if initial else (raw[:a]+partial_marker+raw[a+len(adverb.text):]).strip()
            first = next(t for t in tokens if t != adverb and not t.is_punct)
            preserved = ("It is probable that " if marker == "probably" else "It is possible that ") + _lower_clause(complete, first)
            family = ("initial_possibility_adverb" if initial and marker != "probably" else
                      "epistemic_adverb_probability" if marker == "probably" else "epistemic_adverb_possibility")
    targets = {"complete": complete, "partial": partial, "preserved": preserved}
    if len(set(targets.values()) | {stripped}) != 4:
        return None
    return {"family": family, "marker": marker,
            **{key: _retain_outer_space(raw, value) for key, value in targets.items()},
            "interpretation": "stronger commitment, not necessarily contradiction; preservation intended, not human gold"}


def collect_exclusions(paths, root: Path):
    """Collect raw-row keys and canonical previous text hashes from old inputs."""
    rows, hashes, manifest = set(), set(), []
    def visit(value):
        if isinstance(value, dict):
            domain = value.get("domain")
            if domain in DOMAINS and value.get("source_row") is not None:
                try: rows.add((domain, int(value["source_row"])))
                except (ValueError, TypeError): pass
            for key in ("original", "text"):
                if isinstance(value.get(key), str) and value[key].strip():
                    hashes.add(text_digest(value[key]))
            for item in value.values(): visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
        elif isinstance(value, str):
            for domain, row in _SOURCE_KEY.findall(value):
                rows.add(("WritingPrompts" if domain == "WP" else domain, int(row)))
    for name in paths:
        path = Path(name); path = resolve_path(root, path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing declared exclusion file: {path}")
        if path.suffix == '.csv':
            with path.open(newline='', encoding='utf-8') as handle:
                for record in csv.DictReader(handle): visit(record)
        elif path.suffix == '.jsonl':
            for line in path.read_text().splitlines():
                if line.strip(): visit(json.loads(line))
        else:
            visit(json.loads(path.read_text()))
        manifest.append({"path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows, hashes, manifest


def load_machine_records(raw_dir: Path, exclusion_files, root: Path,
                         minimum_words=80, maximum_words=1000):
    excluded_rows, excluded_texts, exclusions_manifest = collect_exclusions(exclusion_files, root)
    source_rows, parent, owner = {}, {}, {}
    for domain, prefix in DOMAINS.items():
        path = raw_dir/f'{prefix}_LLMs.csv'
        with path.open(newline='', encoding='utf-8') as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                key=(domain,index); source_rows[key]=row; parent[key]=key
    def find(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]];k=parent[k]
        return k
    for key,row in source_rows.items():
        for kind in ('human','prompt'):
            value=process_spaces(row.get(kind,'') or '')
            if kind == 'human': value=trim_incomplete_tail(value)
            if not value: continue
            item=(kind,canonical(value))
            if item in owner:
                a,b=find(key),find(owner[item]);parent[max(a,b)]=min(a,b)
            else: owner[item]=key
    forbidden_components={find(k) for k in excluded_rows if k in parent}
    records=[]; counts=Counter(); seen=set()
    for key,row in source_rows.items():
        component=find(key)
        if component in forbidden_components:
            counts['previously_used_row_or_connected_component']+=1;continue
        human_hash=text_digest(trim_incomplete_tail(process_spaces(source_rows[component].get('human','') or '')))[:16]
        group=f'mgtbench:{component[0]}:{component[1]}:{human_hash}'
        for generator in GENERATORS:
            text=trim_incomplete_tail(process_spaces(row.get(generator,'') or ''))
            if not minimum_words<=len(text.split())<=maximum_words:
                counts['missing_or_document_word_limits']+=1;continue
            digest=text_digest(text)
            if digest in excluded_texts or digest in seen:
                counts['previous_or_duplicate_canonical_text']+=1;continue
            seen.add(digest)
            if re.match(r'\s*(?:write|generate|create|compose|rewrite|summarize)\b',text,re.I):
                counts['instruction_like_opening']+=1;continue
            if not _PREFILTER.search(text):
                counts['no_native_hedge_marker']+=1;continue
            records.append({'source_id':f'hedging:{key[0]}:{key[1]}:{generator}:{digest[:16]}',
                            'source_group_id':group,'source_group':group,'domain':key[0],
                            'generator':generator,'source_row':key[1],'original':text,'text':text,
                            'canonical_original_sha256':digest,'raw_row_key':f'{key[0]}:{key[1]}'})
    return records, {'n_raw_rows':len(source_rows),'n_excluded_raw_rows':len(excluded_rows),
                     'n_excluded_connected_components':len(forbidden_components),
                     'n_excluded_canonical_texts':len(excluded_texts),'filter_counts':dict(counts),
                     'exclusion_files':exclusions_manifest,'n_marker_prefilter_records':len(records)}


def discover_candidates(records, nlp=None, seed=20261111, max_per_cell=10, max_per_domain=None):
    """Parse score-independent queues, retaining native eligible source triples."""
    if nlp is None: nlp=load_perturbation_nlp()
    cells=defaultdict(list)
    for record in records: cells[(record['domain'],record['generator'])].append(record)
    queues={key:deque(sorted(items,key=lambda r:_stable_hex(seed,r['source_id']))) for key,items in sorted(cells.items())}
    accepted=Counter(); accepted_domains=Counter(); diagnostics=[]; candidates=[]; accepted_groups=set()
    # Eligibility is evaluated in a deterministic balanced stream. A cell cap
    # bounds parsing work without using scores or changing experimental labels.
    def stream():
        while any(queues.values()):
            progressed=False
            for key,queue in queues.items():
                # Different generators from one raw-row/source component are
                # not independent sources and must not consume separate caps.
                while queue and queue[0]['source_group_id'] in accepted_groups:
                    skipped=queue.popleft()
                    diagnostics.append({**{k:skipped[k] for k in ('source_id','domain','generator','source_row')},
                                        'eligible':False,'reason':'source_group_already_accepted_before_parse'})
                domain_available = max_per_domain is None or accepted_domains[key[0]] < max_per_domain
                if queue and accepted[key]<max_per_cell and domain_available:
                    progressed=True; yield queue.popleft()
            if not progressed: break
    pending=iter(stream())
    def inputs():
        for record in pending: yield (record['original'],record)
    for doc,record in nlp.pipe(inputs(),as_tuples=True,batch_size=16):
        key=(record['domain'],record['generator'])
        if accepted[key]>=max_per_cell: continue
        if max_per_domain is not None and accepted_domains[record['domain']] >= max_per_domain: continue
        if record['source_group_id'] in accepted_groups:
            diagnostics.append({**{k:record[k] for k in ('source_id','domain','generator','source_row')},
                                'eligible':False,'reason':'source_group_already_accepted_in_prefetched_batch'})
            continue
        choices=[]
        for index,sentence in enumerate(doc.sents):
            triple=sentence_triplet(sentence)
            if triple is not None:
                # Reject exact duplicate support already asserted elsewhere.
                other=canonical(doc.text[:sentence.start_char]+' '+doc.text[sentence.end_char:])
                if canonical(triple['complete']).rstrip('.') in other: continue
                choices.append((index,sentence,triple))
        if not choices:
            diagnostics.append({**{k:record[k] for k in ('source_id','domain','generator','source_row')},'eligible':False,'reason':'no_conservative_native_triplet'})
            continue
        index,sentence,triple=min(choices,key=lambda x:(_FAMILY_PRIORITY[x[2]['family']],_stable_hex(seed,record['source_id'],x[1].text)))
        candidate={**record,'target_original':sentence.text,'target_start_char':sentence.start_char,
                   'target_end_char':sentence.end_char,'sentence_index':index,'family':triple['family'],
                   'marker':triple['marker'],'targets':{name:triple[name] for name in ('complete','partial','preserved')}}
        candidates.append(candidate);accepted[key]+=1;accepted_domains[record['domain']]+=1;accepted_groups.add(record['source_group_id'])
        diagnostics.append({**{k:record[k] for k in ('source_id','domain','generator','source_row')},'eligible':True,'reason':triple['family']})
    return candidates,diagnostics


def select_sources(candidates, n_sources, seed=20261111, excluded_groups=()):
    remaining=[r for r in candidates if r['source_group_id'] not in set(excluded_groups)]
    selected=[];seen=set(excluded_groups);families=Counter();generators=Counter()
    domains=sorted({r['domain'] for r in remaining})
    while len(selected)<n_sources:
        progressed=False
        for domain in domains:
            if len(selected)>=n_sources:break
            eligible=[r for r in remaining if r['domain']==domain and r['source_group_id'] not in seen]
            if not eligible:continue
            pick=min(eligible,key=lambda r:(generators[(domain,r['generator'])],families[(domain,r['family'])],_stable_hex(seed,r['source_id'])))
            selected.append(pick);seen.add(pick['source_group_id']);generators[(domain,pick['generator'])]+=1;families[(domain,pick['family'])]+=1;progressed=True
        if not progressed:break
    return selected


def source_pairs(source, seed=20261111):
    pairs=[]
    for condition,key,label in zip(CONDITIONS,('complete','partial','preserved'),(1,1,0)):
        target=source['targets'][key]
        edit=_edit(source['target_start_char'],source['target_end_char'],source['target_original'],target)
        modified=_replace_edits(source['original'],[edit])
        pair_id='hedge:'+_stable_hex(seed,source['source_id'],condition,modified)[:24]
        pairs.append({**{k:source[k] for k in ('source_id','source_group_id','source_group','domain','generator','source_row')},
                      'id':pair_id,'pair_id':pair_id,'perturbation':condition,'condition':condition,
                      'dataset':'hedging_extension','evaluation_split':'test','status':'ok',
                      'original':source['original'],'modified':modified,'target_original':source['target_original'],
                      'target_modified':target,'intended_label':label,'label_source':'rule_intent_not_human_gold',
                      'metadata':{'rule':source['family']+':'+key,'family':source['family'],'marker':source['marker'],
                                  'edits':[edit],'target_start_char':source['target_start_char'],
                                  'target_end_char':source['target_end_char'],'sentence_index':source['sentence_index'],
                                  'semantic_label_verified':False,'source_kind':'native_machine_text',
                                  'semantic_interpretation':'stronger commitment not necessarily contradiction' if label else 'preserved uncertainty intended',
                                  'canonical_original_sha256':source['canonical_original_sha256']}})
    return pairs


def prepare_panel(protocol_path:Path, output:Path, root:Path):
    started=datetime.now(timezone.utc);root=root.resolve();output=output.resolve();protocol_path=protocol_path.resolve()
    raw_protocol=protocol_path.read_bytes();protocol=json.loads(raw_protocol)
    if (output/'pairs.jsonl').exists():raise FileExistsError('Preserving an existing prepared panel; use a new output directory')
    raw_dir=resolve_path(root,Path(protocol.get('raw_dir','data/raw/mgtbench')))
    exclusions=protocol.get('exclusion_files',[])
    names=[p['path'] if isinstance(p,dict) else p for p in exclusions]
    for name,expected_hash in {**protocol.get('exclusion_sha256',{}),**protocol.get('raw_sha256',{})}.items():
        path=Path(name);path=resolve_path(root, path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_hash:
            raise ValueError(f'Frozen preparation input changed: {name}')
    for item in exclusions:
        if isinstance(item,dict) and item.get('sha256'):
            path=Path(item['path']);path=resolve_path(root, path)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Frozen exclusion input changed')
    records,screen=load_machine_records(raw_dir,names,root,int(protocol.get('minimum_words',80)),int(protocol.get('maximum_words',1000)))
    nlp=load_perturbation_nlp(protocol.get('model','en_core_web_sm'))
    candidates,diagnostics=discover_candidates(records,nlp,int(protocol.get('seed',20261111)),
                                             int(protocol.get('max_per_cell',10)),
                                             protocol.get('max_unique_candidates_per_domain'))
    n=int(protocol.get('n_sources',60));seed=int(protocol.get('seed',20261111))
    def fail_preparation(message):
        # Preserve applicability diagnostics when a registered quota cannot be
        # met. These are candidates, not a completed panel or metric results.
        output.mkdir(parents=True,exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        failure={'status':'incomplete_no_pairs_exported','reason':message,
                 'created_at_utc':datetime.now(timezone.utc).isoformat(),
                 'n_unique_eligible_groups':len({r['source_group_id'] for r in candidates}),
                 'eligible_per_domain':dict(Counter(r['domain'] for r in candidates)),
                 'eligible_per_cell':dict(Counter(r['domain']+':'+r['generator'] for r in candidates)),
                 'eligible_per_family':dict(Counter(r['family'] for r in candidates)),
                 'screening':screen,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (output/f'preparation_failure_{stamp}.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2)+'\n')
        (output/f'candidate_pool_debug_{stamp}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in candidates))
        import pandas as pd
        pd.DataFrame(diagnostics).to_csv(output/f'screening_attempt_{stamp}.csv',index=False)
        raise RuntimeError(message+'; native applicability by domain: '+str(failure['eligible_per_domain']))
    selected=select_sources(candidates,n,seed)
    if len(selected)!=n:fail_preparation(f'Only {len(selected)} eligible independent sources; required {n}; no synthetic hedge insertion or rule relaxation')
    per_domain=Counter(r['domain'] for r in selected)
    expected=protocol.get('target_sources_per_domain',protocol.get('sources_per_domain'))
    if isinstance(expected,int) and set(per_domain.values())!={expected}:fail_preparation('Native applicability cannot meet the frozen domain quota')
    reserve=select_sources(candidates,int(protocol.get('reserve_sources',30)),seed+1,[r['source_group_id'] for r in selected])
    pairs=[p for r in selected for p in source_pairs(r,seed)]
    output.mkdir(parents=True,exist_ok=True)
    def jsonl(name,rows):(output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    jsonl('pairs.jsonl',pairs);jsonl('sources.jsonl',selected)
    jsonl('candidate_reserve.jsonl',[{**r,'pairs':source_pairs(r,seed)} for r in reserve])
    import pandas as pd
    pd.DataFrame(diagnostics).to_csv(output/'exclusions.csv',index=False)
    pd.DataFrame([{k:r[k] for k in ('source_id','source_group_id','domain','generator','source_row','family','marker')} for r in candidates]).to_csv(output/'source_pool.csv',index=False)
    pd.DataFrame([{'id':p['id'],'source_id':p['source_id'],'source_group_id':p['source_group_id'],'condition':p['condition'],
                   'domain':p['domain'],'generator':p['generator'],'original':p['original'],'modified':p['modified'],
                   'target_original':p['target_original'],'target_modified':p['target_modified'],
                   'validity':'','reviewer_source':'','notes':''} for p in pairs]).to_csv(output/'review_pairs.csv',index=False)
    blind=sorted(pairs,key=lambda p:_stable_hex(seed,'human-blind-order',p['id']))
    pd.DataFrame([{'review_id':f'H{index:03d}','original':p['original'],'modified':p['modified'],
                   'meaning_preserved':'','change_type':'','notes':''} for index,p in enumerate(blind,1)]).to_csv(output/'human_annotation.csv',index=False)
    pd.DataFrame([{'review_id':f'H{index:03d}','id':p['id'],'source_group_id':p['source_group_id'],
                   'condition':p['condition'],'intended_label':p['intended_label']} for index,p in enumerate(blind,1)]).to_csv(output/'human_annotation_key.csv',index=False)
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'preparation_started_utc':started.isoformat(),
              'protocol_sha256':hashlib.sha256(raw_protocol).hexdigest(),'n_sources':len(selected),'n_pairs':len(pairs),
              'n_intended_alterations':2*len(selected),'n_presumed_preserved':len(selected),'n_reserve_sources':len(reserve),
              'sources_per_domain':dict(per_domain),'sources_per_generator':dict(Counter(r['generator'] for r in selected)),
              'sources_per_family':dict(Counter(r['family'] for r in selected)),
              'n_eligible_candidates':len(candidates),'screening':screen,'score_independent_selection':True,
              'human_gold_labels_available':False,'synthetic_hedging_added_to_originals':False,
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'pairs_sha256':hashlib.sha256((output/'pairs.jsonl').read_bytes()).hexdigest()}
    (output/'preparation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest

"""Segnalazioni lessicali fisse per modifiche semantiche, indipendenti dagli output NLI.

Queste regole rilevano cambiamenti nel contenuto affermato, non la verità dei
fatti. Gli inventari sono fissati prima di valutare gli score. Le segnalazioni
numeriche confrontano multinsiemi di valori normalizzati: sostituire 20 con 30
è rilevabile anche se entrambi i testi contengono un solo numero. La
normalizzazione dei cardinali e ordinali inglesi è volutamente limitata.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re

HEDGE_MODALS = frozenset({'may', 'might', 'could'})
HEDGE_LEMMAS = frozenset({'possibly', 'probably', 'perhaps', 'apparently', 'likely',
                         'unlikely', 'seem', 'appear', 'suggest', 'indicate',
                         'estimate', 'tend', 'potentially', 'arguably', 'presumably'})
NEGATIONS = {'not': 'not', "n't": 'not', 'never': 'never', 'no': 'no',
             'none': 'none', 'neither': 'neither', 'nor': 'nor', 'without': 'without'}
QUANTIFIERS = {
    'all': 'universal', 'every': 'universal', 'each': 'universal',
    'always': 'always', 'everyone': 'everyone', 'everybody': 'everyone',
    'everything': 'everything', 'some': 'some', 'several': 'several',
    'many': 'many', 'much': 'much', 'most': 'most', 'few': 'few',
    'someone': 'someone', 'somebody': 'someone', 'something': 'something',
    'any': 'any', 'none': 'none', 'no': 'no', 'never': 'never',
}
UNITS = dict(zip(('zero one two three four five six seven eight nine ten eleven '
                  'twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split(), range(20)))
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
SCALES = {'hundred': 100, 'thousand': 1000, 'million': 10**6, 'billion': 10**9, 'trillion': 10**12}
ORDINALS = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6,
            'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10, 'eleventh': 11,
            'twelfth': 12, 'twentieth': 20, 'thirtieth': 30, 'hundredth': 100}
CURRENCIES = {'$': 'USD', 'us$': 'USD', 'usd': 'USD', 'dollar': 'USD', 'dollars': 'USD',
              '£': 'GBP', 'gbp': 'GBP', 'pound': 'GBP', 'pounds': 'GBP',
              '€': 'EUR', 'eur': 'EUR', 'euro': 'EUR', 'euros': 'EUR'}
GUARD_FEATURES = ['guard_hedge_change', 'guard_negation_change',
                  'guard_number_change', 'guard_quantifier_change']
_LITERAL = re.compile(r'^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?$')
_DIGIT_ORDINAL = re.compile(r'^(\d+)(st|nd|rd|th)$', re.I)


def _surface(token):
    return token.text.casefold().replace('’', "'")


def _decimal_string(number):
    return format(Decimal(number).normalize(), 'f')


def _cardinal_words(words):
    """Analizza un piccolo sottoinsieme grammaticale; scarta elenchi come 'one two'."""
    total, current, previous, last_scale = 0, 0, '', 10**15
    seen = False
    for word in words:
        if word == '-':
            continue
        if word == 'and':
            if not seen or previous not in {'hundred', 'scale'}:
                return None
            previous = 'and'
            continue
        if word in UNITS:
            if previous == 'unit' or (previous == 'tens' and UNITS[word] >= 10):
                return None
            current += UNITS[word]
            previous = 'unit'
        elif word in TENS:
            if previous in {'unit', 'tens'}:
                return None
            current += TENS[word]
            previous = 'tens'
        elif word == 'hundred':
            if previous != 'unit' or not 1 <= current <= 9:
                return None
            current *= 100
            previous = 'hundred'
        elif word in SCALES:
            scale = SCALES[word]
            if not seen or not current or scale >= last_scale:
                return None
            total += current * scale
            current, previous, last_scale = 0, 'scale', scale
        else:
            return None
        seen = True
    return total + current if seen and previous != 'and' else None


def numeric_values(doc):
    """Restituisce un multinsieme normalizzato di numeri e unità senza dedurne la verità.

    Le date non vengono scartate silenziosamente: cambiare un anno può essere
    significativo. Non è presente un analizzatore generale per aritmetica,
    notazione scientifica o conversione delle unità.
    """
    tokens, values, index = list(doc), Counter(), 0
    word_numbers = set(UNITS) | set(TENS) | set(SCALES)
    while index < len(tokens):
        token = tokens[index]
        word = _surface(token)
        number, end, ordinal = None, index + 1, False
        literal = word.replace('−', '-')
        if _LITERAL.fullmatch(literal):
            try:
                pieces = literal.replace(',', '').split('/')
                number = Decimal(pieces[0]) / Decimal(pieces[1]) if len(pieces) == 2 else Decimal(pieces[0])
            except (InvalidOperation, ZeroDivisionError):
                number = None
            if number is not None and end < len(tokens) and _surface(tokens[end]) in SCALES:
                number *= SCALES[_surface(tokens[end])]
                end += 1
        elif word in word_numbers:
            sequence = []
            for finish in range(index, min(index + 16, len(tokens))):
                item = _surface(tokens[finish])
                if item not in word_numbers | {'and', '-'}:
                    break
                sequence.append(item)
                candidate = _cardinal_words(sequence)
                if candidate is not None:
                    number, end = Decimal(candidate), finish + 1
        elif word in ORDINALS or _DIGIT_ORDINAL.fullmatch(word):
            number = Decimal(ORDINALS[word] if word in ORDINALS else _DIGIT_ORDINAL.fullmatch(word).group(1))
            ordinal = True
        if number is None:
            index += 1
            continue
        unit = 'ordinal' if ordinal else 'number'
        if index and _surface(tokens[index - 1]) in CURRENCIES:
            unit = CURRENCIES[_surface(tokens[index - 1])]
        if end < len(tokens):
            following = _surface(tokens[end])
            if following in {'%', 'percent', 'percentage'}:
                unit, end = 'percent', end + 1
            elif following == 'per' and end + 1 < len(tokens) and _surface(tokens[end + 1]) == 'cent':
                unit, end = 'percent', end + 2
            elif following in CURRENCIES:
                unit, end = CURRENCIES[following], end + 1
        values[unit + ':' + _decimal_string(number)] += 1
        index = end
    return values


def text_inventory(doc):
    hedges, negatives, quantifiers = Counter(), Counter(), Counter()
    for token in doc:
        word = _surface(token)
        lemma = token.lemma_.casefold() if token.lemma_ else word
        if word in HEDGE_MODALS and token.tag_ == 'MD':
            hedges['modal'] += 1  # Gli scambi may/might non generano una segnalazione di conteggio.
        elif lemma in HEDGE_LEMMAS and token.pos_ not in {'PROPN', 'NUM'}:
            hedges['epistemic_lexical'] += 1
        if word in NEGATIONS:
            negatives[NEGATIONS[word]] += 1
        if word in QUANTIFIERS:
            quantifiers[QUANTIFIERS[word]] += 1
    return {'hedge': hedges, 'negation': negatives, 'number': numeric_values(doc), 'quantifier': quantifiers}


def inventory_delta(original, modified):
    result = {}
    for family in ['hedge', 'negation', 'number', 'quantifier']:
        left, right = Counter(original[family]), Counter(modified[family])
        removed, added = left - right, right - left
        before, after = sum(left.values()), sum(right.values())
        prefix = 'guard_' + family
        result.update({prefix + '_original_count': before, prefix + '_modified_count': after,
                       prefix + '_signed_count_delta': after - before,
                       prefix + '_absolute_count_delta': abs(after - before),
                       prefix + '_lost_items': sum(removed.values()), prefix + '_gained_items': sum(added.values())})
        # Per le cautele usa il totale delle affermazioni, evitando segnalazioni quando
        # una cautela sostituisce un'altra. Le altre famiglie confrontano il multinsieme canonico.
        result[prefix + '_change'] = abs(after - before) if family == 'hedge' else max(sum(removed.values()), sum(added.values()))
        result[prefix + '_count_loss'] = max(before - after, 0)
        result[prefix + '_count_gain'] = max(after - before, 0)
    return result


def score_lexical_guards(pairs, progress_callback=None):
    import spacy
    import pandas as pd
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    records = list(pairs)
    texts = list(dict.fromkeys(text for row in records for text in [row['original'], row['modified']]))
    inventories = {}
    for index, (text, doc) in enumerate(zip(texts, nlp.pipe(texts, batch_size=64)), 1):
        inventories[hashlib.sha256(text.encode()).hexdigest()] = text_inventory(doc)
        if progress_callback and (index % 100 == 0 or index == len(texts)):
            progress_callback(index, len(texts))
    rows = []
    for pair in records:
        left, right = [inventories[hashlib.sha256(pair[key].encode()).hexdigest()] for key in ['original', 'modified']]
        rows.append({'id': pair['id'], **inventory_delta(left, right)})
    return pd.DataFrame(rows)


def definitions():
    return {'hedge_modals_require_tag_MD': sorted(HEDGE_MODALS), 'hedge_lemmas': sorted(HEDGE_LEMMAS),
            'negation_canonicalization': NEGATIONS, 'quantifier_canonicalization': QUANTIFIERS,
            'primary_guard_features': GUARD_FEATURES,
            'number_normalization': 'Decimal literals, English cardinal subset, basic ordinals, percent and USD/GBP/EUR labels; multisets, not just counts.',
            'warnings_are_not_semantic_gold': True,
            'known_limits': ['Modal ability/permission vs epistemic uncertainty is not fully resolved.',
                             'Appear/suggest/indicate and quantifiers are lexical proxies with contextual false positives.',
                             'No general arithmetic, coreference, numerical unit conversion or scope reasoning.']}


def self_check():
    """Piccoli esempi semantici, distinti da test che ripetono l'implementazione."""
    import spacy
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    cases = [
        ('We counted three items.', 'We counted 3 items.', 'guard_number_change', 0),
        ('We spent 1,000 dollars.', 'We spent one thousand dollars.', 'guard_number_change', 0),
        ('The rate is twenty percent.', 'The rate is 20%.', 'guard_number_change', 0),
        ('The rate is 20%.', 'The rate is 30%.', 'guard_number_change', 1),
        ("They did not arrive.", "They didn't arrive.", 'guard_negation_change', 0),
        ('They arrived.', 'They did not arrive.', 'guard_negation_change', 1),
        ('Some people arrived.', 'All people arrived.', 'guard_quantifier_change', 1),
        ('All students arrived.', 'Every student arrived.', 'guard_quantifier_change', 0),
        ('They may arrive.', 'They might arrive.', 'guard_hedge_change', 0),
        ('In May, they may arrive.', 'In May, they arrive.', 'guard_hedge_change', 1),
    ]
    for before, after, key, expected in cases:
        value = inventory_delta(text_inventory(nlp(before)), text_inventory(nlp(after)))[key]
        if value != expected:
            raise AssertionError(f'{key}: expected {expected}, got {value}: {before!r} -> {after!r}')
    return {'passed': True, 'n_semantic_examples': len(cases)}

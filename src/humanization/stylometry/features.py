"""Feature stilistiche fisse e interpretabili a livello di documento.

Qui non si adattano modelli, ricavano dizionari dai dati o scaricano modelli.
Le parole lessicali sono token spaCy con ``is_alpha``, normalizzati con casefold
senza lemmatizzazione. Le lunghezze delle frasi contano tali parole ed escludono
le frasi prive di esse. Le proporzioni POS usano invece tutti i token diversi
da spazi e punteggiatura, conservando numeri e frammenti di contrazioni. I tassi
di punteggiatura contano caratteri letterali ogni 100 parole lessicali; anche
le frequenze delle parole funzionali usano questo denominatore.

Le feature tratte da Wikipedia sono esplicitamente indicatori approssimativi:
la loro presenza non prova la paternità IA e non rende misurabili affermazioni
semantiche come un'analisi priva di supporto. Il dizionario è stato definito dalla
tesi fornita prima di esaminare gli effetti.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import math
import re
from typing import Any, Iterable, Sequence


POS_TAGS = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "SCONJ", "SYM", "VERB", "X",
)

# Vocabolario grammaticale compatto e fisso, non selezionato su questo dataset.
FUNCTION_WORDS = (
    "a", "an", "the", "and", "or", "but", "if", "because", "although",
    "while", "of", "to", "in", "on", "at", "for", "with", "from", "by",
    "as", "that", "which", "who", "this", "these", "it", "its", "they",
    "their", "he", "she", "we", "our", "you", "your", "i", "my", "is",
    "are", "was", "were", "be", "been", "have", "has", "had", "do",
    "does", "did", "can", "could", "will", "would", "should", "may",
    "might", "must", "not",
)

# I lemmi coprono le flessioni comuni, ma il vocabolario rimane fisso.
WIKIPEDIA_AI_LEMMAS = frozenset({
    "delve", "intricate", "tapestry", "landscape", "pivotal", "crucial",
    "foster", "showcase", "underscore", "highlight",
})
WIKIPEDIA_EMPHASIS_PHRASES = (
    "pivotal moment", "crucial role", "lasting legacy", "significant shift",
    "broader trends",
)
WIKIPEDIA_CONCLUSION_PHRASES = (
    "in summary", "in conclusion", "it is important to note", "future prospects",
    "future outlook", "challenges and legacy",
)
WIKIPEDIA_CHAT_PHRASES = (
    "here is a", "i hope this helps", "would you like", "let me know",
)

PUNCTUATION_CHARACTERS = {
    "comma": ",", "period": ".", "semicolon": ";", "colon": ":",
    "question": "?", "exclamation": "!", "em_dash": "—", "hyphen": "-",
    "parenthesis": "()", "double_quote": '\"“”',
}

# Feature -> famiglia generale. Esportata per analisi di esclusione prestabilite.
FEATURE_GROUPS: dict[str, str] = {}
FEATURE_SCHEMA: dict[str, str] = {}


def _register(name: str, group: str, description: str) -> None:
    FEATURE_GROUPS[name] = group
    FEATURE_SCHEMA[name] = description


for _name, _description in {
    "n_words": "Count of alphabetic spaCy tokens (lexical words).",
    "n_characters": "Original text length in Unicode code points.",
    "n_sentences": "Count of spaCy sentences containing at least one lexical word.",
}.items():
    _register(_name, "length", _description)

for _name, _description in {
    "sentence_length_mean": "Mean lexical words per nonempty spaCy sentence.",
    "sentence_length_std": "Population SD of lexical words per sentence (ddof=0).",
    "sentence_length_cv": "Sentence-length population SD divided by its mean.",
}.items():
    _register(_name, "sentence", _description)

for _name, _description in {
    "mattr_50": "Mean TTR over all width-50 lexical-word windows; NaN below 50 words.",
    "ttr": "Unique casefolded lexical words / lexical words; length-sensitive diagnostic.",
    "hapax_ratio": "Types occurring exactly once / lexical words; length-sensitive.",
    "word_length_mean": "Mean number of Unicode characters per lexical word.",
}.items():
    _register(_name, "lexical", _description)

for _name in PUNCTUATION_CHARACTERS:
    # Il trattino lungo ha anche un ruolo negli indicatori Wikipedia; una sola colonna evita duplicati.
    _register(f"punct_{_name}_per100", "punctuation", "Literal character count per 100 lexical words.")
_register("punct_total_per100", "punctuation", "spaCy punctuation-token count per 100 lexical words.")
for _tag in POS_TAGS:
    _register(f"pos_{_tag.lower()}", "pos", "UPOS share of non-space, non-punctuation spaCy tokens.")
for _word in FUNCTION_WORDS:
    _register(f"fw_{_word}", "function_words", "Casefolded literal word count / lexical words.")
_register("fw_total", "function_words", "Share of lexical words in the fixed function-word inventory.")
for _name, _description in {
    "flesch_reading_ease": "textstat Flesch reading ease, en_US syllabification.",
    "flesch_kincaid_grade": "textstat Flesch-Kincaid grade, en_US syllabification.",
    "syllables_per_word": "textstat syllables divided by spaCy lexical words, en_US.",
}.items():
    _register(_name, "readability", _description)

for _name, _description in {
    "wiki_ai_vocabulary_per1000": "Fixed AI-vocabulary lemma occurrences per 1000 lexical words.",
    "wiki_emphasis_per1000": "Fixed thesis emphasis phrase occurrences per 1000 lexical words.",
    "wiki_conclusion_per1000": "Fixed conclusion/caveat phrase occurrences per 1000 lexical words.",
    "wiki_chat_per1000": "Fixed conversational phrase occurrences per 1000 lexical words.",
    "wiki_negative_parallelism_per1000": "Sentence-level not ... but / rather than regex matches per 1000 words.",
    "wiki_markdown_per1000": "Markdown heading/bullet/bold marker occurrences per 1000 words.",
}.items():
    _register(_name, "wikipedia", _description)

# I modelli prestabiliti possono usare questa colonna condivisa insieme alle colonne wiki_.
WIKIPEDIA_FEATURES = tuple(
    name for name, group in FEATURE_GROUPS.items() if group == "wikipedia"
) + ("punct_em_dash_per100",)

# Le conserva per la diagnostica; la valutazione può escludere le misure globali
# di diversità sensibili alla lunghezza dal modello primario basato solo sullo stile.
LENGTH_SENSITIVE_FEATURES = ("n_words", "n_characters", "n_sentences", "ttr", "hapax_ratio")


def mattr(tokens: Sequence[str], window: int = 50) -> float:
    """Calcola esattamente il MATTR a finestra scorrevole in tempo O(N) e memoria O(window).

    I token devono essere già normalizzati. Restituisce NaN per N < window:
    una finestra minore cambierebbe la quantità stimata e renderebbe i confronti
    dipendenti dalla lunghezza.
    """
    if window <= 0:
        raise ValueError("MATTR window must be a positive integer")
    n_tokens = len(tokens)
    if n_tokens < window:
        return math.nan
    counts = Counter(tokens[:window])
    unique_sum = len(counts)
    for end in range(window, n_tokens):
        outgoing = tokens[end - window]
        counts[outgoing] -= 1
        if counts[outgoing] == 0:
            del counts[outgoing]
        counts[tokens[end]] += 1
        unique_sum += len(counts)
    return unique_sum / (window * (n_tokens - window + 1))


def load_nlp(model: str = "en_core_web_sm") -> Any:
    """Carica la pipeline inglese spaCy installata; se manca si interrompe senza scaricarla.

    Mantiene parser e lemmatizzatore per i confini delle frasi e la ricerca nel
    vocabolario fisso. Disabilita soltanto NER, non necessario per queste feature.
    """
    import spacy

    nlp = spacy.load(model, disable=["ner"])
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        raise ValueError("The spaCy model must provide trained sentence boundaries")
    return nlp


@lru_cache(maxsize=1)
def _readability() -> Any:
    # Verifica il corpus NLTK *prima* che textstat possa avviarne il download automatico.
    # Installazione e provenienza spettano allo script di configurazione, non all’estrattore.
    import nltk
    from humanization.paths import project_root
    nltk.data.path.insert(0,str(project_root()/"data/resources/nltk_data"))
    from textstat import textstat
    from textstat.backend.utils._get_cmudict import get_cmudict
    from unittest.mock import patch

    try:
        nltk.data.find("corpora/cmudict")
    except LookupError:
        try:
            nltk.data.find("corpora/cmudict.zip")
        except LookupError as exc:
            raise RuntimeError(
                "Install the NLTK cmudict corpus before extracting en_US "
                "readability features; automatic downloads are disabled."
            ) from exc
    textstat.set_lang("en_US")
    # textstat 0.7.10 chiama sempre nltk.download dentro get_cmudict.
    # Inizializza la cache ordinaria della funzione con il corpus locale verificato,
    # sostituendo solo il download con un controllo della presenza locale. Formule e
    # dizionario rimangono esattamente quelli di textstat en_US. Questa inizializzazione
    # usa un solo thread, in accordo con il funzionamento di extract_records.
    def preinstalled_only(package: str, *args: Any, **kwargs: Any) -> bool:
        if package != "cmudict":
            raise RuntimeError(f"Unexpected textstat corpus request: {package}")
        return True

    with patch.object(nltk, "download", preinstalled_only):
        get_cmudict("en_US")
    return textstat


def _phrase_count(text: str, phrases: Iterable[str]) -> int:
    return sum(
        len(re.findall(r"(?<!\w)" + r"\s+".join(map(re.escape, phrase.split())) + r"(?!\w)", text, flags=re.I))
        for phrase in phrases
    )


def extract_features(doc: Any, text: str | None = None) -> dict[str, float]:
    """Estrae tutte le feature di ``FEATURE_SCHEMA`` da un documento già analizzato.

    Input senza parole lessicali hanno feature normalizzate NaN e conteggi zero.
    La SD della lunghezza delle frasi è zero se il documento ha una sola frase.
    La punteggiatura totale conta token di punteggiatura spaCy; i singoli segni
    contano caratteri letterali, quindi non devono necessariamente sommarsi.
    Le espressioni regolari degli indicatori retorici si applicano dentro ogni
    frase, evitando corrispondenze fra frammenti non collegati di frasi diverse.
    """
    text = doc.text if text is None else text
    words = [token for token in doc if token.is_alpha]
    normalized = [token.text.casefold() for token in words]
    n_words = len(words)
    counts = Counter(normalized)
    sentence_spans = list(doc.sents)
    lengths = [sum(token.is_alpha for token in sentence) for sentence in sentence_spans]
    lengths = [length for length in lengths if length > 0]
    n_sentences = len(lengths)
    mean = sum(lengths) / n_sentences if n_sentences else math.nan
    std = math.sqrt(sum((length - mean) ** 2 for length in lengths) / n_sentences) if n_sentences else math.nan

    def rate(value: float, scale: float = 1.0) -> float:
        return scale * value / n_words if n_words else math.nan

    features = {
        "n_words": float(n_words),
        "n_characters": float(len(text)),
        "n_sentences": float(n_sentences),
        "sentence_length_mean": mean,
        "sentence_length_std": std,
        "sentence_length_cv": std / mean if n_sentences else math.nan,
        "mattr_50": mattr(normalized, 50),
        "ttr": rate(len(counts)),
        "hapax_ratio": rate(sum(count == 1 for count in counts.values())),
        "word_length_mean": rate(sum(len(word) for word in normalized)),
    }
    for name, characters in PUNCTUATION_CHARACTERS.items():
        features[f"punct_{name}_per100"] = rate(sum(text.count(char) for char in characters), 100)
    features["punct_total_per100"] = rate(sum(token.is_punct for token in doc), 100)

    pos_tokens = [token for token in doc if not token.is_space and not token.is_punct]
    pos_counts = Counter(token.pos_ or "X" for token in pos_tokens)
    for tag in POS_TAGS:
        features[f"pos_{tag.lower()}"] = pos_counts[tag] / len(pos_tokens) if pos_tokens else math.nan

    for word in FUNCTION_WORDS:
        features[f"fw_{word}"] = rate(counts[word])
    features["fw_total"] = rate(sum(counts[word] for word in FUNCTION_WORDS))

    if n_words:
        readability = _readability()
        features.update({
            "flesch_reading_ease": float(readability.flesch_reading_ease(text)),
            "flesch_kincaid_grade": float(readability.flesch_kincaid_grade(text)),
            "syllables_per_word": rate(readability.syllable_count(text)),
        })
    else:
        features.update({name: math.nan for name, group in FEATURE_GROUPS.items() if group == "readability"})

    # Usa le forme letterali solo quando un Doc fornito esplicitamente nei test è privo
    # di lemmi; in produzione load_nlp conserva il lemmatizzatore addestrato.
    lemmas = [(token.lemma_ or token.text).casefold() for token in words]
    negative_parallelism = 0
    for sentence in sentence_spans:
        sentence_text = sentence.text
        negative_parallelism += len(re.findall(r"\bnot\b[^.!?]*?\bbut\b|\brather\s+than\b", sentence_text, flags=re.I))
    markdown_count = len(re.findall(r"(?m)^\s{0,3}(?:#{1,6}\s+|[-*+]\s+|\d+\.\s+)|\*\*[^*\n]+\*\*", text))
    features.update({
        "wiki_ai_vocabulary_per1000": rate(sum(lemma in WIKIPEDIA_AI_LEMMAS for lemma in lemmas), 1000),
        "wiki_emphasis_per1000": rate(_phrase_count(text, WIKIPEDIA_EMPHASIS_PHRASES), 1000),
        "wiki_conclusion_per1000": rate(_phrase_count(text, WIKIPEDIA_CONCLUSION_PHRASES), 1000),
        "wiki_chat_per1000": rate(_phrase_count(text, WIKIPEDIA_CHAT_PHRASES), 1000),
        "wiki_negative_parallelism_per1000": rate(negative_parallelism, 1000),
        "wiki_markdown_per1000": rate(markdown_count, 1000),
    })
    return features


def extract_records(
    records: Iterable[dict[str, Any]],
    model: str = "en_core_web_sm",
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Conserva ogni record di input aggiungendo le feature di ``record['text']``.

    L'elaborazione usa un solo processo e mantiene l'ordine per riproducibilità.
    Il chiamante può conservare la tabella in cache; i testi originali non vengono
    modificati né accorciati.
    """
    records = list(records)
    nlp = load_nlp(model)
    results = []
    for record, doc in zip(records, nlp.pipe((record["text"] for record in records), batch_size=batch_size)):
        results.append({**record, **extract_features(doc, record["text"])})
    return results

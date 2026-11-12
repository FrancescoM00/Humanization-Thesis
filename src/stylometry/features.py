"""Fixed, interpretable document-level style features.

No fitting, dictionary induction, or automatic model/download step occurs here.
Lexical words are spaCy tokens with ``is_alpha`` (casefolded, not lemmatized).
Sentence lengths count those words and omit sentences containing none. POS
proportions instead use *all* non-space, non-punctuation tokens, retaining e.g.
numbers and contraction fragments. Punctuation rates count literal characters
per 100 lexical words; all function-word rates use the lexical-word denominator.

Wikipedia features are deliberately named proxies: their presence neither proves
AI authorship nor operationalizes semantic claims such as unsupported analysis.
The dictionary was specified from the supplied thesis before inspecting effects.
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

# A fixed compact grammatical vocabulary, not terms selected on this dataset.
FUNCTION_WORDS = (
    "a", "an", "the", "and", "or", "but", "if", "because", "although",
    "while", "of", "to", "in", "on", "at", "for", "with", "from", "by",
    "as", "that", "which", "who", "this", "these", "it", "its", "they",
    "their", "he", "she", "we", "our", "you", "your", "i", "my", "is",
    "are", "was", "were", "be", "been", "have", "has", "had", "do",
    "does", "did", "can", "could", "will", "would", "should", "may",
    "might", "must", "not",
)

# Lemmas capture ordinary inflections, but the lexical inventory stays fixed.
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

# Feature -> broad family. Exported so evaluators can run fixed ablations.
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
    # em dash has a Wikipedia role as well; one column avoids duplicating it.
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

# Predeclared models may use this shared column together with the wiki_ columns.
WIKIPEDIA_FEATURES = tuple(
    name for name, group in FEATURE_GROUPS.items() if group == "wikipedia"
) + ("punct_em_dash_per100",)

# Retain these for diagnosis, while an evaluator can exclude length-sensitive
# global diversity measures from its primary style-only model.
LENGTH_SENSITIVE_FEATURES = ("n_words", "n_characters", "n_sentences", "ttr", "hapax_ratio")


def mattr(tokens: Sequence[str], window: int = 50) -> float:
    """Compute exact sliding-window MATTR in O(N) time and O(window) memory.

    Tokens must already be normalized. Return NaN for N < window: a shorter
    window would change the estimand and makes comparisons length-dependent.
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
    """Load the installed English spaCy pipeline; fail instead of downloading.

    Keep the parser for sentence boundaries and lemmatizer for fixed vocabulary
    matching. Only NER, unnecessary for these features, is disabled.
    """
    import spacy

    nlp = spacy.load(model, disable=["ner"])
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        raise ValueError("The spaCy model must provide trained sentence boundaries")
    return nlp


@lru_cache(maxsize=1)
def _readability() -> Any:
    # Check NLTK's corpus *before* textstat can invoke its automatic downloader.
    # Installation/provenance belongs to the setup script, never this extractor.
    import nltk
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
    # textstat 0.7.10 unconditionally calls nltk.download inside get_cmudict.
    # Warm that function's normal cache against the verified local corpus while
    # replacing only the downloader with a local-presence guard. Formulas and
    # the dictionary remain exactly those of textstat en_US. This initialization
    # is single-threaded, matching extract_records's processing contract.
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
    """Extract every feature in ``FEATURE_SCHEMA`` from one parsed document.

    Empty lexical inputs have NaN normalized features (counts remain zero).
    Sentence SD for a one-sentence document is zero. Total punctuation counts
    spaCy punctuation tokens, while individual marks count literal characters;
    those measures need not sum. Regex rhetoric proxies are applied within each
    sentence, preventing unrelated cross-sentence fragments from matching.
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

    # Fall back to literal forms only when an explicitly supplied test Doc lacks
    # lemmata; production load_nlp retains the trained lemmatizer.
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
    """Preserve each input record, adding feature columns for ``record['text']``.

    Processing is single-process and ordered for reproducibility. The caller can
    cache the returned table; original texts are not changed or shortened.
    """
    records = list(records)
    nlp = load_nlp(model)
    results = []
    for record, doc in zip(records, nlp.pipe((record["text"] for record in records), batch_size=batch_size)):
        results.append({**record, **extract_features(doc, record["text"])})
    return results

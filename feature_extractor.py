import re
import math

import numpy as np
import textstat
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# spaCy, NLTK and LanguageTool are imported lazily inside the functions that
# need them. They are only used by the feedback layer, so a missing spaCy model
# or absent Java runtime should not stop the app from starting or from scoring.


# The 17 features the scoring model is fitted on.
# This list must stay identical to FAST_FEATURES in export_model.py.
FAST_FEATURES = [
    "word_count",
    "sentence_count",
    "character_count",
    "paragraph_count",
    "avg_word_length",
    "avg_sentence_length",
    "unique_words",
    "type_token_ratio",
    "stopword_ratio",
    "avg_paragraph_len",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog_score",
    "dale_chall_readability_score",
    "guiraud_index",
    "long_word_ratio",
    "conjunction_ratio",
]

CONJUNCTIONS = {
    "and", "but", "or", "nor", "for", "yet", "so",
    "however", "therefore", "although"
}

_POS_COLUMNS = [
    "noun_count", "verb_count", "adjective_count", "adverb_count",
    "pronoun_count", "determiner_count", "preposition_count",
    "conjunction_count"
]


def clean_text(text):
    """
    Normalise line endings and collapse runs of spaces/tabs.

    Newlines are deliberately preserved: paragraph_count splits on blank lines,
    so collapsing all whitespace would force it to 1 for every essay and break
    both avg_paragraph_len and the model's paragraph feature.
    """
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    return re.sub(r"[ \t]+", " ", text)


def _safe(value):
    try:
        value = float(value)
        return 0.0 if not math.isfinite(value) else value
    except Exception:
        return 0.0


def extract_features(text):
    """
    Reproduces the notebook's FAST feature set.
    These 17 values are the scoring model's inputs.
    """
    text = clean_text(text)

    raw_words = text.split()
    word_count = len(raw_words)

    # The notebook used x.lower().split() for unique_words.
    lower_split = text.lower().split()
    unique_words = len(set(lower_split))

    sentence_count = max(len(re.findall(r"[.!?]+", text)), 1)
    character_count = len(text)
    paragraph_count = max(len(re.split(r"\n\s*\n", text.strip())), 1)

    avg_sentence_length = word_count / sentence_count
    avg_word_length = character_count / word_count if word_count else 0
    avg_paragraph_len = sentence_count / paragraph_count

    stopword_ratio = (
        sum(1 for w in lower_split if w in ENGLISH_STOP_WORDS) / word_count
        if word_count else 0
    )

    type_token_ratio = unique_words / word_count if word_count else 0

    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    gunning_fog_score = textstat.gunning_fog(text)
    dale_chall_readability_score = textstat.dale_chall_readability_score(text)

    guiraud_index = unique_words / math.sqrt(word_count) if word_count else 0

    long_word_ratio = (
        sum(1 for w in raw_words if len(w) > 6) / word_count
        if word_count else 0
    )

    conjunction_ratio = (
        sum(1 for w in lower_split if w in CONJUNCTIONS) / word_count
        if word_count else 0
    )

    return {
        "word_count": _safe(word_count),
        "sentence_count": _safe(sentence_count),
        "character_count": _safe(character_count),
        "paragraph_count": _safe(paragraph_count),
        "avg_word_length": _safe(avg_word_length),
        "avg_sentence_length": _safe(avg_sentence_length),
        "unique_words": _safe(unique_words),
        "type_token_ratio": _safe(type_token_ratio),
        "stopword_ratio": _safe(stopword_ratio),
        "avg_paragraph_len": _safe(avg_paragraph_len),
        "flesch_reading_ease": _safe(flesch_reading_ease),
        "flesch_kincaid_grade": _safe(flesch_kincaid_grade),
        "gunning_fog_score": _safe(gunning_fog_score),
        "dale_chall_readability_score": _safe(dale_chall_readability_score),
        "guiraud_index": _safe(guiraud_index),
        "long_word_ratio": _safe(long_word_ratio),
        "conjunction_ratio": _safe(conjunction_ratio),
    }


# Heavy NLP objects are initialised once rather than once per essay.
_NLP = None
_TOOL = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        # Matches the notebook's POS + NER setup.
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _NLP


def _get_tool():
    global _TOOL
    if _TOOL is None:
        import language_tool_python
        _TOOL = language_tool_python.LanguageTool("en-US")
    return _TOOL


def lexical_features(text, window=50):
    from nltk.tokenize import word_tokenize

    words = [w.lower() for w in word_tokenize(str(text)) if w.isalpha()]

    if not words:
        return {"unique_word_count": 0, "lexical_ttr": 0, "mattr": 0}

    unique_words = len(set(words))
    ttr = unique_words / len(words)

    if len(words) < window:
        mattr = ttr
    else:
        ratios = [
            len(set(words[i:i + window])) / window
            for i in range(len(words) - window + 1)
        ]
        mattr = float(np.mean(ratios))

    return {
        "unique_word_count": unique_words,
        "lexical_ttr": ttr,
        "mattr": mattr,
    }


def _pos_ner_features(text, word_count):
    doc = _get_nlp()(text[:4000])  # same MAX_CHARS=4000 used in the notebook

    counts = {
        "noun_count": 0,
        "verb_count": 0,
        "adjective_count": 0,
        "adverb_count": 0,
        "pronoun_count": 0,
        "determiner_count": 0,
        "preposition_count": 0,
        "conjunction_count": 0,
    }

    for token in doc:
        if token.pos_ == "NOUN":
            counts["noun_count"] += 1
        elif token.pos_ == "VERB":
            counts["verb_count"] += 1
        elif token.pos_ == "ADJ":
            counts["adjective_count"] += 1
        elif token.pos_ == "ADV":
            counts["adverb_count"] += 1
        elif token.pos_ == "PRON":
            counts["pronoun_count"] += 1
        elif token.pos_ == "DET":
            counts["determiner_count"] += 1
        elif token.pos_ == "ADP":
            counts["preposition_count"] += 1
        elif token.pos_ == "CCONJ":
            counts["conjunction_count"] += 1

    entities = doc.ents
    counts.update({
        "entity_count": len(entities),
        "person_count": sum(ent.label_ == "PERSON" for ent in entities),
        "organization_count": sum(ent.label_ == "ORG" for ent in entities),
        "location_count": sum(ent.label_ == "GPE" for ent in entities),
        "date_count": sum(ent.label_ == "DATE" for ent in entities),
    })

    denominator = max(int(word_count), 1)
    for col in _POS_COLUMNS:
        counts[col + "_ratio"] = counts[col] / denominator

    return counts


def _grammar_features(text):
    """
    LanguageTool requires a Java runtime. If it is unavailable the app should
    still return a score and the rest of the feedback, so failures degrade to
    'no grammar data' rather than crashing the request.
    """
    try:
        tool = _get_tool()
        matches = tool.check(str(text))
    except Exception:
        return {"grammar_errors": None, "spelling_error_count": None}

    grammar_errors = len(matches)

    spelling_errors = 0
    for match in matches:
        issue_type = str(getattr(match, "ruleIssueType", "")).lower()
        category = str(getattr(getattr(match, "category", None), "id", "")).lower()

        if "misspelling" in issue_type or "typo" in issue_type:
            spelling_errors += 1
        elif "typo" in category or "spelling" in category:
            spelling_errors += 1

    return {
        "grammar_errors": grammar_errors,
        "spelling_error_count": spelling_errors,
    }


def extract_feedback_features(text):
    """
    Features displayed in the interface and consumed by the feedback rules.
    These are NOT inputs to the scoring model.
    """
    cleaned = clean_text(text)

    # The 17 fast features always succeed; they only need textstat and stdlib.
    fast = extract_features(cleaned)

    # The rest depend on optional heavy components. If NLTK's punkt data or the
    # spaCy model is missing, omit those metrics rather than failing the whole
    # request — the score and the core feedback checks do not rely on them.
    try:
        lexical = lexical_features(cleaned)
    except Exception:
        lexical = {}

    try:
        pos_ner = _pos_ner_features(cleaned, fast["word_count"])
    except Exception:
        pos_ner = {}

    grammar = _grammar_features(cleaned)

    result = {**fast, **lexical, **pos_ner, **grammar}

    word_count = result["word_count"]

    # Alias expected by the feedback rules.
    result["grammar_error_count"] = result["grammar_errors"]
    result["sentence_length"] = result["avg_sentence_length"]

    result["grammar_error_rate"] = (
        result["grammar_errors"] / word_count
        if result["grammar_errors"] is not None and word_count else None
    )
    result["spelling_error_rate"] = (
        result["spelling_error_count"] / word_count
        if result["spelling_error_count"] is not None and word_count else None
    )

    return result

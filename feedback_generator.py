"""
Feedback Generation Module - Person 3
Automated Essay Scoring System

Input contract (agree this with your team):
    essay_analysis = {
        "predicted_score": int,        # from RF model
        "max_score": int,               # e.g. 6
        "word_count": int,
        "sentence_count": int,
        "avg_sentence_length": float,
        "type_token_ratio": float,
        "flesch_reading_ease": float,
        "flesch_kincaid_grade": float,
        "grammar_error_count": int,     # optional, from Person 2
        "spelling_error_count": int,    # optional, from Person 2
    }

Output:
    {
        "score": int,
        "max_score": int,
        "overall_comment": str,
        "strengths": [str, ...],
        "weaknesses": [str, ...],
        "suggestions": [str, ...],
    }
"""

# ---------------------------------------------------------------------------
# 1. THRESHOLDS
# These should ideally be derived from your training data's distribution
# (e.g. quartiles of word_count, TTR, etc. across the ASAP dataset),
# not guessed. Ask Person 2 / whoever has the training df for these stats.
# For now these are reasonable starting points -- tune with real data.
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "word_count": {"low": 150, "high": 350},
    "avg_sentence_length": {"low": 10, "high": 25},
    "type_token_ratio": {"low": 0.35, "high": 0.55},
    "flesch_reading_ease": {"low": 30, "high": 70},   # too low = hard to read
    "grammar_error_rate": {"low": 0.005, "high": 0.02},  # errors per word
    "spelling_error_rate": {"low": 0.002, "high": 0.01},
}


def load_thresholds_from_csv(csv_path, low_q=0.25, high_q=0.75):
    """
    Compute data-driven thresholds from a training CSV instead of guessing.
    Call this once, then update THRESHOLDS with the result, e.g.:

        new_thresholds = load_thresholds_from_csv("features_with_scores.csv")
        THRESHOLDS.update(new_thresholds)

    csv_path must contain columns matching the feature names used in THRESHOLDS
    (word_count, avg_sentence_length, type_token_ratio, flesch_reading_ease,
    and optionally grammar/spelling error counts + word_count to build rates).
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    computed = {}

    simple_features = ["word_count", "avg_sentence_length", "type_token_ratio", "flesch_reading_ease"]
    for feat in simple_features:
        if feat in df.columns:
            computed[feat] = {
                "low": round(df[feat].quantile(low_q), 3),
                "high": round(df[feat].quantile(high_q), 3),
            }

    if "grammar_error_count" in df.columns and "word_count" in df.columns:
        rate = df["grammar_error_count"] / df["word_count"].replace(0, pd.NA)
        computed["grammar_error_rate"] = {
            "low": round(rate.quantile(low_q), 5),
            "high": round(rate.quantile(high_q), 5),
        }

    if "spelling_error_count" in df.columns and "word_count" in df.columns:
        rate = df["spelling_error_count"] / df["word_count"].replace(0, pd.NA)
        computed["spelling_error_rate"] = {
            "low": round(rate.quantile(low_q), 5),
            "high": round(rate.quantile(high_q), 5),
        }

    return computed


def _rate(count, word_count):
    """Convert a raw error count into a per-word rate, safely."""
    if not word_count:
        return 0
    return count / word_count


import random

# ---------------------------------------------------------------------------
# 2. OVERALL COMMENT (score-band framing)
# Multiple variants per band so repeated runs / batches of essays don't all
# read identically. A fixed random.seed based on essay content keeps a given
# essay's feedback reproducible across re-runs (nice for grading consistency).
# ---------------------------------------------------------------------------

OVERALL_COMMENTS = {
    "high": [
        "This is a strong essay overall, showing clear command of the topic and writing conventions.",
        "Overall, this essay is well-executed, with clear ideas and confident use of language.",
        "This is a high-quality piece of writing that demonstrates strong control of the topic.",
    ],
    "good": [
        "This is a solid essay with clear strengths, alongside some areas that would benefit from revision.",
        "Overall this essay does a good job addressing the prompt, though a few areas could be sharpened.",
        "This essay shows good understanding of the topic, with some room to polish specific areas.",
    ],
    "developing": [
        "This essay shows a reasonable attempt but needs meaningful improvement in several areas.",
        "This is a developing piece of writing — the core ideas are present but need stronger execution.",
        "This essay has a foundation to build on, but several areas need attention before it's fully effective.",
    ],
    "low": [
        "This essay needs significant revision across structure, language, and development of ideas.",
        "This essay is not yet meeting the expectations of the task and needs substantial rework.",
        "Considerable revision is needed here, particularly around developing and organizing ideas.",
    ],
}


def _overall_comment(score, max_score, rng):
    pct = score / max_score if max_score else 0
    if pct >= 0.85:
        band = "high"
    elif pct >= 0.6:
        band = "good"
    elif pct >= 0.4:
        band = "developing"
    else:
        band = "low"
    return rng.choice(OVERALL_COMMENTS[band])


# ---------------------------------------------------------------------------
# 3. FEATURE-LEVEL CHECKS
# Each returns (is_strength: bool|None, messages: list[str]|None, suggestions: list[str]|None)
# is_strength = True  -> goes to strengths (one message chosen at random)
# is_strength = False -> goes to weaknesses + suggestions
# is_strength = None  -> feature is in the "fine"/average range, skip it
# ---------------------------------------------------------------------------

def _check_length(data):
    wc = data.get("word_count", 0)
    t = THRESHOLDS["word_count"]
    if wc < t["low"]:
        return False, [
            f"The essay is quite short ({wc} words), which limits how fully ideas can be developed.",
            f"At {wc} words, the essay may not give the argument enough room to develop.",
        ], [
            "Aim to expand key points with more explanation, examples, or evidence to strengthen your argument.",
            "Try adding a supporting example or explaining your reasoning further in each paragraph.",
        ]
    elif wc > t["high"]:
        return True, [
            f"The essay is well-developed in length ({wc} words), giving room to explore ideas fully.",
            f"Good length ({wc} words) allows the essay to develop its points thoroughly.",
        ], None
    return None, None, None


def _check_sentence_variety(data):
    asl = data.get("avg_sentence_length", 0)
    t = THRESHOLDS["avg_sentence_length"]
    if asl < t["low"]:
        return False, [
            f"Sentences are quite short on average ({asl:.1f} words/sentence), which can make writing feel choppy.",
        ], [
            "Try combining related short sentences using conjunctions or subordinate clauses for better flow.",
        ]
    elif asl > t["high"]:
        return False, [
            f"Sentences are quite long on average ({asl:.1f} words/sentence), which can hurt clarity.",
        ], [
            "Break up long sentences into shorter ones to improve readability.",
        ]
    return True, [
        f"Sentence length is well balanced ({asl:.1f} words/sentence), supporting readability.",
        f"Sentences flow at a comfortable, readable length ({asl:.1f} words on average).",
    ], None


def _check_vocabulary(data):
    ttr = data.get("type_token_ratio", 0)
    t = THRESHOLDS["type_token_ratio"]
    if ttr < t["low"]:
        return False, [
            f"Vocabulary usage is fairly repetitive (diversity score: {ttr:.2f}).",
        ], [
            "Try varying your word choice — avoid reusing the same words repeatedly; use synonyms where appropriate.",
        ]
    elif ttr >= t["high"]:
        return True, [
            f"The essay shows strong vocabulary diversity (score: {ttr:.2f}).",
            f"Word choice is varied and precise throughout (diversity score: {ttr:.2f}).",
        ], None
    return None, None, None


def _check_readability(data):
    fre = data.get("flesch_reading_ease", 50)
    t = THRESHOLDS["flesch_reading_ease"]
    if fre < t["low"]:
        return False, [
            "The writing is quite dense and may be difficult to read.",
        ], [
            "Simplify complex sentence structures and consider shorter, clearer phrasing.",
        ]
    elif fre > t["high"]:
        return True, [
            "The essay is clear and easy to read.",
            "The writing flows smoothly and is easy for a reader to follow.",
        ], None
    return None, None, None


def _check_grammar(data):
    if "grammar_error_count" not in data:
        return None, None, None  # feature not available yet from Person 2
    rate = _rate(data["grammar_error_count"], data.get("word_count", 1))
    t = THRESHOLDS["grammar_error_rate"]
    if rate > t["high"]:
        return False, [
            f"There are a notable number of grammar issues ({data['grammar_error_count']} detected).",
        ], [
            "Proofread carefully for subject-verb agreement, verb tense consistency, and sentence fragments.",
        ]
    elif rate < t["low"]:
        return True, [
            "Grammar is consistently strong throughout the essay.",
            "The essay is largely free of grammatical errors.",
        ], None
    return None, None, None


def _check_spelling(data):
    if "spelling_error_count" not in data:
        return None, None, None
    rate = _rate(data["spelling_error_count"], data.get("word_count", 1))
    t = THRESHOLDS["spelling_error_rate"]
    if rate > t["high"]:
        return False, [
            f"There are several spelling errors ({data['spelling_error_count']} detected).",
        ], [
            "Run a spellcheck pass and read the essay aloud to catch missed errors.",
        ]
    return None, None, None


CHECKS = [
    _check_length,
    _check_sentence_variety,
    _check_vocabulary,
    _check_readability,
    _check_grammar,
    _check_spelling,
]


# ---------------------------------------------------------------------------
# 4. MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def generate_feedback(essay_analysis: dict, essay_id=None) -> dict:
    """
    essay_analysis: dict matching the input contract at the top of this file.
    essay_id: optional identifier used to seed the random wording choice, so
              the same essay always gets the same feedback wording if you
              re-run it (useful for grading consistency / grading review),
              while different essays still get varied phrasing.
    Returns structured feedback dict.
    """
    score = essay_analysis.get("predicted_score")
    max_score = essay_analysis.get("max_score", 6)

    seed = hash(essay_id) if essay_id is not None else None
    rng = random.Random(seed)

    strengths, weaknesses, suggestions = [], [], []

    for check in CHECKS:
        is_strength, messages, sugg_options = check(essay_analysis)
        if is_strength is True:
            strengths.append(rng.choice(messages))
        elif is_strength is False:
            weaknesses.append(rng.choice(messages))
            if sugg_options:
                suggestions.append(rng.choice(sugg_options))

    # Fallback so feedback never comes back empty
    if not strengths:
        strengths.append("The essay engages with the assigned topic.")
    if not weaknesses:
        weaknesses.append("No major issues detected in the areas analyzed.")

    return {
        "score": score,
        "max_score": max_score,
        "overall_comment": _overall_comment(score, max_score, rng),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
    }


# ---------------------------------------------------------------------------
# 5. QUICK TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = {
        "predicted_score": 3,
        "max_score": 6,
        "word_count": 140,
        "sentence_count": 9,
        "avg_sentence_length": 15.5,
        "type_token_ratio": 0.42,
        "flesch_reading_ease": 55,
        "flesch_kincaid_grade": 8.5,
        "grammar_error_count": 5,
        "spelling_error_count": 3,
    }

    import json
    print(json.dumps(generate_feedback(sample), indent=2))

import random


# These thresholds are hand-set, not derived from the ASAP data. Consider
# recomputing them as quantiles of the essays that scored 5-6 in your training
# set, so the targets reflect what strong writing actually looked like.
THRESHOLDS = {
    "word_count": {"low": 150, "high": 350},
    "avg_sentence_length": {"low": 10, "high": 25},
    "type_token_ratio": {"low": 0.35, "high": 0.55},
    "flesch_reading_ease": {"low": 30, "high": 70},
    "grammar_error_rate": {"low": 0.005, "high": 0.02},
    "spelling_error_rate": {"low": 0.002, "high": 0.01},
}


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
    """
    Worth sanity-checking against your score distribution: in ASAP 2.0, a score
    of 3 was the most common outcome (36.5% of essays) but maps to pct=0.50 and
    therefore the 'developing' band, and a score of 2 (27.7%) maps to 'low'.
    Around two thirds of essays will land in those two bands.
    """
    pct = score / max_score if max_score else 0

    if pct >= 0.85:
        band = "high"
    elif pct >= 0.60:
        band = "good"
    elif pct >= 0.40:
        band = "developing"
    else:
        band = "low"

    return rng.choice(OVERALL_COMMENTS[band])


def _check_length(data):
    wc = data.get("word_count", 0)
    t = THRESHOLDS["word_count"]

    if wc < t["low"]:
        return False, [
            f"The essay is quite short ({int(wc)} words), which limits how fully ideas can be developed."
        ], [
            "Aim to expand key points with more explanation, examples, or evidence to strengthen your argument."
        ]

    if wc > t["high"]:
        return True, [
            f"The essay is well-developed in length ({int(wc)} words), giving room to explore ideas fully."
        ], None

    return None, None, None


def _check_sentence_variety(data):
    asl = data.get("avg_sentence_length", 0)
    t = THRESHOLDS["avg_sentence_length"]

    if asl < t["low"]:
        return False, [
            f"Sentences are quite short on average ({asl:.1f} words/sentence), which can make writing feel choppy."
        ], [
            "Try combining related short sentences using conjunctions or subordinate clauses for better flow."
        ]

    if asl > t["high"]:
        return False, [
            f"Sentences are quite long on average ({asl:.1f} words/sentence), which can hurt clarity."
        ], [
            "Break up long sentences into shorter ones to improve readability."
        ]

    return True, [
        f"Sentence length is well balanced ({asl:.1f} words/sentence), supporting readability."
    ], None


def _check_vocabulary(data):
    ttr = data.get("type_token_ratio", 0)
    t = THRESHOLDS["type_token_ratio"]

    if ttr < t["low"]:
        return False, [
            f"Vocabulary usage is fairly repetitive (diversity score: {ttr:.2f})."
        ], [
            "Try varying your word choice — avoid reusing the same words repeatedly; use synonyms where appropriate."
        ]

    if ttr >= t["high"]:
        return True, [
            f"The essay shows strong vocabulary diversity (score: {ttr:.2f})."
        ], None

    return None, None, None


def _check_readability(data):
    fre = data.get("flesch_reading_ease", 50)
    t = THRESHOLDS["flesch_reading_ease"]

    if fre < t["low"]:
        return False, [
            "The writing is quite dense and may be difficult to read."
        ], [
            "Simplify complex sentence structures and consider shorter, clearer phrasing."
        ]

    if fre > t["high"]:
        return True, [
            "The essay is clear and easy to read."
        ], None

    return None, None, None


def _check_grammar(data):
    count = data.get("grammar_error_count")
    if count is None:
        count = data.get("grammar_errors")

    # None means LanguageTool was unavailable, which is not the same as zero
    # errors. Stay silent rather than claiming the grammar is clean.
    if count is None:
        return None, None, None

    word_count = data.get("word_count", 0)
    rate = count / word_count if word_count else 0
    t = THRESHOLDS["grammar_error_rate"]

    if rate > t["high"]:
        return False, [
            f"There are a notable number of grammar issues ({int(count)} detected)."
        ], [
            "Proofread carefully for subject-verb agreement, verb tense consistency, and sentence fragments."
        ]

    if rate < t["low"]:
        return True, [
            "Grammar is consistently strong throughout the essay."
        ], None

    return None, None, None


def _check_spelling(data):
    count = data.get("spelling_error_count")
    if count is None:
        return None, None, None

    word_count = data.get("word_count", 0)
    rate = count / word_count if word_count else 0
    t = THRESHOLDS["spelling_error_rate"]

    if rate > t["high"]:
        return False, [
            f"There are several spelling errors ({int(count)} detected)."
        ], [
            "Run a spellcheck pass and read the essay aloud to catch missed errors."
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


def generate_feedback(essay_analysis, essay_id=None):
    score = essay_analysis.get("predicted_score")
    max_score = essay_analysis.get("max_score", 6)

    # random.Random accepts a string seed and hashes it deterministically, so
    # the same essay_id produces the same wording across runs.
    rng = random.Random(essay_id)

    strengths = []
    weaknesses = []
    suggestions = []

    for check in CHECKS:
        is_strength, messages, sugg_options = check(essay_analysis)

        if is_strength is True:
            strengths.append(rng.choice(messages))
        elif is_strength is False:
            weaknesses.append(rng.choice(messages))
            if sugg_options:
                suggestions.append(rng.choice(sugg_options))

    if not strengths:
        strengths.append("The essay engages with the assigned topic.")

    if not weaknesses:
        weaknesses.append("No major issues detected in the areas analyzed.")

    return {
        "overall_comment": _overall_comment(score, max_score, rng),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
    }

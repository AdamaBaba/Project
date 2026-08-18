
# Essay Scoring & NLP Feedback Interface
A user uploads one or more essays and receives a predicted score plus
rule-based written feedback.

## What the application does

1. Upload one or many essays (`.txt`, `.docx`, `.pdf`).
2. Get a predicted score from 1–6.
3. See the linguistic measurements behind it — word and sentence counts,
   readability indices, lexical diversity, POS counts, named entities, grammar
   and spelling issue counts.
4. Receive strengths, weaknesses and suggestions derived from those
   measurements.


**Feedback layer** — spaCy POS/NER, NLTK lexical diversity (TTR, MATTR) and
LanguageTool grammar checking. These are computed for display and for the
feedback rules. **None of them are inputs to the scoring model.** If spaCy or
LanguageTool is unavailable, the app still returns a score and omits the
affected metrics rather than failing.

### Why Gradient Boosting

Test-set comparison from the notebook (24-feature version):

| Model | MSE | MAE | R² | Accuracy | QWK |
|---|---|---|---|---|---|
| Gradient Boosting | 0.4006 | 0.4801 | 0.6239 | 0.6243 | 0.7387 |
| Random Forest | 0.4124 | 0.4867 | 0.6128 | 0.6128 | 0.7331 |
| Ridge | 0.4314 | 0.4995 | 0.5950 | 0.5948 | 0.7157 |

Gradient Boosting wins on every metric, though the margin over Random Forest is
small (about 0.006 QWK). Describe it as the best of the three.
## Setup

```bash
python -m venv .venv
```

Windows: `.venv\Scripts\activate` — macOS/Linux: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```
## Run

```bash
python app.py
```
Then open `http://127.0.0.1:5000`.


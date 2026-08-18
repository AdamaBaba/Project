# AI Essay Assessment System

An AI-based essay scoring and NLP feedback web application developed for the **CS 254 — Introduction to Artificial Intelligence Final Project** at **Ashesi University**.

The system allows users to upload one or more essays and receive a predicted score from **1–6**, together with linguistic statistics and rule-based feedback.

## Group Members

- Adama Baba - 79532028
- Marfo Mavis - 47492028
- Mariam Goge Kessa - 30682027
- Boniphace Benjamin Makoga - 75052028

## Live Application

https://final-project-essay-assessment-proj.vercel.app/

## What the Application Does

Users can:

1. Upload one or multiple essays.
2. Upload `.txt`, `.docx`, or `.pdf` files.
3. Receive a predicted essay score from **1–6**.
4. View linguistic and structural measurements.
5. Receive strengths, weaknesses, and suggestions for improvement.

The system is designed as a **decision-support tool** and not as a replacement for human academic judgment.

## AI Technique

The essay scoring task is treated as a **supervised machine-learning regression problem** because the system predicts a numerical score between 1 and 6.

The project compared:

- Linear/Ridge Regression
- Random Forest Regression
- Gradient Boosting Regression

**Gradient Boosting Regression** was selected as the deployed model because it achieved the best overall performance among the tested models.

## Dataset

The project uses the **Automated Student Assessment Prize (ASAP2)** dataset, which contains student essays and human-assigned scores.

The processed dataset contains approximately **24,728 essay records**.

The score distribution is imbalanced, with scores around 2–4 occurring more frequently than scores at the extremes. This is considered when interpreting the model's performance.

## Features Used for Scoring

The deployed model uses numerical linguistic and structural features extracted from each essay, including:

- Word count
- Character count
- Sentence count
- Average sentence length
- Unique word count
- Vocabulary richness
- Paragraph count
- Other engineered numerical features used by the trained model

The essay is converted into numerical features before being passed to the trained Gradient Boosting model.

## NLP and Feedback Analysis

The application also performs additional NLP analysis to provide feedback.

The feedback layer uses:

- **spaCy** for POS tagging and named-entity recognition
- **NLTK** for lexical diversity measures such as TTR and MATTR
- **LanguageTool** for grammar checking

The system can display:

- Word and sentence counts
- Readability measures
- Lexical diversity
- POS counts
- Named entities
- Grammar errors
- Spelling errors
- Vocabulary information
- Sentence development
- Strengths
- Weaknesses
- Suggestions for improvement

These NLP feedback features are primarily used for analysis and feedback and are **not all inputs to the scoring model**.

If some optional NLP tools are unavailable, the application can still return the essay score while omitting the affected feedback metrics.

## Model Performance

The models were evaluated using MSE, MAE, R², Accuracy, and Quadratic Weighted Kappa (QWK).

| Model | MSE | MAE | R² | Accuracy | QWK |
|---|---:|---:|---:|---:|---:|
| **Gradient Boosting** | **0.4006** | **0.4801** | **0.6239** | **0.6243** | **0.7387** |
| Random Forest | 0.4124 | 0.4867 | 0.6128 | 0.6128 | 0.7331 |
| Ridge | 0.4314 | 0.4995 | 0.5950 | 0.5948 | 0.7157 |

Gradient Boosting achieved the strongest results across the reported metrics and was therefore selected for deployment.

## Application Workflow

    Essay Upload
         ↓
    Text Extraction
         ↓
    Feature Extraction
         ↓
    Feature Scaling
         ↓
    Gradient Boosting Model
         ↓
    Predicted Score (1–6)
         ↓
    NLP Analysis
         ↓
    Feedback Generation
         ↓
    Results Display

## Supported File Types

- `.txt`
- `.pdf`
- `.docx`

Multiple essays can be uploaded at the same time.

## Project Structure

    Essay2/
    │
    ├── app.py
    ├── predictor.py
    ├── document_reader.py
    ├── feature_extractor.py
    ├── feedback.py
    ├── export_model.py
    ├── requirements.txt
    ├── README.md
    ├── vercel.json
    │
    ├── models/
    │   └── trained model files
    │
    ├── static/
    │   ├── style.css
    │   └── app.js
    │
    ├── templates/
    │   └── index.html
    │
    └── notebooks/
        └── project notebooks

## Setup and Running the Application

Create and activate a virtual environment, install the required packages and NLP resources, then run the Flask application.

### Windows

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
    python app.py

Then open:

    http://127.0.0.1:5000

### macOS/Linux

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
    python app.py

Then open:

    http://127.0.0.1:5000

## Deployment

The application is deployed as a Flask web application and made publicly accessible through Vercel.

**Live Demo:**  
https://final-project-essay-assessment-proj.vercel.app/

## Ethical Considerations

### Fairness and Bias

The model is trained using essays and human-assigned scores, so biases present in the training data or human scoring may affect predictions.

The dataset also has an imbalanced score distribution, meaning the model may perform differently across different score ranges.

The system should therefore be used as an assessment aid rather than an autonomous grading system.

### Privacy

Essays may contain personal information. Users should avoid submitting unnecessary personally identifiable information.

A production version would require stronger privacy and security measures such as authentication, encryption, access controls, and appropriate data-retention policies.

### Human Oversight

The predicted score should not be treated as an unquestionable final grade. Lecturers should review the original essay and make the final academic decision.

## Future Improvements

Possible future improvements include:

- More advanced NLP features
- Transformer-based models
- Semantic embeddings
- Better handling of score imbalance
- More extensive hyperparameter tuning
- Fairness evaluation across linguistic groups
- Lecturer dashboards
- Authentication and secure student data storage
- Explainable AI and feature-importance analysis
- Prediction confidence or uncertainty estimates

## Project Status

**Completed Prototype**

The system currently supports:

- Machine-learning essay scoring
- Gradient Boosting prediction
- TXT, PDF, and DOCX uploads
- Multiple essay uploads
- NLP analysis
- Grammar and spelling analysis
- Linguistic statistics
- Automated feedback
- Flask web application
- Vercel deployment
- Public live demo

## Disclaimer

The predicted score is an AI-generated estimate and should not replace human academic assessment. The system is intended to support lecturers and provide additional feedback on student writing.

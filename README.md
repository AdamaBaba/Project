Final Project-Essay Assessment Project

https://final-project-essay-assessment-proj.vercel.app/

AI Essay Assessment System
CS 254 — Introduction to Artificial Intelligence Final Project

Project: AI-Based Essay Assessment System Course: CS 254 — Introduction to Artificial Intelligence Institution: Ashesi University Primary AI Technique: Machine Learning Regression Best Model: Gradient Boosting Regressor Dataset: Automated Student Assessment Prize (ASAP2) Deployment: Flask + Vercel

##GROUP MEMBERS

Adama Baba - 79532028
Marfo Mavis - 47492028
Mariam Goge Kessa - 30682027
Boniphace Benjamin Makoga - 75052028
Live Demo

Essay Assessment System: https://final-project-essay-assessment-proj.vercel.app/

1. Project Overview

The AI Essay Assessment System is a machine learning-based web application designed to assist lecturers and educators in assessing student essays and applications.

The system accepts an essay either by:

Pasting the essay directly into the website
Uploading a .txt file
Uploading a .pdf file
Uploading a .docx file

The system then extracts and analyzes the essay, generates numerical linguistic features, predicts an assessment score from 1 to 6, and provides automated feedback about the essay's strengths, weaknesses, vocabulary, structure, readability, and areas for improvement.

The system is intended as a decision-support tool for lecturers, rather than a replacement for human academic judgment.

2. Real-World Problem

Essay assessment can be time-consuming for lecturers, particularly when large numbers of student scripts must be reviewed. Manual assessment also requires consistent attention to writing quality, organization, vocabulary, and other characteristics.

The project explores whether machine learning can assist with the initial assessment of essays by predicting a score based on measurable linguistic and structural features.

Target Users

The primary stakeholder is:

educators
Lectures

Potential secondary users include:

Teaching assistants
Academic departments
Students seeking formative feedback
Educational institutions
Admission essay applications for University etc.

The system is designed primarily to support lecturers by providing an additional automated assessment signal and structured feedback.

3. Project Objectives; The main objectives are to.
Develop a working AI-based essay assessment prototype.
Train machine-learning models using the ASAP2 essay dataset.
Extract meaningful linguistic and structural features from essays.
Compare multiple regression models.
Select the best-performing model for deployment.
Generate automated feedback that helps identify strengths and weaknesses.
Deploy the working prototype as a publicly accessible web application.
Evaluate the system using appropriate machine-learning metrics.
Analyze ethical risks including bias, fairness, privacy, and transparency.
4. AI Technique

The project uses Supervised machine learning regression.

The essay assessment problem is treated as a regression task because the model predicts a numerical score between 1 and 6.

The project experimented with:

Linear Regression
Random Forest Regression
Gradient Boosting Regression

The models were implemented using Python and scikit-learn.

Selected Model

Gradient Boosting Regression achieved the strongest overall performance among the tested models and was therefore selected for the deployed prototype.

5. Dataset

The project uses the ASAP2 Automated Student Assessment Prize dataset.

The dataset contains student essays together with human-assigned scores.

The processed dataset used in the project contains:

24,728 essay records

The score distribution was:

ScoreNumber of Essays	
1	1,751
2	6,847
3	9,021
4	5,553
5	1,356
6	200

The distribution is not balanced. In particular, scores 2–4 are much more common than scores 1, 5, and 6.

This imbalance is considered in the interpretation of the model's results and ethical analysis.

6. Feature Engineering

Instead of directly feeding the complete essay into the regression model, the system extracts numerical linguistic and structural features.

The deployed model uses:

word_count
character_count
sentence_count
avg_sentence_length
unique_word_count
vocabulary_richness
paragraph_count

These features attempt to represent measurable characteristics of essay structure and language.

The features are created by the essay_analyzer.py service.

Before prediction, the feature values are transformed using the trained regression scaler.

7. Machine Learning Pipeline

The overall pipeline is:

ASAP2 Dataset
      |
      v
Data Cleaning
      |
      v
Feature Engineering
      |
      v
Train/Test Data
      |
      +---------------------+
      |                     |
      v                     v
Linear Regression     Random Forest
      |                     |
      +----------+----------+
                 |
                 v
        Gradient Boosting
                 |
                 v
       Model Evaluation
                 |
                 v
       Select Best Model
                 |
                 v
       Save Trained Model
                 |
                 v
          Flask Website
                 |
                 v
          Essay Submitted
                 |
                 v
       Feature Extraction
                 |
                 v
       Gradient Boosting
                 |
                 v
       Predicted Score 1–6
                 |
                 v
       Automated Feedback

8. Model Evaluation

Three regression models were evaluated using:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R² Score
Results
ModelMAEMSERMSER²				
Linear Regression	0.5255	0.4763	0.6901	0.5561
Random Forest	0.5131	0.4549	0.6744	0.5761
Gradient Boosting	0.5006	0.4326	0.6577	0.5968
Interpretation

Gradient Boosting produced the lowest MAE, MSE, and RMSE and the highest R² among the three tested models.

Its MAE of 0.5006 means that, on average, the predicted score differs from the actual score by approximately half a point.

The R² value of 0.5968 indicates that the model explains approximately 59.68% of the variance in the target scores within the evaluated dataset.

Because Gradient Boosting performed best among the tested models, it was selected for the deployed application.

9. Feedback Generation

In addition to numerical scoring, the system generates structured feedback.

The feedback service analyzes:

Structure
Paragraph organization
Sentence development
Introduction indicators
Conclusion indicators
Vocabulary
Number of unique words
Vocabulary diversity
Frequently repeated words
Readability

The system calculates an approximate Flesch Reading Ease score.

Writing Quality

The system evaluates:

Average sentence length
Vocabulary diversity
Sentence development
Final Feedback

The system produces:

Overall assessment
Strengths
Areas for improvement
Recommendations
Essay statistics
Repeated-word information

The feedback is rule-based and complements the machine-learning score.

10. Web Application

The prototype is implemented using Flask.

Supported input methods
1. Pasted text

The user can paste the complete essay directly into the website.

2. TXT

The system extracts text from .txt files.

3. PDF

The system uses pypdf to extract text from PDF documents.

4. DOCX

The system uses python-docx to extract text from Microsoft Word documents.

11. Example Usage

A typical workflow is:

text

Open the Essay Assessment System.
Paste an essay or upload a supported file.
Click the assessment button.
The application extracts the essay text.
Linguistic features are calculated.
The trained Gradient Boosting model predicts a score.
The feedback service analyzes the essay.
The result page displays the score and feedback.

Example:

Input: Student essay

   ↓


Feature Extraction

word_count character_count sentence_count average_sentence_length unique_word_count vocabulary_richness paragraph_count

   ↓


Gradient Boosting Model

   ↓


Predicted Score

Example: 4.32 / 6

   ↓


Feedback

Strengths Improvements Recommendations Statistics

12. System Architecture

text USER | v Flask Web Interface | +----------+----------+ | | v v Pasted Essay File Upload PDF/DOCX/TXT | | +----------+----------+ | v Text Extraction | v Essay Analyzer | v Feature Creation | v Feature Scaling | v Gradient Boosting Model | v Score Prediction | +----------------+ | | v v Feedback Service Result Page | | +----------------+ | v User Result


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


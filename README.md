# Final Project-Essay Assessment Project
https://final-project-essay-assessment-proj.vercel.app/


# AI Essay Assessment System

## CS 254 — Introduction to Artificial Intelligence Final Project

**Project:** AI-Based Essay Assessment System
**Course:** CS 254 — Introduction to Artificial Intelligence
**Institution:** Ashesi University
**Primary AI Technique:** Machine Learning Regression
**Best Model:** Gradient Boosting Regressor
**Dataset:** Automated Student Assessment Prize (ASAP2)
**Deployment:** Flask + Vercel

##GROUP MEMBERS


### Live Demo

**Essay Assessment System:**
https://final-project-essay-assessment-proj.vercel.app/


# 1. Project Overview

The AI Essay Assessment System is a machine-learning-based web application designed to assist lecturers and educators in assessing student essays and applications.

The system accepts an essay either by:

* Pasting the essay directly into the website
* Uploading a `.txt` file
* Uploading a `.pdf` file
* Uploading a `.docx` file

The system then extracts and analyzes the essay, generates numerical linguistic features, predicts an assessment score from **1 to 6**, and provides automated feedback about the essay's strengths, weaknesses, vocabulary, structure, readability, and areas for improvement.

The system is intended as a decision-support tool for lecturers, rather than a replacement for human academic judgment.

# 2. Real-World Problem

Essay assessment can be time-consuming for lecturers, particularly when large numbers of student scripts must be reviewed. Manual assessment also requires consistent attention to writing quality, organization, vocabulary, and other characteristics.

The project explores whether machine learning can assist with the initial assessment of essays by predicting a score based on measurable linguistic and structural features.

## Target Users

The primary stakeholder is:
- educators
- Lectures

Potential secondary users include:

* Teaching assistants
* Academic departments
* Students seeking formative feedback
* Educational institutions
* Admission essay applications for University etc.

The system is designed primarily to support lecturers by providing an additional automated assessment signal and structured feedback.

# 3. Project Objectives; The main objectives are to.

1. Develop a working AI-based essay assessment prototype.
2. Train machine-learning models using the ASAP2 essay dataset.
3. Extract meaningful linguistic and structural features from essays.
4. Compare multiple regression models.
5. Select the best-performing model for deployment.
6. Generate automated feedback that helps identify strengths and weaknesses.
7. Deploy the working prototype as a publicly accessible web application.
8. Evaluate the system using appropriate machine-learning metrics.
9. Analyze ethical risks including bias, fairness, privacy, and transparency.

# 4. AI Technique
The project uses Supervised machine learning regression.

The essay assessment problem is treated as a regression task because the model predicts a numerical score between 1 and 6.

The project experimented with:

* Linear Regression
* Random Forest Regression
* Gradient Boosting Regression

The models were implemented using Python and scikit-learn.

### Selected Model

Gradient Boosting Regression achieved the strongest overall performance among the tested models and was therefore selected for the deployed prototype.


# 5. Dataset

The project uses the ASAP2 Automated Student Assessment Prize dataset.

The dataset contains student essays together with human-assigned scores.

The processed dataset used in the project contains:

24,728 essay records

The score distribution was:

| Score | Number of Essays |
| ----: | ---------------: |
|     1 |            1,751 |
|     2 |            6,847 |
|     3 |            9,021 |
|     4 |            5,553 |
|     5 |            1,356 |
|     6 |              200 |

The distribution is not balanced. In particular, scores 2–4 are much more common than scores 1, 5, and 6.

This imbalance is considered in the interpretation of the model's results and ethical analysis.

# 6. Feature Engineering

Instead of directly feeding the complete essay into the regression model, the system extracts numerical linguistic and structural features.

The deployed model uses:

* `word_count`
* `character_count`
* `sentence_count`
* `avg_sentence_length`
* `unique_word_count`
* `vocabulary_richness`
* `paragraph_count`

These features attempt to represent measurable characteristics of essay structure and language.

The features are created by the `essay_analyzer.py` service.

Before prediction, the feature values are transformed using the trained regression scaler.

---

# 7. Machine Learning Pipeline

The overall pipeline is:

```text
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
```

---

# 8. Model Evaluation

Three regression models were evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

## Results

| Model                 |        MAE |        MSE |       RMSE |         R² |
| --------------------- | ---------: | ---------: | ---------: | ---------: |
| Linear Regression     |     0.5255 |     0.4763 |     0.6901 |     0.5561 |
| Random Forest         |     0.5131 |     0.4549 |     0.6744 |     0.5761 |
| **Gradient Boosting** | **0.5006** | **0.4326** | **0.6577** | **0.5968** |

## Interpretation

Gradient Boosting produced the lowest MAE, MSE, and RMSE and the highest R² among the three tested models.

Its MAE of **0.5006** means that, on average, the predicted score differs from the actual score by approximately half a point.

The R² value of **0.5968** indicates that the model explains approximately 59.68% of the variance in the target scores within the evaluated dataset.

Because Gradient Boosting performed best among the tested models, it was selected for the deployed application.


# 9. Feedback Generation

In addition to numerical scoring, the system generates structured feedback.

The feedback service analyzes:

### Structure

* Paragraph organization
* Sentence development
* Introduction indicators
* Conclusion indicators

### Vocabulary

* Number of unique words
* Vocabulary diversity
* Frequently repeated words

### Readability

The system calculates an approximate **Flesch Reading Ease** score.

### Writing Quality

The system evaluates:

* Average sentence length
* Vocabulary diversity
* Sentence development

### Final Feedback

The system produces:

* Overall assessment
* Strengths
* Areas for improvement
* Recommendations
* Essay statistics
* Repeated-word information

The feedback is rule-based and complements the machine-learning score.


# 10. Web Application

The prototype is implemented using Flask.

### Supported input methods

#### 1. Pasted text

The user can paste the complete essay directly into the website.

#### 2. TXT

The system extracts text from `.txt` files.

#### 3. PDF

The system uses `pypdf` to extract text from PDF documents.

#### 4. DOCX

The system uses `python-docx` to extract text from Microsoft Word documents.


# 11. Example Usage

A typical workflow is:

text
1. Open the Essay Assessment System.
2. Paste an essay or upload a supported file.
3. Click the assessment button.
4. The application extracts the essay text.
5. Linguistic features are calculated.
6. The trained Gradient Boosting model predicts a score.
7. The feedback service analyzes the essay.
8. The result page displays the score and feedback.


Example:

Input:
Student essay

       ↓

Feature Extraction

word_count
character_count
sentence_count
average_sentence_length
unique_word_count
vocabulary_richness
paragraph_count

       ↓

Gradient Boosting Model

       ↓

Predicted Score

Example: 4.32 / 6

       ↓

Feedback

Strengths
Improvements
Recommendations
Statistics


# 12. System Architecture

text
                    USER
                     |
                     v
             Flask Web Interface
                     |
          +----------+----------+
          |                     |
          v                     v
     Pasted Essay          File Upload
                          PDF/DOCX/TXT
          |                     |
          +----------+----------+
                     |
                     v
              Text Extraction
                     |
                     v
              Essay Analyzer
                     |
                     v
             Feature Creation
                     |
                     v
              Feature Scaling
                     |
                     v
          Gradient Boosting Model
                     |
                     v
              Score Prediction
                     |
                     +----------------+
                     |                |
                     v                v
              Feedback Service    Result Page
                     |                |
                     +----------------+
                              |
                              v
                         User Result


# 13. Project Structure

ext
Final-Project-Essay-Assessment-Project/
│
├── app.py
├── requirements.txt
├── vercel.json
├── README.md
├── .gitignore
│
├── data/
│   └── dataset files
│
├── models/
│   ├── gradient_boosting_model.pkl
│   ├── regression_scaler.pkl
│   └── regression_feature_names.pkl
│
├── services/
│   ├── essay_analyzer.py
│   ├── feedback_service.py
│   ├── prediction_service.py
│   ├── model_training.py
│   ├── model_training_regression.py
│   └── model_evaluation.py
│
├── notebooks/
│   └── experimentation notebooks
│
├── templates/
│   ├── base.html
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── css/
│   └── js/
│
└── tests/




# 14. Installation

## Requirements

The project requires:

* Python 3.x
* pip
* Git

## Clone the repository
bash
git clone https://github.com/boniphacebenja78-lgtm/Final-Project-Essay-Assessment-Project.git
cd Final-Project-Essay-Assessment-Project


## Create a virtual environment

### Windows

bash
python -m venv venv
venv\Scripts\activate


### macOS/Linux

bash
python3 -m venv venv
source venv/bin/activate


## Install dependencies
bash
pip install -r requirements.txt


# 15. Running the Application Locally

Run:
bash
python app.py


The application will normally be available at:

text
http://127.0.0.1:5000


Open the address in a browser.

# 16. Deployment

The application is deployed using Vercel.

Live application:

https://final-project-essay-assessment-proj.vercel.app/


## 17.1 Fairness and Bias

### Potential risk

The model is trained using essays and human-assigned scores. Human scoring can contain inconsistencies or biases.

The dataset also has an imbalanced score distribution, with substantially more essays receiving scores around 2–4 than scores of 1, 5, or 6.

This means the model may perform differently across score ranges.

### Potential impact

Students whose writing styles or linguistic characteristics differ from patterns strongly represented in the training data may receive less accurate predictions.

For example, students who use non-standard English structures may be disadvantaged if the model associates particular language patterns with lower scores.

### Mitigation

* The model is presented as a decision-support tool rather than an autonomous grader.
* Human lecturers should review final grades.
* Multiple evaluation metrics are reported rather than relying on one metric.
* Model limitations are made explicit.
* The score distribution is documented.
* Future versions should evaluate performance across demographic and linguistic groups where ethically and legally appropriate data is available.

---

# 17.2 Privacy

### Risk

Student essays may contain personal or sensitive information. Uploading essays to an online system can therefore create privacy concerns.

### Mitigation

The current prototype processes submitted essays for assessment and does not intentionally use them to retrain the model. Users should avoid including unnecessary personally identifiable information in essays used for testing. For a production educational deployment, additional controls would be required, including:

* Authentication
* Secure storage policies
* Encryption
* Data retention policies
* Access controls
* Institutional privacy compliance


# 17.3 Transparency

Users should understand that the predicted score is generated by a machine-learning model and is not guaranteed to represent a lecturer's final judgment. The system therefore provides information about:

* The model used
* The features used for prediction
* Model evaluation results
* Known limitations
* The role of human review

The system should not present the AI prediction as an unquestionable or objective grade.

# 17.4 Accountability

The lecturer or institution remains responsible for the final academic decision. The AI system should assist with assessment rather than replace the person responsible for grading. A lecturer should be able to review the original essay and disagree with the model's prediction.


# 19. Future Improvements

Future versions could include:

* More advanced NLP features
* Transformer-based language models
* Semantic embeddings
* Essay topic relevance analysis
* Argument and evidence analysis
* Better handling of score imbalance
* Cross-validation
* More extensive hyperparameter tuning
* Human evaluation of generated feedback
* Fairness testing across linguistic groups
* Authentication and role-based access
* Secure student data storage
* Lecturer dashboard
* Historical assessment tracking
* Explainable AI methods such as feature importance
* Calibration of predicted scores
* Confidence intervals or uncertainty estimates

# 22. Conclusion

The AI Essay Assessment System demonstrates how supervised machine learning can be applied to a practical educational problem.

The project developed a complete pipeline from dataset preparation and feature engineering to model training, evaluation, web integration, automated feedback, and public deployment.

Among the evaluated models, Gradient Boosting achieved the strongest performance with:

* **MAE:** 0.5006
* **MSE:** 0.4326
* **RMSE:** 0.6577
* **R²:** 0.5968

The final system is available as a working web application and is intended to support lecturers rather than replace human academic judgment.

The ethical analysis highlights that automated essay assessment can introduce fairness, privacy, transparency, and accountability concerns. Therefore, human oversight remains an important part of responsible use.



# 23. Project Status

**Status: Completed Prototype**

The system currently supports:

* [x] Dataset preparation
* [x] Feature engineering
* [x] Machine-learning model training
* [x] Model comparison
* [x] Model evaluation
* [x] Gradient Boosting model selection
* [x] Flask web application
* [x] TXT upload
* [x] PDF upload
* [x] DOCX upload
* [x] Essay prediction
* [x] Automated feedback
* [x] Ethical analysis
* [x] Requirements file
* [x] GitHub repository
* [x] Vercel deployment
* [x] Public live demo


## Live Application

https://final-project-essay-assessment-proj.vercel.app/

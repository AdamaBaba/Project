from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from feature_extractor import extract_features, extract_feedback_features
from feedback import generate_feedback
from document_reader import read_document


class EssayPredictor:
    """
    Deployment wrapper around the Gradient Boosting model from
    Project_Integrated.ipynb, refitted on the 17 prompt-independent features.

    No scaler is loaded. Gradient Boosting was fitted on unscaled features and
    tree splits are invariant to monotonic rescaling, so applying one would be
    wrong as well as unnecessary.

    Heavy NLP features (POS, NER, lexical diversity, LanguageTool) are computed
    separately for the feedback layer and are NOT inputs to the scoring model.
    """

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model = None
        self.feature_columns = None
        self.medians = {}
        self.score_min = 1
        self.score_max = 6
        self.metadata = {}
        self.model_ready = False
        self._load_model()

    def _load_model(self):
        model_path = self.model_dir / "gb_model.joblib"
        meta_path = self.model_dir / "model_meta.json"

        if not (model_path.exists() and meta_path.exists()):
            return

        self.model = joblib.load(model_path)
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        self.feature_columns = self.metadata["feature_columns"]
        self.medians = self.metadata.get("medians", {})
        self.score_min = self.metadata.get("score_min", 1)
        self.score_max = self.metadata.get("score_max", 6)
        self.model_ready = True

    @property
    def test_metrics(self):
        """Test-set metrics recorded at export time, for display in the UI."""
        return self.metadata.get("test_metrics", {})

    def _build_model_row(self, text):
        """
        Assemble a single-row DataFrame with exactly the columns the model was
        fitted on, in the same order. Missing values fall back to the training
        median, matching the notebook's X_train.fillna(X_train.median()).
        """
        features = extract_features(text)

        row = {}
        for column in self.feature_columns:
            value = features.get(column)
            if value is None or not np.isfinite(value):
                value = self.medians.get(column, 0.0)
            row[column] = float(value)

        return pd.DataFrame([row], columns=self.feature_columns)

    def analyze_text(self, text, essay_id="essay"):
        if not text or not text.strip():
            raise ValueError("The document contains no readable text.")

        model_row = self._build_model_row(text)
        predicted = float(self.model.predict(model_row)[0])
        predicted = float(np.clip(predicted, self.score_min, self.score_max))

        feedback_features = extract_feedback_features(text)
        analysis = {
            **feedback_features,
            "predicted_score": round(predicted, 2),
            "max_score": self.score_max,
        }

        feedback = generate_feedback(analysis, essay_id=essay_id)

        return {
            "predicted_score": round(predicted, 2),
            "rounded_score": int(
                np.clip(np.rint(predicted), self.score_min, self.score_max)
            ),
            "max_score": self.score_max,
            "metrics": feedback_features,
            "feedback": feedback,
        }

    def analyze_file(self, path, filename):
        text = read_document(path)

        if not text.strip():
            raise ValueError("The uploaded document contains no readable text.")

        result = self.analyze_text(text, essay_id=filename)
        result["filename"] = filename
        return result

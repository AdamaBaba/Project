"""
Retrain Gradient Boosting on the 17 prompt-independent features and export it.

WHY RETRAIN
-----------
The Gradient Boosting model in Project_Integrated.ipynb was fitted on 24 inputs:
the 17 fast features PLUS 7 one-hot prompt columns. Every training row had
exactly one prompt column set to 1.

The deployed application accepts essays on ANY topic, so it has no valid value
for those 7 columns. Sending all-zeros would feed the model an input pattern it
never saw during training. The correct fix is to refit the same model on the 17
prompt-independent features only.

HOW TO USE
----------
Run this AFTER the train/test split cells in Project_Integrated.ipynb, so that
`X_train`, `X_val`, `X_test`, `ytrain`, `yval` and `ytest` already exist.

Either:
  (a) paste the body of `retrain_and_export()` into a new notebook cell, or
  (b) upload this file to Colab and run:
          from export_model import retrain_and_export
          retrain_and_export(X_train, ytrain, X_val, yval, X_test, ytest)

Note that no StandardScaler is exported. Gradient Boosting was fitted on
unscaled features in the notebook (`gb_model.fit(X_train, ytrain)`), and trees
are invariant to monotonic rescaling, so a scaler would be both unused and
misleading. Only Ridge needed one.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# The 17 prompt-independent features. Order matters and is preserved on export.
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

SCORE_MIN = 1
SCORE_MAX = 6


def _evaluate(y_true, y_pred, score_min=SCORE_MIN, score_max=SCORE_MAX):
    """Same metric set the notebook used, with predictions clipped to the score range."""
    rounded = np.clip(np.round(y_pred), score_min, score_max).astype(int)
    return {
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Accuracy": float(accuracy_score(y_true, rounded)),
        "QWK": float(cohen_kappa_score(y_true, rounded, weights="quadratic")),
    }


def retrain_and_export(
    X_train,
    ytrain,
    X_val,
    yval,
    X_test,
    ytest,
    output_dir="essay_scoring_app/models",
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Drop the 7 prompt one-hot columns; keep the 17 features in a fixed order.
    train_17 = X_train[FAST_FEATURES]
    val_17 = X_val[FAST_FEATURES]
    test_17 = X_test[FAST_FEATURES]

    # Identical hyperparameters to the notebook's gb_model.
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    model.fit(train_17, ytrain)

    val_metrics = _evaluate(yval, model.predict(val_17))
    test_metrics = _evaluate(ytest, model.predict(test_17))

    print("Gradient Boosting, 17 prompt-independent features")
    print("  Validation:", {k: round(v, 4) for k, v in val_metrics.items()})
    print("  Test:      ", {k: round(v, 4) for k, v in test_metrics.items()})
    print()
    print("Compare these against the 24-feature results in your report.")
    print("Some drop is expected: prompt mean scores ranged 2.48-3.20, so the")
    print("prompt columns were carrying real signal. Record the actual numbers.")
    print()

    medians = train_17.median()

    metadata = {
        "model_type": "GradientBoostingRegressor",
        "model_params": model.get_params(),
        "feature_columns": FAST_FEATURES,
        "medians": {k: float(v) for k, v in medians.items()},
        "score_min": SCORE_MIN,
        "score_max": SCORE_MAX,
        "n_train": int(len(train_17)),
        "sklearn_version": sklearn.__version__,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "notes": (
            "Trained on ASAP 2.0 source-based essays, scores 1-6. The 7 prompt "
            "one-hot columns used during notebook experimentation were dropped "
            "so the model can score essays on arbitrary topics."
        ),
    }

    joblib.dump(model, output / "gb_model.joblib")
    (output / "model_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Exported:")
    print(" ", output / "gb_model.joblib")
    print(" ", output / "model_meta.json")
    print()
    print(f"scikit-learn version used: {sklearn.__version__}")
    print("Pin this exact version in requirements.txt or joblib.load() may")
    print("warn or fail when the app runs on a different machine.")

    return model, metadata


if __name__ == "__main__":
    print(__doc__)

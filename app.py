from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from predictor import EssayPredictor
from pathlib import Path
import tempfile
import os

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB per request

ALLOWED_EXTENSIONS = {"txt", "docx", "pdf"}

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "gb_model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
predictor = EssayPredictor(MODEL_DIR)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_ready=predictor.model_ready,
        test_metrics=predictor.test_metrics,
    )


@app.post("/analyze")
def analyze():
    if not predictor.model_ready:
        return jsonify({
            "error": (
                "The trained model has not been exported yet. "
                "Run retrain_and_export() from export_model.py in the notebook first."
            )
        }), 503

    uploaded = request.files.getlist("files")
    if not uploaded:
        return jsonify({"error": "Please upload at least one essay."}), 400

    results = []
    errors = []

    for file in uploaded:
        filename = secure_filename(file.filename or "")
        if not filename:
            continue

        if not allowed_file(filename):
            errors.append(f"{filename}: unsupported file type.")
            continue

        temp_path = None
        try:
            suffix = Path(filename).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                file.save(tmp.name)
                temp_path = tmp.name

            results.append(predictor.analyze_file(temp_path, filename))

        except Exception as exc:
            errors.append(f"{filename}: {exc}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    return jsonify({
        "results": results,
        "errors": errors,
        "count": len(results),
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

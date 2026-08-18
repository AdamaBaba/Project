from flask import (
    Flask,
    render_template,
    request
)

import os
import tempfile

from pypdf import PdfReader
from docx import Document
from PIL import Image
import pytesseract

from services.prediction_service import predict_score
from services.feedback_service import generate_feedback


# =========================================================
# Flask application
# =========================================================

app = Flask(__name__)


# =========================================================
# Configuration
# =========================================================

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# =========================================================
# Home page
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# Extract text from TXT
# =========================================================

def extract_txt(file):

    content = file.read()

    try:

        return content.decode(
            "utf-8"
        )

    except UnicodeDecodeError:

        return content.decode(
            "latin-1"
        )


# =========================================================
# Extract text from PDF
# =========================================================

def extract_pdf(file):

    text = []

    reader = PdfReader(
        file
    )

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text.append(
                page_text
            )

    return "\n\n".join(
        text
    )


# =========================================================
# Extract text from DOCX
# =========================================================

def extract_docx(file):

    document = Document(
        file
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            paragraphs.append(
                text
            )

    return "\n\n".join(
        paragraphs
    )


# =========================================================
# Extract text from image
# =========================================================

def extract_image(file):

    image = Image.open(
        file
    )

    text = pytesseract.image_to_string(
        image
    )

    return text


# =========================================================
# Extract essay text from uploaded file
# =========================================================

def extract_essay_from_file(file):

    if not file:

        return ""


    filename = (
        file.filename or ""
    ).lower()


    if not filename:

        return ""


    extension = os.path.splitext(
        filename
    )[1]


    # -----------------------------------------------------
    # TXT
    # -----------------------------------------------------

    if extension == ".txt":

        return extract_txt(
            file
        )


    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    if extension == ".pdf":

        return extract_pdf(
            file
        )


    # -----------------------------------------------------
    # DOCX
    # -----------------------------------------------------

    if extension == ".docx":

        return extract_docx(
            file
        )


    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    if extension in [
        ".png",
        ".jpg",
        ".jpeg"
    ]:

        return extract_image(
            file
        )


    raise ValueError(
        "Unsupported file type. "
        "Please upload a PDF, DOCX, TXT, JPG, JPEG, or PNG file."
    )


# =========================================================
# Assess essay
# =========================================================

@app.route(
    "/assess",
    methods=["POST"]
)
def assess():

    essay = request.form.get(
        "essay",
        ""
    ).strip()


    # =====================================================
    # STEP 1 — CHECK NORMAL PASTED ESSAY
    # =====================================================

    uploaded_file = request.files.get(
        "essay_file"
    )


    camera_file = request.files.get(
        "camera_file"
    )


    try:

        # =================================================
        # STEP 2 — HANDLE COMPUTER UPLOAD
        # =================================================

        if uploaded_file and uploaded_file.filename:

            uploaded_text = (
                extract_essay_from_file(
                    uploaded_file
                )
            )

            if uploaded_text:

                essay = uploaded_text.strip()


        # =================================================
        # STEP 3 — HANDLE CAMERA IMAGE
        # =================================================

        elif camera_file and camera_file.filename:

            camera_text = (
                extract_essay_from_file(
                    camera_file
                )
            )

            if camera_text:

                essay = camera_text.strip()


        # =================================================
        # STEP 4 — VALIDATE ESSAY
        # =================================================

        if not essay:

            return render_template(
                "index.html",
                error=(
                    "Please paste an essay, "
                    "upload an essay, or take a picture "
                    "of an essay."
                )
            )


        # =================================================
        # STEP 5 — PREDICT SCORE
        # =================================================

        score = predict_score(
            essay
        )


        # =================================================
        # STEP 6 — GENERATE FEEDBACK
        # =================================================

        feedback = generate_feedback(
            essay,
            score
        )


        # =================================================
        # STEP 7 — DISPLAY RESULT
        # =================================================

        return render_template(
            "result.html",

            score=score,

            essay=essay,

            feedback=feedback
        )


    except Exception as error:

        return render_template(
            "index.html",

            error=f"Error assessing essay: {error}"
        )


# =========================================================
# Run Flask
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
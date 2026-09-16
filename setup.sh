#!/usr/bin/env bash
# Set up a local virtual environment and install dependencies.
set -euo pipefail

PYTHON="${PYTHON:-python}"

"$PYTHON" -m venv .venv --system-site-packages

# Activate for this script (POSIX); on Windows run the .bat equivalent.
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt

# spaCy English model used by the resume parser.
python -m spacy download en_core_web_sm

echo "Setup complete."
echo "System dependencies also required:"
echo "  - Tesseract OCR (pytesseract) for scanned resumes"
echo "  - Poppler (pdf2image) for PDF rasterisation"
echo "  - Google Chrome + matching ChromeDriver (selenium) for scraping"

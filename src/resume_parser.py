"""Resume text extraction and parsing.

Ported from the ``EnhancedJobMatcher`` parsing half of ``finalv2.py``. The
Streamlit calls (``st.spinner``/``st.error``) were removed so the parser is
usable outside a Streamlit run and unit-testable. spaCy and the OCR toolchain
are only imported/loaded on first use.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from pypdf import PdfReader

import config

logger = logging.getLogger(__name__)

EXPERIENCE_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:\+\s*)?years?\b", re.IGNORECASE)


def normalize_experience(value: Any) -> float:
    """Normalise a scraped experience value to a float (years).

    Handles strings like ``"1-4 Yrs"``, ``"3 years"`` and plain numbers.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        matches = re.findall(r"\d+(?:\.\d+)?", value)
        return float(matches[0]) if matches else 0.0
    return 0.0


def clean_text(text: str) -> str:
    """Lower-case and strip punctuation/extra whitespace for embedding."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class ResumeParser:
    """Extract raw text and structured fields from a resume PDF."""

    def __init__(self, nlp: Any = None, ocr_lang: str = "eng", ocr_config: str = "--psm 1") -> None:
        self._nlp = nlp
        self.ocr_lang = ocr_lang
        self.ocr_config = ocr_config

    @property
    def nlp(self) -> Any:
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(config.SPACY_MODEL)
        return self._nlp

    # -- text extraction ---------------------------------------------------
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text directly, falling back to OCR for scanned PDFs."""
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - any reader failure -> OCR
            logger.warning("Direct PDF extraction failed: %s", exc)

        if len(text.strip()) < 100:
            logger.info("Limited text extracted; switching to OCR.")
            return self._extract_text_with_ocr(pdf_path)
        return text

    def _extract_text_with_ocr(self, pdf_path: str) -> str:
        from pdf2image import convert_from_path
        import pytesseract

        text_parts: List[str] = []
        for image in convert_from_path(pdf_path):
            text_parts.append(
                pytesseract.image_to_string(
                    image, lang=self.ocr_lang, config=self.ocr_config
                )
            )
        return "\n".join(text_parts)

    # -- structured fields -------------------------------------------------
    def extract_resume_details(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """Return parsed resume info, or ``None`` when no text was found."""
        text = self.extract_text_from_pdf(pdf_path)
        if not text or not text.strip():
            return None

        doc = self.nlp(text)
        skills = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "PRODUCT")]

        matches = EXPERIENCE_PATTERN.findall(text)
        experience = max((float(y) for y in matches), default=0.0)

        return {
            "full_text": text,
            "skills": list({s.strip() for s in skills if s.strip()}),
            "experience": experience,
            "extracted_text": clean_text(text),
        }

    def extract_location_from_text(self, text: str) -> Optional[str]:
        """Return the first GPE/LOC entity found in ``text``, if any."""
        for ent in self.nlp(text).ents:
            if ent.label_ in ("GPE", "LOC"):
                return ent.text
        return None

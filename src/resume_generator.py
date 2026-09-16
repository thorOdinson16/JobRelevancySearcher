"""Resume PDF generation.

Ported from the PDF-generator half of ``finalv2.py``. Fixes:

* user-provided text is XML-escaped before being handed to reportlab, so
  characters like ``&``, ``<`` and ``>`` no longer corrupt the PDF;
* an uploaded photo (file-like object or bytes) is buffered safely;
* missing optional collections never raise.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate

KEYWORD_DESCRIPTIONS: Dict[str, str] = {
    "leadership": "Demonstrated exceptional leadership skills by managing diverse teams and driving projects to success.",
    "data analysis": "Proficient in analyzing complex datasets to derive actionable insights and support decision-making.",
    "problem-solving": "Skilled in identifying challenges and implementing innovative solutions to achieve objectives.",
    "communication": "Exceptional verbal and written communication skills, ensuring effective collaboration across teams.",
    "project management": "Expert in planning, executing, and delivering projects on time and within budget.",
    "teamwork": "Adept at fostering collaboration and building strong working relationships within cross-functional teams.",
    "time management": "Highly organized and efficient, able to prioritize tasks and meet deadlines consistently.",
    "critical thinking": "Experienced in evaluating situations to make informed and strategic decisions.",
    "customer service": "Committed to providing outstanding customer support and ensuring client satisfaction.",
    "adaptability": "Quick to adapt to new environments and technologies, ensuring seamless transitions and productivity.",
    "technical expertise": "Proficient in utilizing advanced tools and software to achieve technical objectives.",
    "creativity": "Innovative thinker with a strong ability to develop original ideas and approaches.",
    "marketing": "Skilled in developing and executing effective marketing strategies to drive brand awareness and growth.",
    "sales": "Proven track record of exceeding sales targets and building lasting client relationships.",
    "negotiation": "Strong negotiator capable of securing favorable outcomes in contracts and agreements.",
    "training": "Experienced in designing and delivering training programs to enhance team skills and knowledge.",
    "research": "Highly skilled in conducting thorough research to support data-driven decisions and innovations.",
    "organization": "Detail-oriented with excellent organizational skills to manage multiple projects and priorities effectively.",
    "financial analysis": "Expert in analyzing financial data to optimize budgets, investments, and growth opportunities.",
    "risk management": "Proficient in identifying and mitigating risks to ensure business continuity and success.",
    "innovation": "Consistently driving innovation through creative problem-solving and process improvements.",
    "presentation": "Experienced in creating and delivering impactful presentations tailored to diverse audiences.",
    "networking": "Skilled at building and maintaining professional networks to drive partnerships and opportunities.",
    "analytics": "Advanced skills in utilizing data analytics to uncover trends and inform strategies.",
    "coding": "Proficient in programming languages such as Python, Java, or C++, developing efficient solutions to complex problems.",
    "strategic planning": "Expertise in formulating and executing strategic plans to achieve organizational goals.",
    "digital marketing": "Knowledgeable in SEO, social media, and digital advertising to maximize online visibility and engagement.",
    "conflict resolution": "Skilled at mediating disputes and fostering positive outcomes in challenging situations.",
    "operations management": "Experienced in streamlining processes and improving efficiency to optimize operational performance.",
}


def generate_description(keyword: str) -> str:
    """Return a canned description for a competency keyword."""
    return KEYWORD_DESCRIPTIONS.get(
        keyword.lower(), f"Description not available for '{keyword}'."
    )


def _p(value: Any) -> str:
    """Escape arbitrary user text for safe inclusion in a Paragraph."""
    return escape(str(value if value is not None else ""))


def _as_flowable(photo: Any) -> Optional[Image]:
    """Convert an uploaded photo (file-like/bytes) into a reportlab Image."""
    if not photo:
        return None
    if isinstance(photo, (bytes, bytearray)):
        stream: Any = io.BytesIO(photo)
    elif hasattr(photo, "getvalue"):
        stream = io.BytesIO(photo.getvalue())
    elif hasattr(photo, "read"):
        stream = io.BytesIO(photo.read())
    else:
        stream = photo
    try:
        if hasattr(stream, "seek"):
            stream.seek(0)
        return Image(stream, width=100, height=100)
    except Exception:  # noqa: BLE001 - a bad image should not break the resume
        return None


def create_pdf(data: Dict[str, Any], file_name: str = "resume_with_photo.pdf") -> str:
    """Generate a resume PDF and return the written path."""
    doc = SimpleDocTemplate(file_name, pagesize=letter)
    styles = getSampleStyleSheet()
    elements: List[Any] = []

    elements.append(Paragraph(f"<b>{_p(data.get('name'))}</b>", styles["Title"]))
    elements.append(
        Paragraph(
            f"<br/><b>Contact</b>: {_p(data.get('phone'))} | {_p(data.get('email'))}<br/>"
            f"<b>LinkedIn</b>: {_p(data.get('linkedin'))}<br/>"
            f"<b>Location</b>: {_p(data.get('location'))}",
            styles["Normal"],
        )
    )

    photo = _as_flowable(data.get("photo"))
    if photo is not None:
        elements.append(photo)

    elements.append(Paragraph("<br/><b>CAREER OBJECTIVE</b>", styles["Heading2"]))
    elements.append(Paragraph(_p(data.get("career_objective")), styles["Normal"]))

    elements.append(Paragraph("<br/><b>EDUCATION</b>", styles["Heading2"]))
    for edu in data.get("education", []) or []:
        elements.append(
            Paragraph(
                f"<b>{_p(edu.get('degree'))}</b> - {_p(edu.get('institution'))} ({_p(edu.get('year'))})",
                styles["Normal"],
            )
        )

    elements.append(Paragraph("<br/><b>CORE COMPETENCIES</b>", styles["Heading2"]))
    for comp in data.get("core_competencies", []) or []:
        elements.append(Paragraph(f"- {_p(generate_description(comp))}", styles["Normal"]))

    elements.append(Paragraph("<br/><b>INTERNSHIPS</b>", styles["Heading2"]))
    for internship in data.get("internships", []) or []:
        elements.append(
            Paragraph(
                f"<b>{_p(internship.get('role'))}</b> at {_p(internship.get('organization'))} "
                f"({_p(internship.get('duration'))})<br/>{_p(internship.get('description'))}",
                styles["Normal"],
            )
        )

    elements.append(Paragraph("<br/><b>SKILLS</b>", styles["Heading2"]))
    elements.append(
        Paragraph(
            f"<b>Hard Skills</b>: {_p(', '.join(data.get('hard_skills', []) or []))}",
            styles["Normal"],
        )
    )
    elements.append(
        Paragraph(
            f"<b>Soft Skills</b>: {_p(', '.join(data.get('soft_skills', []) or []))}",
            styles["Normal"],
        )
    )

    elements.append(Paragraph("<br/><b>ACHIEVEMENTS</b>", styles["Heading2"]))
    for achievement in data.get("achievements", []) or []:
        elements.append(Paragraph(f"- {_p(achievement)}", styles["Normal"]))

    elements.append(Paragraph("<br/><b>CERTIFICATIONS</b>", styles["Heading2"]))
    for cert in data.get("certifications", []) or []:
        elements.append(Paragraph(f"- {_p(cert)}", styles["Normal"]))

    doc.build(elements)
    return file_name

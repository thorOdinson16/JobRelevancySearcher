import os

from src.resume_generator import create_pdf, generate_description


def test_generate_description_known_keyword():
    assert "leadership" in generate_description("Leadership").lower()


def test_generate_description_unknown_keyword():
    assert "not available" in generate_description("underwater basket weaving")


def test_create_pdf_escapes_special_characters(tmp_path):
    out = os.path.join(tmp_path, "resume.pdf")
    data = {
        "name": "A & B <Corp>",
        "phone": "+1-555",
        "email": "a@example.com",
        "linkedin": "linkedin.com/in/a",
        "location": "Mumbai",
        "career_objective": "Build things <fast> & reliably",
        "education": [{"degree": "B.Tech", "institution": "IIT & Co", "year": "2020"}],
        "core_competencies": ["leadership"],
        "internships": [],
        "hard_skills": ["Python", "C++"],
        "soft_skills": ["communication"],
        "achievements": ["Won 1st place & more"],
        "certifications": [],
        "photo": None,
    }
    path = create_pdf(data, out)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_create_pdf_handles_missing_optional_fields(tmp_path):
    out = os.path.join(tmp_path, "resume.pdf")
    create_pdf({"name": "Only Name"}, out)
    assert os.path.exists(out)

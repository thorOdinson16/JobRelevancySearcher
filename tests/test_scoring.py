import numpy as np

from src.matcher import JobMatcher

USER = (19.0760, 72.8777)  # Mumbai
NEAR = (19.0896, 72.8656)
FAR = (28.6139, 77.2090)  # Delhi


def test_location_score_same_point_is_one():
    assert JobMatcher.calculate_location_score(USER, USER) == 1.0


def test_location_score_zero_when_missing_coords():
    assert JobMatcher.calculate_location_score(None, NEAR) == 0.0
    assert JobMatcher.calculate_location_score(USER, None) == 0.0


def test_location_score_decays_with_distance():
    near_score = JobMatcher.calculate_location_score(USER, NEAR)
    far_score = JobMatcher.calculate_location_score(USER, FAR)
    assert 0.0 <= far_score < near_score <= 1.0


def _matcher_without_model() -> JobMatcher:
    matcher = JobMatcher()
    matcher.get_embedding = lambda text: np.ones((1, 4), dtype=float)  # type: ignore[assignment]
    return matcher


def test_similarity_rewards_skill_overlap():
    matcher = _matcher_without_model()
    resume = {"extracted_text": "python sql", "skills": ["Python", "SQL"], "experience": 5.0}
    job = {
        "id": 1,
        "title": "Data Engineer",
        "company": "Acme",
        "location": "Mumbai",
        "experience": "2-4 Yrs",
        "skills": ["python", "spark"],
        "link": "http://example.com/1",
    }
    result = matcher.calculate_similarity(resume, job)
    assert result["matching_skills"] == ["python"]
    assert result["has_matching_skills"] is True
    assert 0.0 <= result["overall_match"] <= 100.0


def test_similarity_without_location_has_no_location_fields():
    matcher = _matcher_without_model()
    resume = {"extracted_text": "python", "skills": [], "experience": 1.0}
    job = {"title": "Dev", "company": "Co", "skills": [], "experience": "0", "link": "#"}
    result = matcher.calculate_similarity(resume, job, location_weight=0.0)
    assert result["location_score"] is None
    assert result["distance_map"] is None

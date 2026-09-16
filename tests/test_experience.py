from src.resume_parser import clean_text, normalize_experience


def test_normalize_plain_number():
    assert normalize_experience(3) == 3.0
    assert normalize_experience(2.5) == 2.5


def test_normalize_range_string_uses_lower_bound():
    assert normalize_experience("1-4 Yrs") == 1.0
    assert normalize_experience("3-5 years") == 3.0


def test_normalize_single_year_string():
    assert normalize_experience("7 years") == 7.0
    assert normalize_experience("5+ Yrs") == 5.0


def test_normalize_invalid_values():
    assert normalize_experience("Not specified") == 0.0
    assert normalize_experience(None) == 0.0
    assert normalize_experience("") == 0.0


def test_clean_text_strips_punctuation_and_case():
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("  Python   &  SQL  ") == "python sql"

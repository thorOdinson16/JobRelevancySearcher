"""Canonical field names shared by the scraper and the matcher.

Keeping the keys in one place prevents the two modules drifting apart (the
original code wrote ``Required_Skills`` but read ``Skills``, which silently
disabled skill matching).
"""

from __future__ import annotations

JOB_TITLE = "title"
JOB_COMPANY = "company"
JOB_LOCATION = "location"
JOB_EXPERIENCE = "experience"
JOB_SALARY = "salary"
JOB_LINK = "link"
JOB_SKILLS = "skills"
JOB_DESCRIPTION = "description"
JOB_ROLE = "role"
JOB_INDUSTRY = "industry"
JOB_EMPLOYMENT_TYPE = "employment_type"
JOB_STATUS = "status"
JOB_ERROR = "error"
JOB_SCRAPED_AT = "scraped_at"

# Keys accepted by :func:`db.upsert_jobs`.
JOB_FIELDS = (
    JOB_TITLE,
    JOB_COMPANY,
    JOB_LOCATION,
    JOB_EXPERIENCE,
    JOB_SALARY,
    JOB_LINK,
    JOB_SKILLS,
    JOB_DESCRIPTION,
    JOB_ROLE,
    JOB_INDUSTRY,
    JOB_EMPLOYMENT_TYPE,
    JOB_STATUS,
    JOB_ERROR,
    JOB_SCRAPED_AT,
)

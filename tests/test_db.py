from db import fetch_jobs, upsert_jobs


def test_upsert_inserts_new_jobs(sqlite_session):
    jobs = [
        {"link": "http://example.com/1", "title": "Engineer", "company": "Acme", "skills": ["python"]},
        {"link": "http://example.com/2", "title": "Analyst", "company": "Beta"},
    ]
    assert upsert_jobs(sqlite_session, jobs) == 2
    assert len(fetch_jobs(sqlite_session)) == 2


def test_upsert_updates_existing_job_instead_of_duplicating(sqlite_session):
    upsert_jobs(sqlite_session, [{"link": "http://example.com/1", "title": "Old", "company": "Acme"}])
    upsert_jobs(sqlite_session, [{"link": "http://example.com/1", "title": "New", "company": "Acme"}])

    rows = fetch_jobs(sqlite_session)
    assert len(rows) == 1
    assert rows[0].title == "New"


def test_upsert_skips_jobs_without_link(sqlite_session):
    assert upsert_jobs(sqlite_session, [{"title": "No link"}]) == 0
    assert fetch_jobs(sqlite_session) == []


def test_to_dict_round_trip(sqlite_session):
    upsert_jobs(sqlite_session, [{"link": "http://example.com/1", "title": "Engineer", "skills": ["a", "b"]}])
    job = fetch_jobs(sqlite_session)[0]
    data = job.to_dict()
    assert data["title"] == "Engineer"
    assert data["skills"] == ["a", "b"]
    assert data["link"] == "http://example.com/1"

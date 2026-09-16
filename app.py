"""Streamlit entrypoint for JobRelevancySearcher.

Run with::

    streamlit run app.py

Two modes are available from the sidebar: a resume PDF generator and the
semantic job matcher backed by PostgreSQL.
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st

# Allow Streamlit secrets to configure the DB before any engine is created.
try:
    for _key in ("DATABASE_URL", "PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE"):
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:  # noqa: BLE001 - no secrets file is fine
    pass


def run_pdf_generator() -> None:
    from src.resume_generator import create_pdf

    st.title("Advanced Resume Generator with Keywords and Photo")
    st.write("Fill in the details below to generate your resume with a photo and keyword descriptions.")

    name = st.text_input("Full Name")
    phone = st.text_input("Phone Number")
    email = st.text_input("Email Address")
    linkedin = st.text_input("LinkedIn Profile URL")
    location = st.text_input("Location")
    photo = st.file_uploader("Upload a photo for your resume", type=["jpg", "png"])

    st.subheader("Career Objective")
    career_objective = st.text_area("Write your career objective")

    st.subheader("Education")
    num_education = st.number_input("Number of Education Entries", min_value=1, max_value=5, step=1, value=1)
    education = []
    for i in range(int(num_education)):
        st.write(f"Education Entry {i + 1}")
        education.append(
            {
                "degree": st.text_input(f"Degree (Education {i + 1})", key=f"degree_{i}"),
                "institution": st.text_input(f"Institution (Education {i + 1})", key=f"institution_{i}"),
                "year": st.text_input(f"Year (Education {i + 1})", key=f"year_{i}"),
            }
        )

    st.subheader("Core Competencies")
    core_competencies = st.text_area("List core competencies (comma-separated)").split(",")

    st.subheader("Internships")
    num_internships = st.number_input("Number of Internships", min_value=0, max_value=5, step=1, value=1)
    internships = []
    for i in range(int(num_internships)):
        st.write(f"Internship {i + 1}")
        internships.append(
            {
                "role": st.text_input(f"Role (Internship {i + 1})", key=f"role_{i}"),
                "organization": st.text_input(f"Organization (Internship {i + 1})", key=f"organization_{i}"),
                "duration": st.text_input(f"Duration (Internship {i + 1})", key=f"duration_{i}"),
                "description": st.text_area(f"Description (Internship {i + 1})", key=f"description_{i}"),
            }
        )

    st.subheader("Skills")
    hard_skills = st.text_input("List hard skills (comma-separated)").split(",")
    soft_skills = st.text_input("List soft skills (comma-separated)").split(",")

    st.subheader("Achievements")
    achievements = st.text_area("List achievements (one per line)").split("\n")

    st.subheader("Certifications")
    certifications = st.text_area("List certifications (one per line)").split("\n")

    if st.button("Generate Resume"):
        if name and phone and email and linkedin and location and career_objective:
            data = {
                "name": name,
                "phone": phone,
                "email": email,
                "linkedin": linkedin,
                "location": location,
                "career_objective": career_objective,
                "education": education,
                "core_competencies": [c.strip() for c in core_competencies if c.strip()],
                "internships": internships,
                "hard_skills": [s.strip() for s in hard_skills if s.strip()],
                "soft_skills": [s.strip() for s in soft_skills if s.strip()],
                "achievements": [a.strip() for a in achievements if a.strip()],
                "certifications": [c.strip() for c in certifications if c.strip()],
                "photo": photo,
            }
            with tempfile.TemporaryDirectory() as tmp_dir:
                pdf_path = os.path.join(tmp_dir, "resume_with_photo.pdf")
                create_pdf(data, pdf_path)
                with open(pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
            st.success("Resume generated successfully!")
            st.download_button(
                label="Download Resume",
                data=pdf_bytes,
                file_name="resume_with_photo.pdf",
                mime="application/pdf",
            )
        else:
            st.error("Please fill in all required fields.")


def run_job_matcher() -> None:
    from db import ensure_database, fetch_jobs, get_session, init_db
    from src.matcher import JobMatcher

    st.title("Enhanced Resume Job Matcher")
    st.write("Upload your resume and find matching jobs with location-based scoring")

    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    col1, col2 = st.columns(2)
    with col1:
        consider_location = st.checkbox("Consider location in job matching", value=True)
    with col2:
        location_weight = st.slider(
            "Location importance (0-100%)",
            min_value=0,
            max_value=100,
            value=30,
            disabled=not consider_location,
        ) / 100

    user_location = None
    if consider_location:
        user_location = st.text_input(
            "Your location (city, country)",
            help="Enter your current location or preferred job location",
        )

    num_matches = st.slider("Number of top matches to show", min_value=1, max_value=10, value=3)

    if uploaded_file and st.button("Find Matches"):
        pdf_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name

            with st.spinner("Initializing enhanced job matcher..."):
                matcher = JobMatcher()

            with st.spinner("Processing resume and finding matches..."):
                resume_info = matcher.parser.extract_resume_details(pdf_path)
                if not resume_info:
                    st.error("Could not extract any text from the resume.")
                    return

                if consider_location and not user_location:
                    extracted = matcher.parser.extract_location_from_text(resume_info["full_text"])
                    if extracted:
                        user_location = extracted
                        st.info(f"Extracted location from resume: {user_location}")

                try:
                    ensure_database()
                    init_db()
                except Exception as exc:  # noqa: BLE001 - surface DB errors to the user
                    st.error(f"Database unavailable: {exc}")
                    return

                session = get_session()
                try:
                    jobs = fetch_jobs(session, limit=100)
                finally:
                    session.close()

                if not jobs:
                    st.warning("No jobs found in the database. Run the scraper first.")
                    return

                top_matches = matcher.match_jobs(
                    resume_info,
                    jobs,
                    user_location=user_location if consider_location else None,
                    location_weight=location_weight if consider_location else 0.0,
                    top_n=num_matches,
                    build_maps=consider_location,
                )

            st.success(f"Found {len(jobs)} jobs. Showing top {len(top_matches)}.")
            for idx, match in enumerate(top_matches, 1):
                with st.expander(
                    f"#{idx}: {match['job_title']} at {match['company']} - {match['overall_match']}% Match"
                ):
                    cols = st.columns([2, 1])
                    with cols[0]:
                        st.markdown("### Job Details")
                        st.write(f"**Company:** {match['company']}")
                        st.write(f"**Location:** {match['location']}")
                        st.write(f"**Required Experience:** {match['required_experience']}")

                        st.markdown("### Match Metrics")
                        st.write(f"**Overall Match:** {match['overall_match']}%")
                        st.write(f"**Semantic Similarity:** {match['semantic_similarity']}%")
                        st.write(f"**Experience Match:** {match['experience_match']}%")
                        if match["location_score"] is not None:
                            st.write(f"**Location Score:** {match['location_score']}%")
                            if match["distance_km"] is not None:
                                st.write(f"**Distance:** {match['distance_km']} km")

                        if match["matching_skills"]:
                            st.markdown("### Matching Skills")
                            st.write(", ".join(match["matching_skills"]))

                        st.markdown(f"[View on Naukri]({match['url']})")
                    with cols[1]:
                        if match["distance_map"]:
                            st.markdown("### Location Map")
                            from streamlit_folium import st_folium

                            st_folium(match["distance_map"], width=400, height=300)
        except Exception as exc:  # noqa: BLE001 - keep the UI alive
            st.error(f"An error occurred: {exc}")
        finally:
            if pdf_path and os.path.exists(pdf_path):
                os.unlink(pdf_path)


def main() -> None:
    st.set_page_config(page_title="JobRelevancySearcher", layout="wide")
    st.sidebar.title("JobRelevancySearcher")
    choice = st.sidebar.selectbox(
        "Select Functionality",
        ("PDF Generator", "Database and AI Functions"),
    )
    if choice == "PDF Generator":
        run_pdf_generator()
    else:
        run_job_matcher()


if __name__ == "__main__":
    main()

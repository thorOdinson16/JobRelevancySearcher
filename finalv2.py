# Importing necessary libraries
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pdf2image import convert_from_path
from pymongo import MongoClient
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_folium import folium_static
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional
import PyPDF2
import folium
import json
import nltk
import os
import pytesseract
import re
import spacy
import streamlit as st
import tempfile
import torch

# PDF Generator Division

def run_pdf_generator():
    """Contains all functions from pdfgeneratorv1.py"""

    def generate_description(keyword):
        """Generate descriptions based on provided keywords."""
        descriptions = {
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
        # Add more if needed...
        }
        return descriptions.get(keyword.lower(), f"Description not available for '{keyword}'.")

    def create_pdf(data, file_name="resume_with_photo.pdf"):
        """Generate a PDF resume with keyword descriptions and photo."""
        doc = SimpleDocTemplate(file_name, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Header Section
        header = Paragraph(f"<b>{data['name']}</b>", styles['Title'])
        contact_info = Paragraph(f"""
            <br/><b>Contact</b>: {data['phone']} | {data['email']}<br/>
            <b>LinkedIn</b>: {data['linkedin']}<br/>
            <b>Location</b>: {data['location']}
        """, styles['Normal'])
        elements.extend([header, contact_info])

        # Add Photo
        if data['photo']:
            elements.append(Image(data['photo'], width=100, height=100))

        elements.append(Paragraph("<br/><b>CAREER OBJECTIVE</b>", styles['Heading2']))
        elements.append(Paragraph(data['career_objective'], styles['Normal']))

        # Education Section
        elements.append(Paragraph("<br/><b>EDUCATION</b>", styles['Heading2']))
        for edu in data['education']:
            elements.append(Paragraph(f"<b>{edu['degree']}</b> - {edu['institution']} ({edu['year']})", styles['Normal']))

        # Core Competencies Section
        elements.append(Paragraph("<br/><b>CORE COMPETENCIES</b>", styles['Heading2']))
        for comp in data['core_competencies']:
            elements.append(Paragraph(f"- {generate_description(comp)}", styles['Normal']))

        # Internships Section
        elements.append(Paragraph("<br/><b>INTERNSHIPS</b>", styles['Heading2']))
        for internship in data['internships']:
            elements.append(Paragraph(f"<b>{internship['role']}</b> at {internship['organization']} ({internship['duration']})<br/>{internship['description']}", styles['Normal']))

        # Skills Section
        elements.append(Paragraph("<br/><b>SKILLS</b>", styles['Heading2']))
        elements.append(Paragraph(f"<b>Hard Skills</b>: {', '.join(data['hard_skills'])}", styles['Normal']))
        elements.append(Paragraph(f"<b>Soft Skills</b>: {', '.join(data['soft_skills'])}", styles['Normal']))

        # Achievements Section
        elements.append(Paragraph("<br/><b>ACHIEVEMENTS</b>", styles['Heading2']))
        for achievement in data['achievements']:
            elements.append(Paragraph(f"- {achievement}", styles['Normal']))

        # Certifications Section
        elements.append(Paragraph("<br/><b>CERTIFICATIONS</b>", styles['Heading2']))
        for cert in data['certifications']:
            elements.append(Paragraph(f"- {cert}", styles['Normal']))

        # Build PDF
        doc.build(elements)

    def main():
        st.title("Advanced Resume Generator with Keywords and Photo")
        st.write("Fill in the details below to generate your resume with a photo and keyword descriptions.")

        # Input Fields
        name = st.text_input("Full Name")
        phone = st.text_input("Phone Number")
        email = st.text_input("Email Address")
        linkedin = st.text_input("LinkedIn Profile URL")
        location = st.text_input("Location")

        # Photo Upload
        photo = st.file_uploader("Upload a photo for your resume", type=["jpg", "png"])

        st.subheader("Career Objective")
        career_objective = st.text_area("Write your career objective")

        st.subheader("Education")
        num_education = st.number_input("Number of Education Entries", min_value=1, max_value=5, step=1, value=1)
        education = []
        for i in range(int(num_education)):
            st.write(f"Education Entry {i+1}")
            degree = st.text_input(f"Degree (Education {i+1})", key=f"degree_{i}")
            institution = st.text_input(f"Institution (Education {i+1})", key=f"institution_{i}")
            year = st.text_input(f"Year (Education {i+1})", key=f"year_{i}")
            education.append({"degree": degree, "institution": institution, "year": year})

        st.subheader("Core Competencies")
        core_competencies = st.text_area("List core competencies (comma-separated)").split(",")

        st.subheader("Internships")
        num_internships = st.number_input("Number of Internships", min_value=0, max_value=5, step=1, value=1)
        internships = []
        for i in range(int(num_internships)):
            st.write(f"Internship {i+1}")
            role = st.text_input(f"Role (Internship {i+1})", key=f"role_{i}")
            organization = st.text_input(f"Organization (Internship {i+1})", key=f"organization_{i}")
            duration = st.text_input(f"Duration (Internship {i+1})", key=f"duration_{i}")
            description = st.text_area(f"Description (Internship {i+1})", key=f"description_{i}")
            internships.append({"role": role, "organization": organization, "duration": duration, "description": description})

        st.subheader("Skills")
        hard_skills = st.text_input("List hard skills (comma-separated)").split(",")
        soft_skills = st.text_input("List soft skills (comma-separated)").split(",")

        st.subheader("Achievements")
        achievements = st.text_area("List achievements (one per line)").split("\n")

        st.subheader("Certifications")
        certifications = st.text_area("List certifications (one per line)").split("\n")

        # Generate Resume Button
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
                    "core_competencies": [comp.strip() for comp in core_competencies if comp.strip()],
                    "internships": internships,
                    "hard_skills": [skill.strip() for skill in hard_skills if skill.strip()],
                    "soft_skills": [skill.strip() for skill in soft_skills if skill.strip()],
                    "achievements": [ach.strip() for ach in achievements if ach.strip()],
                    "certifications": [cert.strip() for cert in certifications if cert.strip()],
                    "photo": photo
                }
                create_pdf(data, "resume_with_photo.pdf")
                st.success("Resume generated successfully!")
                with open("resume_with_photo.pdf", "rb") as pdf_file:
                    st.download_button(
                        label="Download Resume",
                        data=pdf_file,
                        file_name="resume_with_photo.pdf",
                        mime="application/pdf"
                    )
            else:
                st.error("Please fill in all required fields.")

    if __name__ == "__main__":
        main()

# Database and AI Functions Division

def run_semifinal():
    """Contains all functions from semifinalv1.py"""

    class EnhancedJobMatcher:
        def __init__(self, mongodb_uri="mongodb://localhost:27017/"):
            # Initialize MongoDB connection
            self.client = MongoClient(mongodb_uri)
            self.db = self.client.JobScrapingDB
            self.jobs_collection = self.db.NaukriJobs
        
            # Initialize BERT model
            with st.spinner("Loading BERT model..."):
                self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
            # Initialize spaCy
            with st.spinner("Loading spaCy model..."):
                self.nlp = spacy.load('en_core_web_sm')
        
            # Initialize Nominatim geocoder
            self.geolocator = Nominatim(user_agent="job_matcher")
        
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        
            # Configure Tesseract
            self.tesseract_config = {
                'lang': 'eng',
                'config': '--psm 1'
            }

        def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
            """Extract text from PDF using both direct extraction and OCR if needed."""
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ''
                
                    with st.spinner("Extracting text from PDF..."):
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                
                    if len(text.strip()) < 100:
                        st.warning("Direct extraction yielded limited text. Switching to OCR...")
                        text = self._extract_text_with_ocr(pdf_path)
                
                    return text
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                return self._extract_text_with_ocr(pdf_path)

        def _extract_text_with_ocr(self, pdf_path: str) -> str:
            """Extract text using OCR for scanned PDFs."""
            with st.spinner("Performing OCR on PDF..."):
                images = convert_from_path(pdf_path)
                text = ""
                for image in images:
                    page_text = pytesseract.image_to_string(
                        image,
                        lang=self.tesseract_config['lang'],
                        config=self.tesseract_config['config']
                    )
                    text += page_text + "\n"
                return text

        def extract_resume_details(self, pdf_path: str) -> Dict:
            """Extract detailed information from resume."""
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                return None

            # Process text with spaCy
            doc = self.nlp(text)
        
            # Extract skills and experience
            skills = []
            experience = 0.0
        
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT']:
                    skills.append(ent.text)
            
            # Extract years of experience
            exp_pattern = r'\b(\d+(?:\.\d+)?)\s*(?:\+\s*)?years?\b'
            exp_matches = re.findall(exp_pattern, text, re.IGNORECASE)
            if exp_matches:
                experience = max([float(y) for y in exp_matches])
            
            # Structure the extracted information
            resume_info = {
                'full_text': text,
                'skills': list(set(skills)),
                'experience': experience,
                'extracted_text': self._clean_text(text)
            }
        
            return resume_info

        def _clean_text(self, text: str) -> str:
            """Clean and normalize text."""
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        def extract_location_from_text(self, text: str) -> Optional[str]:
            """Extract location information from text using spaCy."""
            doc = self.nlp(text)
            locations = []
        
            # Extract locations from named entities
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC']:
                    locations.append(ent.text)
                
            return locations[0] if locations else None

        def get_coordinates(self, location_name: str) -> Optional[tuple]:
            """Get coordinates for a location name."""
            try:
                location = self.geolocator.geocode(location_name)
                if location:
                    return (location.latitude, location.longitude)
            except Exception as e:
                st.warning(f"Could not get coordinates for {location_name}: {str(e)}")
            return None

        def fetch_job_postings(self, limit: int = 100) -> List[Dict]:
            """Fetch job postings from MongoDB."""
            return list(self.jobs_collection.find().limit(limit))

        def get_embedding(self, text: str) -> torch.Tensor:
            """Generate embeddings using BERT."""
            inputs = self.tokenizer(str(text), return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).numpy()

        def calculate_location_score(self, user_coords: tuple, job_coords: tuple, max_distance: float = 100) -> float:
            """Calculate location score based on distance."""
            if not user_coords or not job_coords:
                return 0.0
            
            distance = geodesic(user_coords, job_coords).kilometers
            # Inverse relationship: closer locations get higher scores
            return max(0, 1 - (distance / max_distance))

        def create_distance_map(self, user_coords: tuple, job_coords: tuple, user_location: str, job_location: str) -> folium.Map:
            """Create a map showing the distance between user and job location."""
            midpoint = [(user_coords[0] + job_coords[0]) / 2,
                       (user_coords[1] + job_coords[1]) / 2]
        
            m = folium.Map(location=midpoint, zoom_start=4)
        
            # Add markers
            folium.Marker(
                user_coords,
                popup=f"Your Location: {user_location}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
        
            folium.Marker(
                job_coords,
                popup=f"Job Location: {job_location}",
                icon=folium.Icon(color='green', icon='info-sign')
            ).add_to(m)
        
            # Add line between points
            distance = geodesic(user_coords, job_coords).kilometers
            folium.PolyLine(
                locations=[user_coords, job_coords],
                weight=3,
                color='blue',
                opacity=0.8
            ).add_to(m)
        
            # Add distance label
            folium.Marker(
                midpoint,
                popup=f"Distance: {distance:.2f} km",
                icon=folium.DivIcon(html=f'<div style="background-color: white; padding: 5px; border: 1px solid black; border-radius: 5px;">{distance:.2f} km</div>')
            ).add_to(m)
        
            return m

        def calculate_similarity(self, resume_info: Dict, job_posting: Dict, 
                               user_location: str = None, location_weight: float = 0.0) -> Dict:
            """Calculate similarity between resume and job posting with location scoring."""
            # Get embeddings
            resume_embed = self.get_embedding(resume_info['extracted_text'])
        
            # Combine relevant job posting fields
            job_text = f"{job_posting.get('Title', '')} {job_posting.get('Company', '')} {' '.join(job_posting.get('Skills', []))}"
            job_embed = self.get_embedding(job_text)
        
            # Calculate semantic similarity
            semantic_sim = cosine_similarity(resume_embed, job_embed)[0][0]
        
            # Calculate skill overlap
            resume_skills = set(s.lower() for s in resume_info['skills'])
            job_skills = set(s.lower() for s in job_posting.get('Skills', []))
            has_matching_skills = bool(resume_skills & job_skills)
        
            # Experience match
            required_exp = self._normalize_experience(job_posting.get('Experience', "0"))
            candidate_exp = resume_info['experience']
            exp_match = 1.0 if candidate_exp >= required_exp else candidate_exp / required_exp if required_exp else 0.5
        
            # Calculate location score if enabled
            location_score = 0.0
            distance_km = None
            distance_map = None
            if location_weight > 0 and user_location:
                job_location = job_posting.get('Location', '')
                user_coords = self.get_coordinates(user_location)
                job_coords = self.get_coordinates(job_location)
            
                if user_coords and job_coords:
                    location_score = self.calculate_location_score(user_coords, job_coords)
                    distance_km = geodesic(user_coords, job_coords).kilometers
                    distance_map = self.create_distance_map(
                        user_coords, job_coords, user_location, job_location
                    )
        
            # Calculate final score with weights
            semantic_weight = 0.90 * (1 - location_weight)
            skill_weight = 0.05 * (1 - location_weight)
            exp_weight = 0.05 * (1 - location_weight)
        
            final_score = (
                semantic_weight * semantic_sim +
                skill_weight * (1 if has_matching_skills else 0) +
                exp_weight * exp_match +
                location_weight * location_score
            )
        
            return {
                'job_id': job_posting['_id'],
                'job_title': job_posting.get('Title', ''),
                'company': job_posting.get('Company', ''),
                'location': job_posting.get('Location', ''),
                'required_experience': f"{required_exp} years",
                'semantic_similarity': round(semantic_sim * 100, 2),
                'has_matching_skills': has_matching_skills,
                'experience_match': round(exp_match * 100, 2),
                'location_score': round(location_score * 100, 2) if location_weight > 0 else None,
                'distance_km': round(distance_km, 2) if distance_km else None,
                'overall_match': round(final_score * 100, 2),
                'matching_skills': list(resume_skills & job_skills),
                'naukri_url': job_posting.get('Link', '#'),
                'distance_map': distance_map
            }

        def _normalize_experience(self, exp_value: any) -> float:
            """Normalize experience value to float."""
            if isinstance(exp_value, str):
                # Extract the first number from strings like "1-4 Yrs"
                matches = re.findall(r'\d+(?:\.\d+)?', exp_value)
                return float(matches[0]) if matches else 0.0
            elif isinstance(exp_value, (int, float)):
                return float(exp_value)
            return 0.0

    def create_streamlit_app():
        st.title("Enhanced Resume Job Matcher")
        st.write("Upload your resume and find matching jobs with location-based scoring")
    
        # MongoDB connection string input
        mongodb_uri = st.text_input(
            "MongoDB URI",
            value="mongodb://localhost:27017/",
            help="Enter your MongoDB connection string"
        )
    
        # File uploader
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type=['pdf'])
    
        # Location preferences
        col1, col2 = st.columns(2)
        with col1:
            consider_location = st.checkbox("Consider location in job matching", value=True)
        with col2:
            location_weight = st.slider(
                "Location importance (0-100%)",
                min_value=0,
                max_value=100,
                value=30,
                disabled=not consider_location
            ) / 100
    
        user_location = None
        if consider_location:
            user_location = st.text_input(
                "Your location (city, country)",
                help="Enter your current location or preferred job location"
            )
    
        # Number of matches selector
        num_matches = st.slider("Number of top matches to show", min_value=1, max_value=10, value=3)
    
        if uploaded_file and st.button("Find Matches"):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    pdf_path = tmp_file.name
            
                # Initialize matcher
                with st.spinner("Initializing enhanced job matcher..."):
                    matcher = EnhancedJobMatcher(mongodb_uri)
                
                # Extract resume info and get matches
                with st.spinner("Processing resume and finding matches..."):
                    resume_info = matcher.extract_resume_details(pdf_path)
                
                    # If location not provided, try to extract from resume
                    if consider_location and not user_location:
                        extracted_location = matcher.extract_location_from_text(resume_info['full_text'])
                        if extracted_location:
                            user_location = extracted_location
                            st.info(f"Extracted location from resume: {user_location}")
                
                    # Fetch and process job postings
                    job_postings = matcher.fetch_job_postings()
                    matches = []
                
                    progress_bar = st.progress(0)
                    for i, job in enumerate(job_postings):
                        match = matcher.calculate_similarity(
                            resume_info,
                            job,
                            user_location if consider_location else None,
                            location_weight if consider_location else 0.0
                        )
                        matches.append(match)
                        progress_bar.progress((i + 1) / len(job_postings))
                
                    # Sort matches by overall match score
                    matches.sort(key=lambda x: x['overall_match'], reverse=True)
                    top_matches = matches[:num_matches]

                    # Display results
                    st.success(f"Found {len(matches)} potential matches. Showing top {num_matches}")
                
                    for idx, match in enumerate(top_matches, 1):
                        with st.expander(f"#{idx}: {match['job_title']} at {match['company']} - {match['overall_match']}% Match"):
                            cols = st.columns([2, 1])
                        
                            with cols[0]:
                                st.markdown(f"### Job Details")
                                st.write(f"**Company:** {match['company']}")
                                st.write(f"**Location:** {match['location']}")
                                st.write(f"**Required Experience:** {match['required_experience']}")
                            
                                st.markdown("### Match Metrics")
                                st.write(f"**Overall Match:** {match['overall_match']}%")
                                st.write(f"**Semantic Similarity:** {match['semantic_similarity']}%")
                                st.write(f"**Experience Match:** {match['experience_match']}%")
                            
                                if match['location_score'] is not None:
                                    st.write(f"**Location Score:** {match['location_score']}%")
                                    if match['distance_km']:
                                        st.write(f"**Distance:** {match['distance_km']} km")
                            
                                if match['matching_skills']:
                                    st.markdown("### Matching Skills")
                                    st.write(", ".join(match['matching_skills']))
                            
                                st.markdown(f"[View on Naukri]({match['naukri_url']})")
                        
                            with cols[1]:
                                if match['distance_map']:
                                    st.markdown("### Location Map")
                                    folium_static(match['distance_map'])
                
                # Clean up temporary file
                os.unlink(pdf_path)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                if 'pdf_path' in locals():
                    os.unlink(pdf_path)

    if __name__ == "__main__":
        create_streamlit_app()
# Main Menu for Selecting Functionalities

def main():
    import streamlit as st

    st.title("Choose Your Functionality")
    choice = st.sidebar.selectbox(
        "Select Functionality",
        ("PDF Generator", "Database and AI Functions")
    )

    if choice == "PDF Generator":
        run_pdf_generator()
    elif choice == "Database and AI Functions":
        run_semifinal()

if __name__ == "__main__":
    main()

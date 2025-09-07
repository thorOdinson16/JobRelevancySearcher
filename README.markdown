# JobRelevancySearcher: AI-Powered Job Matching Platform

## Overview

JobRelevancySearcher is an advanced, AI-driven platform designed to streamline the job search process. It integrates real-time job scraping, semantic search, location-based filtering, and intelligent resume parsing to deliver highly relevant job matches tailored to your skills and preferences. Powered by Streamlit and hosted on the cloud, JobRelevancySearcher provides a modern, user-friendly experience for job seekers.

## Key Features

- **Real-Time Job Scraping**  
  Pulls the latest job postings from [Naukri.com](https://www.naukri.com), ensuring access to current opportunities across industries.

- **Semantic Job Matching**  
  Employs Sentence-BERT (SBERT) to perform intelligent, semantic-based matching, aligning job postings with your skills, experience, and career goals.

- **Location-Based Optimization**  
  Uses Geopy to prioritize jobs near your preferred location, making your job search more practical and commute-friendly.

- **Smart Resume Parsing**  
  Analyzes your resume to extract key skills and experiences, matching them with job description keywords for optimized applications.

- **Streamlit-Powered Interface**  
  Delivers a clean, interactive UI built with Streamlit, deployed on the cloud for seamless access from any device.

## Tech Stack

- **Web Scraping**: Python, BeautifulSoup/Scrapy
- **Semantic Search**: Sentence-BERT (SBERT)
- **Geolocation**: Geopy
- **Frontend & Deployment**: Streamlit, Cloud Hosting
- **Backend**: Python
- **Dependencies**: `sentence-transformers`, `geopy`, `beautifulsoup4`, `requests`

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Streamlit
- pip for installing dependencies

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/thorOdinson16/JobRelevancySearcher.git
   ```
2. Navigate to the project directory:
   ```bash
   cd JobRelevancySearcher
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the application:
   ```bash
   streamlit run app.py
   ```

### Cloud Deployment
1. Push the repository to a cloud platform (e.g., Streamlit Cloud, Heroku).
2. Configure environment variables for any API keys or credentials.
3. Access the app via the provided URL.

## How to Use

1. **Enter Your Profile**: Input your skills, experience, and preferred job location.
2. **Upload Resume**: Upload your resume to extract and match relevant skills with job requirements.
3. **Browse Matches**: Explore job listings ranked by relevance and proximity.
4. **Apply Easily**: Use parsed resume data to create tailored applications directly through the platform.

## Future Enhancements

- Expand integration with additional job boards (e.g., LinkedIn, Indeed).
- Add real-time resume optimization suggestions.
- Support multiple languages for global job markets.
- Introduce personalized dashboards to track job applications and progress.

# JobRelevancySearcher: AI-Powered Job Matching Platform

## Overview

JobRelevancySearcher is an AI-driven platform that streamlines the job search
process. It scrapes live job postings from [Naukri.com](https://www.naukri.com),
stores them in PostgreSQL, and ranks them against an uploaded resume using
sentence embeddings with optional location-aware scoring. A built-in resume PDF
generator rounds out the toolkit.

## Features

- **Real-time job scraping** from Naukri.com via Selenium, deduplicated on upsert.
- **Semantic job matching** using a Sentence-Transformers model
  (`all-MiniLM-L6-v2`).
- **Location-based scoring** with Geopy distance decay and an interactive
  Folium map.
- **Smart resume parsing** (text extraction + OCR fallback) with spaCy entity
  recognition.
- **Resume PDF generator** with keyword-driven competency descriptions.
- **Streamlit UI** for both modes.

## Tech Stack

- **Frontend**: Streamlit, streamlit-folium, Folium
- **Database**: PostgreSQL (SQLAlchemy 2.0 ORM + psycopg2)
- **Scraping**: Selenium
- **NLP/ML**: Transformers, spaCy, scikit-learn, PyTorch
- **PDF**: pypdf, pdf2image, pytesseract, reportlab
- **Geo**: geopy, Nominatim

## Project Structure

```
app.py                     # Streamlit entrypoint
config.py                  # Environment-driven configuration
db.py                      # Engine, schema bootstrap, upsert helpers
src/
  models.py                # Job ORM model
  schema.py                # Shared job field constants
  scraper.py               # Naukri scraper -> PostgreSQL
  resume_parser.py         # PDF/OCR text extraction + parsing
  matcher.py               # Embeddings, geocoding, scoring
  resume_generator.py      # Resume PDF generation
tests/                     # pytest suite
requirements.txt           # Runtime dependencies
requirements-dev.txt       # Dev dependencies
```

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL running locally (default `localhost:5432`)
- System tools (only needed for their respective features):
  - **Tesseract OCR** — OCR of scanned resumes
  - **Poppler** — PDF rasterisation for OCR
  - **Google Chrome + ChromeDriver** — scraping

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/thorOdinson16/JobRelevancySearcher.git
   cd JobRelevancySearcher
   ```
2. Create the environment and install dependencies:
   ```bash
   bash setup.sh
   # or, on Windows:
   python -m venv .venv --system-site-packages
   .venv\Scripts\activate
   pip install -r requirements.txt -r requirements-dev.txt
   python -m spacy download en_core_web_sm
   ```

### Configuration

Connection settings are read from environment variables (or Streamlit secrets):

| Variable       | Default       |
| -------------- | ------------- |
| `PGHOST`       | `localhost`   |
| `PGPORT`       | `5432`        |
| `PGUSER`       | `postgres`    |
| `PGPASSWORD`   | `psql123`     |
| `PGDATABASE`   | `jobrelevancy`|
| `DATABASE_URL` | full override |

The application creates the `jobrelevancy` database and its tables on first use.

### Scrape Jobs

```bash
python -m src.scraper
```

You will be prompted for a role, location, and number of jobs. Listings are
inserted or updated (deduplicated by URL) in PostgreSQL.

### Run the App

```bash
streamlit run app.py
```

Select **PDF Generator** or **Database and AI Functions** from the sidebar.

### Tests

```bash
pytest
ruff check .
```

## How to Use

1. **Generate a resume** from the PDF Generator tab, or
2. **Find matches**: upload a resume PDF, optionally enable location scoring,
   and browse jobs ranked by relevance.

## Future Enhancements

- Integrations with additional job boards (LinkedIn, Indeed).
- Real-time resume optimization suggestions.
- Multi-language support.
- Personalized application-tracking dashboards.

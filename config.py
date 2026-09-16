"""Central configuration.

Values are read from environment variables so that credentials are never
hard-coded in application logic. ``python-dotenv`` is intentionally avoided to
keep the dependency surface small; set the variables in your shell or via
Streamlit secrets (see ``app.py``).
"""

from __future__ import annotations

import os

# --- PostgreSQL -----------------------------------------------------------
PG_HOST = os.environ.get("PGHOST", "localhost")
PG_PORT = os.environ.get("PGPORT", "5432")
PG_USER = os.environ.get("PGUSER", "postgres")
PG_PASSWORD = os.environ.get("PGPASSWORD", "psql123")
PG_DATABASE = os.environ.get("PGDATABASE", "jobrelevancy")

# A full SQLAlchemy URL always wins over the individual components above.
DATABASE_URL = os.environ.get("DATABASE_URL") or (
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
)
ADMIN_DATABASE_URL = os.environ.get("ADMIN_DATABASE_URL") or (
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/postgres"
)

# --- Models ---------------------------------------------------------------
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")

# --- Matching heuristics --------------------------------------------------
MAX_DISTANCE_KM = float(os.environ.get("MAX_DISTANCE_KM", "100"))
DEFAULT_NUM_JOBS = int(os.environ.get("DEFAULT_NUM_JOBS", "50"))


def get_database_url() -> str:
    """Return the SQLAlchemy URL for the application database."""
    return DATABASE_URL


def get_admin_database_url() -> str:
    """Return the SQLAlchemy URL used to bootstrap the database itself."""
    return ADMIN_DATABASE_URL

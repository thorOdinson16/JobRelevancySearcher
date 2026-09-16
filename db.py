"""Database engine, schema bootstrap and job upsert helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

import config

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


def _create_engine(url: str) -> Engine:
    connect_args = {}
    if url.startswith("postgresql"):
        # psycopg2 otherwise waits forever if the server disappears.
        connect_args = {"connect_timeout": 10}
    return create_engine(url, pool_pre_ping=True, future=True, connect_args=connect_args)


engine: Engine = _create_engine(config.get_database_url())
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def ensure_database() -> None:
    """Create the application database if it does not yet exist.

    Uses a separate connection to the ``postgres`` maintenance database with
    autocommit so that ``CREATE DATABASE`` (which cannot run in a transaction)
    succeeds. Non-PostgreSQL URLs are ignored so tests can use SQLite.
    """
    url = config.get_database_url()
    if not url.startswith("postgresql"):
        return

    admin_engine = create_engine(
        config.get_admin_database_url(),
        isolation_level="AUTOCOMMIT",
        future=True,
    )
    db_name = config.PG_DATABASE
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": db_name},
            ).scalar()
            if not exists:
                # Identifier cannot be parameterised; db_name comes from config.
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
                logger.info("Created database %s", db_name)
    finally:
        admin_engine.dispose()


def init_db(bound_engine: Engine | None = None) -> None:
    """Create all tables. Safe to call repeatedly."""
    from src import models  # noqa: F401  (register models on Base.metadata)

    target = bound_engine or engine
    Base.metadata.create_all(target)


def get_session() -> Session:
    """Return a new ORM session. Caller is responsible for closing it."""
    return SessionLocal()


# Columns refreshed when a job with the same link is scraped again.
_UPSERT_COLUMNS = (
    "title",
    "company",
    "location",
    "experience",
    "salary",
    "skills",
    "description",
    "role",
    "industry",
    "employment_type",
    "status",
    "error",
    "scraped_at",
)


def upsert_jobs(session: Session, jobs: Iterable[Dict[str, Any]]) -> int:
    """Insert jobs, updating existing rows that share the same link.

    Returns the number of rows processed. Works on PostgreSQL (native
    ``ON CONFLICT``) and falls back to a select/merge strategy elsewhere so the
    same code path is testable on SQLite.
    """
    jobs = [job for job in jobs if job.get("link")]
    if not jobs:
        return 0

    from src.models import Job

    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(Job).values(jobs)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Job.link],
            set_={col: getattr(stmt.excluded, col) for col in _UPSERT_COLUMNS},
        )
        session.execute(stmt)
    else:
        for job in jobs:
            existing = session.query(Job).filter_by(link=job["link"]).one_or_none()
            if existing is None:
                session.add(Job(**job))
            else:
                for col in _UPSERT_COLUMNS:
                    if col in job:
                        setattr(existing, col, job[col])
    session.commit()
    return len(jobs)


def fetch_jobs(session: Session, limit: int = 100) -> List[Any]:
    """Return up to ``limit`` job rows ordered by most recently scraped."""
    from src.models import Job

    return (
        session.query(Job)
        .order_by(Job.scraped_at.desc().nullslast())
        .limit(limit)
        .all()
    )

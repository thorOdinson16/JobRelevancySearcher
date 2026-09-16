"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db import Base


@pytest.fixture()
def sqlite_session():
    """An isolated in-memory SQLite session with the schema created."""
    from src import models  # noqa: F401  (register models)

    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()

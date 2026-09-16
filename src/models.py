"""ORM models for the job store."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from db import Base


class Job(Base):
    """A scraped job posting.

    ``link`` is unique so re-scraping updates the existing row instead of
    creating duplicates.
    """

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(512), default="")
    company: Mapped[str] = mapped_column(String(512), default="")
    location: Mapped[str] = mapped_column(String(512), default="")
    experience: Mapped[str] = mapped_column(String(128), default="")
    salary: Mapped[str] = mapped_column(String(256), default="")
    link: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    skills: Mapped[List[str]] = mapped_column(JSON, default=list)
    description: Mapped[str] = mapped_column(Text, default="")
    role: Mapped[str] = mapped_column(Text, default="")
    industry: Mapped[str] = mapped_column(String(512), default="")
    employment_type: Mapped[str] = mapped_column(String(128), default="")
    status: Mapped[str] = mapped_column(String(32), default="Success")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def to_dict(self) -> dict:
        """Return a plain dict for consumers that do not want ORM objects."""
        return {
            "id": self.id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "experience": self.experience,
            "salary": self.salary,
            "link": self.link,
            "skills": self.skills or [],
            "description": self.description,
            "role": self.role,
            "industry": self.industry,
            "employment_type": self.employment_type,
            "status": self.status,
            "scraped_at": self.scraped_at.isoformat() if self.scraped_at else None,
        }

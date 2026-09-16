"""Semantic job matching with optional location weighting.

Ported from the ``EnhancedJobMatcher`` scoring half of ``finalv2.py``. Key
improvements:

* the resume embedding is computed once per match run instead of once per job;
* geocoding results are cached, avoiding a Nominatim request per job;
* distance maps are only built on demand;
* the wrong ``-> torch.Tensor`` type hint and ``Any`` annotation are fixed;
* works with ORM ``Job`` rows, dicts, or anything exposing the same attributes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from src.resume_parser import ResumeParser, normalize_experience

logger = logging.getLogger(__name__)

Coordinates = Tuple[float, float]


class JobMatcher:
    """Score jobs against a parsed resume using sentence embeddings."""

    def __init__(self, parser: Optional[ResumeParser] = None) -> None:
        self.parser = parser or ResumeParser()
        self._tokenizer = None
        self._model = None
        self._geocode_cache: Dict[str, Optional[Coordinates]] = {}
        self._geolocator = None

    # -- lazy heavy resources ---------------------------------------------
    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
        return self._tokenizer

    @property
    def model(self) -> Any:
        if self._model is None:
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(config.EMBEDDING_MODEL)
        return self._model

    @property
    def geolocator(self) -> Any:
        if self._geolocator is None:
            from geopy.geocoders import Nominatim

            self._geolocator = Nominatim(user_agent="job_relevancy_searcher")
        return self._geolocator

    # -- embeddings --------------------------------------------------------
    def get_embedding(self, text: str) -> np.ndarray:
        """Mean-pooled token embedding for ``text`` as a 2-D numpy array."""
        import torch

        inputs = self.tokenizer(str(text), return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    # -- geocoding ---------------------------------------------------------
    def get_coordinates(self, location_name: Optional[str]) -> Optional[Coordinates]:
        """Cached coordinate lookup for a place name."""
        if not location_name:
            return None
        key = location_name.strip().lower()
        if key not in self._geocode_cache:
            try:
                result = self.geolocator.geocode(location_name)
                self._geocode_cache[key] = (
                    (result.latitude, result.longitude) if result else None
                )
            except Exception as exc:  # noqa: BLE001 - network/geocoder errors
                logger.warning("Geocoding failed for %s: %s", location_name, exc)
                self._geocode_cache[key] = None
        return self._geocode_cache[key]

    @staticmethod
    def calculate_location_score(
        user_coords: Optional[Coordinates],
        job_coords: Optional[Coordinates],
        max_distance: float = config.MAX_DISTANCE_KM,
    ) -> float:
        """Linear decay from 1.0 (same place) to 0.0 at ``max_distance``."""
        if not user_coords or not job_coords:
            return 0.0
        from geopy.distance import geodesic

        distance = geodesic(user_coords, job_coords).kilometers
        return max(0.0, 1 - (distance / max_distance))

    @staticmethod
    def create_distance_map(
        user_coords: Coordinates,
        job_coords: Coordinates,
        user_location: str,
        job_location: str,
    ) -> Any:
        """Build a folium map connecting user and job locations."""
        import folium
        from geopy.distance import geodesic

        midpoint = [
            (user_coords[0] + job_coords[0]) / 2,
            (user_coords[1] + job_coords[1]) / 2,
        ]
        distance = geodesic(user_coords, job_coords).kilometers
        fmap = folium.Map(location=midpoint, zoom_start=4)
        folium.Marker(
            user_coords,
            popup=f"Your Location: {user_location}",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(fmap)
        folium.Marker(
            job_coords,
            popup=f"Job Location: {job_location}",
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(fmap)
        folium.PolyLine([user_coords, job_coords], weight=3, color="blue", opacity=0.8).add_to(fmap)
        folium.Marker(
            midpoint,
            popup=f"Distance: {distance:.2f} km",
            icon=folium.DivIcon(
                html=(
                    '<div style="background-color:white;padding:5px;'
                    f'border:1px solid black;border-radius:5px;">{distance:.2f} km</div>'
                )
            ),
        ).add_to(fmap)
        return fmap

    # -- scoring -----------------------------------------------------------
    @staticmethod
    def _field(job: Any, name: str, default: Any = "") -> Any:
        if isinstance(job, dict):
            return job.get(name, default)
        return getattr(job, name, default)

    def calculate_similarity(
        self,
        resume_info: Dict[str, Any],
        job: Any,
        resume_embedding: Optional[np.ndarray] = None,
        user_coords: Optional[Coordinates] = None,
        location_weight: float = 0.0,
        user_location: str = "",
        build_map: bool = False,
    ) -> Dict[str, Any]:
        """Score a single job. Pass ``resume_embedding`` to avoid recomputing it."""
        if resume_embedding is None:
            resume_embedding = self.get_embedding(resume_info["extracted_text"])

        skills = self._field(job, "skills", []) or []
        job_text = " ".join(
            [
                str(self._field(job, "title")),
                str(self._field(job, "company")),
                " ".join(str(s) for s in skills),
            ]
        )
        job_embedding = self.get_embedding(job_text)
        semantic_sim = float(cosine_similarity(resume_embedding, job_embedding)[0][0])

        resume_skills = {s.lower() for s in resume_info.get("skills", [])}
        job_skills = {str(s).lower() for s in skills}
        matching_skills = sorted(resume_skills & job_skills)

        required_exp = normalize_experience(self._field(job, "experience", "0"))
        candidate_exp = resume_info.get("experience", 0.0)
        if required_exp:
            exp_match = 1.0 if candidate_exp >= required_exp else candidate_exp / required_exp
        else:
            exp_match = 0.5

        location_score = 0.0
        distance_km = None
        distance_map = None
        if location_weight > 0 and user_coords:
            job_location = self._field(job, "location", "")
            job_coords = self.get_coordinates(job_location)
            if job_coords:
                location_score = self.calculate_location_score(user_coords, job_coords)
                from geopy.distance import geodesic

                distance_km = geodesic(user_coords, job_coords).kilometers
                if build_map:
                    distance_map = self.create_distance_map(
                        user_coords, job_coords, user_location, job_location
                    )

        semantic_weight = 0.90 * (1 - location_weight)
        skill_weight = 0.05 * (1 - location_weight)
        exp_weight = 0.05 * (1 - location_weight)
        final_score = (
            semantic_weight * semantic_sim
            + skill_weight * (1.0 if matching_skills else 0.0)
            + exp_weight * exp_match
            + location_weight * location_score
        )

        return {
            "job_id": self._field(job, "id", None),
            "job_title": self._field(job, "title", ""),
            "company": self._field(job, "company", ""),
            "location": self._field(job, "location", ""),
            "required_experience": f"{required_exp} years",
            "semantic_similarity": round(semantic_sim * 100, 2),
            "has_matching_skills": bool(matching_skills),
            "experience_match": round(exp_match * 100, 2),
            "location_score": round(location_score * 100, 2) if location_weight > 0 else None,
            "distance_km": round(distance_km, 2) if distance_km is not None else None,
            "overall_match": round(final_score * 100, 2),
            "matching_skills": matching_skills,
            "url": self._field(job, "link", "#"),
            "distance_map": distance_map,
        }

    def match_jobs(
        self,
        resume_info: Dict[str, Any],
        jobs: List[Any],
        user_location: Optional[str] = None,
        location_weight: float = 0.0,
        top_n: int = 3,
        build_maps: bool = False,
    ) -> List[Dict[str, Any]]:
        """Score every job and return the top ``top_n`` by overall match."""
        if not resume_info:
            return []

        resume_embedding = self.get_embedding(resume_info["extracted_text"])
        user_coords = self.get_coordinates(user_location) if user_location else None

        matches = [
            self.calculate_similarity(
                resume_info,
                job,
                resume_embedding=resume_embedding,
                user_coords=user_coords,
                location_weight=location_weight,
                user_location=user_location or "",
                build_map=build_maps,
            )
            for job in jobs
        ]
        matches.sort(key=lambda m: m["overall_match"], reverse=True)
        return matches[:top_n]
